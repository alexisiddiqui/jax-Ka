"""PyTorch input gradients through continuous structural features and oGQT."""
from __future__ import annotations
import torch
from jaxpropka.torch import TorchInputsReadout
from .compilation import configure_compilation_cache
from .differentiable import ATOM_NAMES


def _gate(value, enabled, mask):
    if not enabled: return value.detach()
    if mask is None: return value
    if not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool:
        raise TypeError('Gradient masks must be boolean Torch tensors')
    mask = mask.to(value.device)
    if mask.ndim > value.ndim: raise ValueError('Gradient mask has too many dimensions')
    # Residue/atom masks are prefix-shaped: [N] or [N,atoms].
    mask = mask.reshape(*mask.shape, *((1,)*(value.ndim-mask.ndim)))
    try: mask = torch.broadcast_to(mask,value.shape)
    except RuntimeError as exc: raise ValueError('Gradient mask does not match input shape') from exc
    return torch.where(mask, value, value.detach())


class TorchStructureReadout(torch.nn.Module):
    """Wrap ``predict_graph(featurise(sequence, backbone, sidechains))``.

    The callable may implement any model with the same feature contract. Fixed
    parameters are captured; this module exposes input gradients, not parameter
    training. A supplied custom featuriser must itself be JAX-differentiable.
    Masks affect gradients only. Disabling a group detaches that input; individual
    masked entries receive zero gradients. This is first-order reverse mode.
    """
    def __init__(self, features, predict_graph, *, compilation_cache=None, jit=True):
        super().__init__()
        if compilation_cache is not False: configure_compilation_cache(compilation_cache)
        self.features = features
        self.readout = TorchInputsReadout(lambda sequence, backbone, sidechains:
            predict_graph(features(sequence,backbone,sidechains)), jit=jit)

    @classmethod
    def from_ogqt(cls, params, features, *, output='pka', **kwargs):
        """Standard 24/152-feature oGQT; portable JAX forward and full VJP.

        Use the generic constructor for other model/input variants. The existing
        Triton VJP omits switch derivatives, so it is not used for coordinates.
        """
        import jax
        import jax.numpy as jnp
        params = jax.tree.map(jnp.asarray, params)
        from .ogqt import predict_with_trace
        from .model import PKPDB_PK_MOD
        expected = 152 if features.sidechains else 24
        if params['embed']['w'].shape[0] != expected:
            raise ValueError(f'Model input width does not match {expected}-feature template')
        if params['query']['edge']['w'].shape[0] != 20:
            raise ValueError('Orientation-edge variants require their own model/feature adapter')
        if output not in ('pka','shift'): raise ValueError('output must be pka or shift')
        def predict(graph):
            shift = predict_with_trace(params, graph)['predicted_shift']
            return shift if output=='shift' else shift+PKPDB_PK_MOD[graph['query_group']]
        return cls(features,predict,**kwargs)

    @classmethod
    def from_default(cls, features, *, state='bound', output='pka', model_path=None, **kwargs):
        """Pinned frozen relational GQT, with gradients to structural inputs.

        Pass a backbone-only StructureFeatures template. Free-state attention is
        restricted to each chain; bound-state attention includes partner chains.
        """
        from pkatrain.production_defaults import load_frozen_default
        from .relational import RelationalStructureFeatures
        from .model import PKPDB_PK_MOD
        if output not in ('pka', 'shift'):
            raise ValueError('output must be pka or shift')
        adapter = RelationalStructureFeatures(features, state)
        _, shift, metadata = load_frozen_default(model_path, input_gradients=True)
        def predict(graph):
            value = shift(graph)
            return value if output == 'shift' else value + PKPDB_PK_MOD[graph['query_group']]
        result = cls(adapter, predict, **kwargs)
        result.model_metadata = metadata
        return result

    def forward(self, sequence, backbone, sidechains, *,
                gradients=('sequence','backbone','sidechains'),
                sequence_mask=None, backbone_mask=None, sidechain_mask=None):
        groups = frozenset(gradients)
        if not groups <= {'sequence','backbone','sidechains'}:
            raise ValueError('Unknown gradient group')
        return self.readout(_gate(sequence,'sequence' in groups,sequence_mask),
            _gate(backbone,'backbone' in groups,backbone_mask),
            _gate(sidechains,'sidechains' in groups,sidechain_mask))

    def forward_atoms(self, sequence, atoms, **kwargs):
        """Named heavy atoms [N,36,3], ordered by ATOM_NAMES; split gradients recombine."""
        if atoms.ndim != 3 or atoms.shape[1:] != (len(ATOM_NAMES),3):
            raise ValueError('Expected named heavy atoms [N,36,3] in ATOM_NAMES order')
        return self(sequence,atoms[:,:4],atoms[:,4:],**kwargs)
