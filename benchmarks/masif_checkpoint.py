"""Strict MaSIF-search checkpoint checks; compatible with the TF1 container."""
import os
import numpy as np


def resolve_checkpoint(prefix=None):
    return prefix or os.environ.get('MASIF_CHECKPOINT_PREFIX',
        '/masif/data/masif_ppi_search/nn_models/sc05/all_feat/model_data/model')


def normalize_paper_electrostatics(values):
    """Original paper feature scaling, after the PLY potential /10 scaling.

    The later author-container implementation accidentally repeats the upper
    clip and omits the lower clip. The paper trained with both bounds.
    """
    return np.clip(np.asarray(values),-3.,3.)/3.


def expected_shapes():
    result = {}
    for feature in range(5):
        for name in ('mu_rho', 'mu_theta', 'sigma_rho', 'sigma_theta'):
            result['{}_{}'.format(name, feature)] = (1, 80)
        result['W_conv_{}'.format(feature)] = (80, 80)
        result['b_conv_{}'.format(feature)] = (80,)
    result['fully_connected/weights'] = (400, 80)
    result['fully_connected/biases'] = (80,)
    return result


def validate_arrays(arrays):
    """Reject missing, incompatible or nonfinite weights before inference.

    Check all supplied floating tensors, including optimizer slots when reading
    a full checkpoint. Never replace bad weights with zeros or random values.
    """
    expected = expected_shapes()
    missing = sorted(set(expected) - set(arrays))
    if missing:
        raise ValueError('MaSIF checkpoint missing tensors: '+', '.join(missing))
    incompatible = [name for name, shape in sorted(expected.items())
                    if tuple(np.shape(arrays[name])) != shape]
    if incompatible:
        raise ValueError('MaSIF checkpoint incompatible tensor shapes: '+', '.join(incompatible))
    bad = [name for name, array in sorted(arrays.items())
           if np.issubdtype(np.asarray(array).dtype, np.inexact)
           and not np.isfinite(array).all()]
    if bad:
        raise ValueError('MaSIF checkpoint has nonfinite tensors: '+', '.join(bad))
    nonfloating = [name for name in expected
                   if not np.issubdtype(np.asarray(arrays[name]).dtype, np.floating)]
    if nonfloating:
        raise ValueError('MaSIF checkpoint requires floating weights: '+', '.join(sorted(nonfloating)))
    return dict(tensors_checked=len(arrays), model_tensors=len(expected),
                model_parameters=sum(int(np.prod(shape)) for shape in expected.values()),
                parameters_finite=True)


def validate_checkpoint(prefix):
    import tensorflow as tf
    reader = tf.train.NewCheckpointReader(str(prefix))
    arrays = {name: reader.get_tensor(name)
              for name in reader.get_variable_to_shape_map()}
    return validate_arrays(arrays)
