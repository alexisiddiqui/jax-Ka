"""Standalone JSON worker runnable inside isolated method environments."""
import importlib.metadata
import json
import os
from pathlib import Path
import sys


def main():
    from pkabench.runtime import require_compute, atomic_json
    require_compute()
    request = json.loads(Path(sys.argv[1]).read_text())
    method = request["method"]
    rows = []; extra = {}
    if method == "pkai_training_probe":
        from importlib.util import find_spec
        sys.path.insert(0,str(Path(find_spec("pkai").origin).parent))
        import torch
        from pKAI import load_model
        from protein import Protein
        torch.set_num_threads(1)
        protein=Protein(request["pdb"]); protein.apply_cutoff()
        x=torch.stack([r.input_layer for r in protein.iter_residues(titrable_only=True)])
        results={}
        for name in ("pKAI","pKAI+"):
            model=load_model(name,"cpu"); model.train()
            parameters=[p for p in model.parameters() if p.requires_grad]
            before=[p.detach().clone() for p in parameters]
            optimizer=torch.optim.SGD(parameters,lr=1e-4)
            optimizer.zero_grad(); prediction=model(x)
            loss=((prediction-(prediction.detach()+.1))**2).mean()
            loss.backward()
            finite=all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in parameters)
            optimizer.step()
            changed=any(not torch.equal(a,p.detach()) for a,p in zip(before,parameters))
            results[name]={"finite_gradients":finite,"parameters_changed":changed,"loss":float(loss.detach())}
            if not finite or not changed: raise ValueError(f"training probe failed for {name}")
        atomic_json(sys.argv[2],{"status":"pass","models":results,"persisted_weights":False})
        return
    if method == "pypka":
        from pypka import Titration
        from pypka.config import Config
        params = dict(request["config"])
        params.update(structure=request["pdb"], ncpus=1, pH="-2,16", pHstep=.25,
            save_mc_energies="mc-energies.json", output="midpoints.txt", titration_output="curves.txt")
        model = Titration(params)
        for site in model:
            curve = site.getTitrationCurve()
            # This API exposes native Monte Carlo occupancies, unlike getAverageProt.
            values = [float(curve[round(-2+i*.25, 2)]) for i in range(73)]
            rows.append({"chain": site.molecule.chain, "resnum": site.getResNumber(),
                "group": {"NTR":"NTERM", "CTR":"CTERM"}.get(site.res_name, site.res_name),
                "pka": site.getpK(), "curve": values, "curve_source": "native",
                "intrinsic_tautomers":{t.name:float(t.pKint) for t in site.iterTautomers() if hasattr(t,"pKint")}})
        extra["intermediates"] = "mc-energies.json" if Path("mc-energies.json").exists() else None
        extra["resolved_config"] = {name: dict(getattr(Config, name).get_clean_params()) for name in ("pypka_params", "delphi_params", "mc_params")}
        # Native interaction data uses tautomer states. Retain it verbatim rather
        # than misrepresenting it as a single scalar per pair of titratable sites.
        version = importlib.metadata.version("pypka")
    elif method in ("pkai", "pkai_plus"):
        from importlib.util import find_spec
        package=find_spec("pkai")
        sys.path.insert(0, str(Path(package.origin).parent))
        from pKAI import pKAI
        results = pKAI(request["pdb"], model_name="pKAI" if method == "pkai" else "pKAI+", device="cpu", threads=1)
        for chain, number, residue, value in results:
            rows.append({"chain": str(chain), "resnum": int(number), "group": str(residue), "pka": float(value)})
        version = importlib.metadata.version("pKAI")
    elif method == "propka":
        from propka.run import single
        # Request file is already canonical and mapped by the runner.
        model = single(request["pdb"], write_pka=True)
        from jaxpropka.reference import parse_pka
        files = list(Path(".").glob("*.pka"))
        if len(files) != 1: raise ValueError("expected one PROPKA output")
        for site in parse_pka(files[0].read_text()):
            rows.append({"chain":site.key.chain,"resnum":site.key.number,"group":site.group,"pka":site.pka})
        version = importlib.metadata.version("propka")
    elif method == "jaxka":
        import numpy as np
        from jaxpropka import TitrationModel, prepare
        from pkabench.prep import read_cif
        from pkabench.schema import GROUPS, PH
        cache = prepare(read_cif(request["cif"]), topology_options={"gap_policy":"cap", "freeze_disulfides":True}, geometry_options={"missing_sidechain":"error"})
        model = TitrationModel(cache, backend="packed")
        curves = model.curves(PH)(model.native_probabilities)
        midpoints = model.pka_from_grid(PH)(model.native_probabilities)
        for i, key in enumerate(cache.keys):
            for g, group in enumerate(GROUPS):
                if float(curves.probability[i,g]) == 0:
                    if group=="CYS" and bool(cache.frozen[i]):
                        rows.append({"chain":key.chain,"resnum":key.number,"icode":key.insertion,"group":group,
                            "pka":None,"status":"not_titrating","curve":[0.]*73,"curve_source":"native"})
                    continue
                status = "ok" if bool(midpoints.valid[i,g]) else "out_of_range" if not bool(midpoints.bracketed[i,g]) else "failed"
                if not np.asarray(curves.converged).all(): status = "failed"
                rows.append({"chain":key.chain,"resnum":key.number,"icode":key.insertion,"group":group,
                    "pka":float(midpoints.value[i,g]) if bool(midpoints.valid[i,g]) else None,
                    "status":status,"curve":np.asarray(curves.protonated[:,i,g]).tolist() if status != "failed" else None,
                    "curve_source":"native","intrinsic_pka":float(curves.intrinsic_pka[i,g])})
        version = importlib.metadata.version("jax-propka")
        extra["max_residual"] = float(np.max(curves.residual))
    elif method == "kaml":
        raise RuntimeError("G3: released KaML-CBtree supports only single-chain inputs; excluded until a joint-system gate passes")
    else:
        raise ValueError(f"unknown worker method {method}")
    # Some upstream resolved parameters contain objects; save an audit string as
    # well as the exact user-supplied config, without serializing Python objects.
    if "resolved_config" in extra:
        Path("resolved-config.txt").write_text(repr(extra.pop("resolved_config")))
    atomic_json(sys.argv[2], {"version":version, "rows":rows, **extra})


if __name__ == "__main__": main()
