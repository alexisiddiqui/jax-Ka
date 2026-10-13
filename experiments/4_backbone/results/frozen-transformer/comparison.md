# Frozen free site transformer comparison

Frozen 50% relational GQT, three seeds. Transformer 33,665 parameters; matched MLP 33,661. Both new heads use dropout 0.1. Original MLP uses no dropout.

| Head | Seed | Selected epoch | Macro site AP | Pooled AP | AUROC |
|---|---:|---:|---:|---:|---:|
| original-mlp | 17 | 1 | 0.4738522118619113 | 0.2753381113572882 | 0.6196711140355561 |
| transformer | 17 | 3 | 0.4654698561917183 | 0.28596615799800995 | 0.6088868074275491 |
| mlp | 17 | 2 | 0.46301576392919586 | 0.2753168189100508 | 0.6159937145904698 |
| original-mlp | 29 | 1 | 0.4636771354201246 | 0.2707470793327934 | 0.615570809120801 |
| transformer | 29 | 2 | 0.45662145222432 | 0.2988034012321536 | 0.6121344982682864 |
| mlp | 29 | 2 | 0.4666219548340236 | 0.26599275028134384 | 0.6105430947852107 |
| original-mlp | 43 | 3 | 0.47562724133713347 | 0.27705501340942484 | 0.621995684963546 |
| transformer | 43 | 1 | 0.4624151927240934 | 0.2844449962555974 | 0.612806859132892 |
| mlp | 43 | 1 | 0.46601348944307447 | 0.28231660169204775 | 0.6227855875330981 |

Means: {"original-mlp": {"site_macro_ap": 0.4710521962063898, "pooled_ap": 0.27438006803316883, "auroc": 0.6190792027066343}, "transformer": {"site_macro_ap": 0.46150216704671054, "pooled_ap": 0.28973818516192035, "auroc": 0.6112760549429092}, "mlp": {"site_macro_ap": 0.465217069402098, "pooled_ap": 0.2745420569611475, "auroc": 0.6164407989695929}}

Labels, free RSA >=0.1 surface gate, and validation support unchanged. Transformer attends within each chain, among cached eligible pKa sites only. No bound tokens, partner coordinates, or new geometric/positional input. GQT is frozen. Macro AP excludes no-positive complexes; pooled metrics include them. Selected validation results are not an independent test estimate.
