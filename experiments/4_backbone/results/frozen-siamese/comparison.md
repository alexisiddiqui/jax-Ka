# Frozen Siamese interface heads

Three frozen 50% relational GQT checkpoints. All primary results use free tokens only. Equal weights for free BCE, bound BCE and detached bound-to-free probability consistency.

| Head | Objective | Seed | Epoch | Free macro AP | Free pooled AP | Free AUROC | Bound macro AP |
|---|---|---:|---:|---:|---:|---:|---:|
| small | free-only | 17 | 1 | 0.4738522118619113 | 0.2753381113572882 | 0.6196711140355561 | — |
| small | siamese | 17 | 11 | 0.4420077924150978 | 0.23190146619346239 | 0.5683944203998921 | 0.5689157884884035 |
| mlp | free-only | 17 | 2 | 0.46301576392919586 | 0.2753168189100508 | 0.6159937145904698 | — |
| mlp | siamese | 17 | 3 | 0.42986084093242216 | 0.219921884259014 | 0.5531490942607566 | 0.5776565032957804 |
| transformer | free-only | 17 | 3 | 0.4654698561917183 | 0.28596615799800995 | 0.6088868074275491 | — |
| transformer | siamese | 17 | 2 | 0.450624245572336 | 0.2517905420203845 | 0.5840371795618757 | 0.5845118583106245 |
| small | free-only | 29 | 1 | 0.4636771354201246 | 0.2707470793327934 | 0.615570809120801 | — |
| small | siamese | 29 | 12 | 0.4314541619101668 | 0.21649471361201644 | 0.5510431096928396 | 0.5880483046957907 |
| mlp | free-only | 29 | 2 | 0.4666219548340236 | 0.26599275028134384 | 0.6105430947852107 | — |
| mlp | siamese | 29 | 2 | 0.43128946310074606 | 0.21412442724815703 | 0.5479293828605624 | 0.5912184645412636 |
| transformer | free-only | 29 | 2 | 0.45662145222432 | 0.2988034012321536 | 0.6121344982682864 | — |
| transformer | siamese | 29 | 2 | 0.44078655288770896 | 0.2535606158741719 | 0.5796765498741301 | 0.5763116682099307 |
| small | free-only | 43 | 3 | 0.47562724133713347 | 0.27705501340942484 | 0.621995684963546 | — |
| small | siamese | 43 | 1 | 0.4216368577774929 | 0.2115330065262606 | 0.5499839474266627 | 0.5739400559909933 |
| mlp | free-only | 43 | 1 | 0.46601348944307447 | 0.28231660169204775 | 0.6227855875330981 | — |
| mlp | siamese | 43 | 9 | 0.4228720091896579 | 0.21642354155567373 | 0.5442145242804819 | 0.5708728726393104 |
| transformer | free-only | 43 | 1 | 0.4624151927240934 | 0.2844449962555974 | 0.612806859132892 | — |
| transformer | siamese | 43 | 3 | 0.44363961822152387 | 0.23431443198776233 | 0.5659589781125189 | 0.588091813066553 |

Means: {"small/free-only": {"site_macro_ap": 0.4710521962063898, "pooled_ap": 0.27438006803316883, "auroc": 0.6190792027066343}, "small/siamese": {"site_macro_ap": 0.4316996040342525, "pooled_ap": 0.21997639544391315, "auroc": 0.5564738258397982}, "mlp/free-only": {"site_macro_ap": 0.465217069402098, "pooled_ap": 0.2745420569611475, "auroc": 0.6164407989695929}, "mlp/siamese": {"site_macro_ap": 0.428007437740942, "pooled_ap": 0.21682328435428158, "auroc": 0.548431000467267}, "transformer/free-only": {"site_macro_ap": 0.46150216704671054, "pooled_ap": 0.28973818516192035, "auroc": 0.6112760549429092}, "transformer/siamese": {"site_macro_ap": 0.445016805560523, "pooled_ap": 0.2465551966274396, "auroc": 0.5765575691828416}}

Selection uses free-only validation macro AP. GQT is frozen and its pKa predictions unchanged. This evaluates the combined bound-supervision plus consistency objective; separating those contributions needs a bound-supervision-only control. Labels cover contacts between eligible pKa-labelled surface sites, not all residues. Free means partner-removed bound conformation. Scores are not calibrated probabilities. No independent test data used.
