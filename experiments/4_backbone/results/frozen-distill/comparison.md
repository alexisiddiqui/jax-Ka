# Frozen teacher interface distillation

Small MLPs on three frozen 50% relational GQT checkpoints. Bound teacher is independently trained, selected and frozen before free-only students train.

| Role | Input | Seed | Epoch | Macro AP | Pooled AP | AUROC |
|---|---|---:|---:|---:|---:|---:|
| labels-only | free | 17 | 1 | 0.4738522118619113 | 0.2753381113572882 | 0.6196711140355561 |
| teacher | bound | 17 | 3 | 0.617183850238734 | 0.4710007226054389 | 0.7368268303660872 |
| lambda-0.1 | free | 17 | 1 | 0.47417849592426736 | 0.2747246873856745 | 0.6191244189493359 |
| lambda-1 | free | 17 | 1 | 0.4652908359523483 | 0.2616200233806219 | 0.5989277663698356 |
| labels-only | free | 29 | 1 | 0.4636771354201246 | 0.2707470793327934 | 0.615570809120801 |
| teacher | bound | 29 | 12 | 0.6282879858250817 | 0.49097396385534264 | 0.7436851150151329 |
| lambda-0.1 | free | 29 | 1 | 0.46281619360049636 | 0.26906574315986637 | 0.6132734762659897 |
| lambda-1 | free | 29 | 2 | 0.4554247585384972 | 0.2512883889613452 | 0.5944801750398305 |
| labels-only | free | 43 | 3 | 0.47562724133713347 | 0.27705501340942484 | 0.621995684963546 |
| teacher | bound | 43 | 9 | 0.6288979170923527 | 0.4961721596368945 | 0.7478636092360038 |
| lambda-0.1 | free | 43 | 2 | 0.4734090597496119 | 0.2766588317291344 | 0.6224816023959461 |
| lambda-1 | free | 43 | 2 | 0.46143135408512725 | 0.25436765344469836 | 0.596065846254321 |

Means: {"labels-only": {"macro_ap": 0.4710521962063898, "pooled_ap": 0.27438006803316883, "auroc": 0.6190792027066343}, "teacher": {"macro_ap": 0.6247899177187227, "pooled_ap": 0.4860489486992254, "auroc": 0.7427918515390747}, "lambda-0.1": {"macro_ap": 0.4701345830914585, "pooled_ap": 0.27348308742489175, "auroc": 0.6182931658704239}, "lambda-1": {"macro_ap": 0.46071564952532423, "pooled_ap": 0.2557586885955552, "auroc": 0.5964912625546623}}

Teacher rows use bound input and are diagnostic, not directly comparable inference performance. Students use free tokens only. Teachers selected by bound validation AP; students by free validation AP. Teacher probabilities are training-only targets. Labels cover contacts between eligible pKa-labelled surface sites; free structures are partner-removed bound conformations. No GQT updates or independent test evaluation.
