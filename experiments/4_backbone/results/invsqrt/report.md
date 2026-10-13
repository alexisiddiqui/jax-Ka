# Inverse-square-root reference and ablation results

Unweighted validation; selected by the unchanged overall criterion. Architectural comparisons are exploratory single-seed results.

| Run | Epoch | Selection | State | Interface Δ | pKPDB | Benchmark |
|---|---:|---:|---:|---:|---:|---:|
| invsqrt-v2-50-s17 | 18 | 0.6555 | 0.4111 | 0.2444 | 0.4938 | 0.5144 |
| invsqrt-v2-50-s29 | 20 | 0.6592 | 0.4138 | 0.2454 | 0.5015 | 0.5195 |
| invsqrt-v2-50-s43 | 20 | 0.6623 | 0.4149 | 0.2474 | 0.4996 | 0.5153 |
| invsqrt-v2-100-s17 | 20 | 0.6283 | 0.3941 | 0.2342 | 0.4802 | 0.4894 |
| invsqrt-v2-100-s29 | 16 | 0.6311 | 0.3991 | 0.2321 | 0.4843 | 0.5016 |
| invsqrt-v2-100-s43 | 20 | 0.6376 | 0.3989 | 0.2386 | 0.4842 | 0.5020 |
| control-v2-50-s17 | 17 | 0.6444 | 0.4065 | 0.2379 | 0.4933 | 0.5165 |
| control-v2-50-s29 | 20 | 0.6510 | 0.4131 | 0.2379 | 0.4959 | 0.5200 |
| control-v2-50-s43 | 17 | 0.6529 | 0.4109 | 0.2420 | 0.4963 | 0.5155 |
| invsqrt-v2-50-fourier-s17 | 17 | 0.6795 | 0.4296 | 0.2499 | 0.5158 | 0.5249 |
| invsqrt-v2-50-rope-s17 | 17 | 0.6597 | 0.4153 | 0.2444 | 0.5039 | 0.5167 |
| invsqrt-v2-50-shared-projection-s17 | 17 | 0.6517 | 0.4096 | 0.2421 | 0.4955 | 0.5044 |
| invsqrt-v2-50-relations-s17 | 17 | 0.6459 | 0.4052 | 0.2407 | 0.4906 | 0.5108 |
| invsqrt-v2-50-paired-delta-s17 | 17 | 0.6562 | 0.4108 | 0.2454 | 0.4927 | 0.5169 |
| prod-100pct-w64-drop10-bur1-orient-dihed | 17 | 0.6183 | 0.3903 | 0.2280 | 0.4702 | 0.4960 |
| prod-100pct-w64-drop10-bur1-orient-dihed-s29 | 17 | 0.6182 | 0.3916 | 0.2265 | 0.4755 | 0.4952 |
| prod-100pct-w64-drop10-bur1-orient-dihed-s43 | 18 | 0.6272 | 0.3950 | 0.2322 | 0.4776 | 0.5001 |

## Matched weighted minus unweighted reference differences

50%: seed differences [0.011071474752728072, 0.008275865558901474, 0.009426277596874866]; mean 0.0096.
100%: seed differences [0.010007862664724643, 0.012930852284037342, 0.010343517323880391]; mean 0.0111.
