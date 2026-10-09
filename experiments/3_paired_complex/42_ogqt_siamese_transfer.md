# Pretrained oGQT Siamese transfer

The scratch structural-weight factorial selected the unweighted vanilla paired
objective.  The transfer test therefore keeps that objective fixed and compares
the selected scratch checkpoint with the selected epoch-16 pKPDB oGQT checkpoint
at epoch zero and after paired fine-tuning.

Two fine-tuning schedules run with identical seed-17 batches:

| Arm | Learning rate through epoch 10 | Cosine endpoint |
|---|---:|---:|
| Standard | `1e-3` | `1e-5` |
| Low | `1e-4` | `1e-6` |

Both reset AdamW state after loading the pretrained parameters.  Epoch zero is
eligible for checkpoint selection, preventing a destructive fine-tune from
being reported as an improvement.  Selection remains unweighted validation
state MAE plus interface paired MAE.  The split, model, inputs, targets, batch
plans and test-data exclusion are unchanged from the scratch factorial.

After transfer, score the original pKPDB-pretrained checkpoint, scratch vanilla
Siamese checkpoint, and both transferred checkpoints on the unchanged
142-complex pKPDB/PypKa validation set.  This is a single-state evaluation: it
tests default pKa retention and does not claim a pKPDB binding-shift result.
