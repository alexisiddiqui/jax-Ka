#!/bin/bash
# abdata-v5: the pool-builder + ledger inputs from the Coulson runtime as ONE archive for Isambard (extracted node-local by
# the Isambard jobs; no files land on scratch). Paths inside are relative to the runtime root, so PKABENCH_RUNTIME=<extract dir>.
#   bash make_bundle.sh OUT_DIR      -> OUT_DIR/pool-inputs.tar.zst, files.txt, pool-inputs.sha256
set -euo pipefail
R=/home/coulson/oc/lina4225/_runtime/jax-Ka/pkabench; OUT=$1; mkdir -p "$OUT"; cd "$R"
{
  ls pretraining/pinder-pkai-v1/index/label_*.jsonl
  echo pretraining/pinder-pkai-v1/pinder_heldout_exclusions_v3.tsv
  find pretraining/pinder-pkai-v1/entries -mindepth 2 -maxdepth 2 \( -name sites.json -o -name labels.json \)
  printf '%s\n' pretraining/pkpdb-full-v1/{verification.json,pilot.json,protocol.json,references.json}
  find pretraining/pkpdb-full-v1/entries -mindepth 2 -maxdepth 2 -name defects.json
  printf '%s\n' audits/pinder-prefilter-v1/{prefiltered_v1.parquet,shortlist_v1.parquet,uniq.fasta,ref_heldout_all.fasta,prefilter.py}
  printf '%s\n' universe/structural-freeze-v1/assignments.parquet universe/combined-split-v1/index.json training/ogqt-pinder-factorial-v1/cohort.json
  printf '%s\n' training/pool-v4/{pinder.tsv,pinder-val.tsv,pkpdb.tsv,pkpdb-val.tsv,pkpdb-chain-clusters.tsv,pool-v4.json}
  printf '%s\n' audits/pinder-index-v1/{index.parquet,metadata.parquet}
  ls audits/pinder-prep-v2/results_v7/pf_*.jsonl audits/pinder-prep-v2/results_v7/fb_*.jsonl
} > "$OUT/files.txt"
wc -l < "$OUT/files.txt"
nice -n 19 tar -cf - -T "$OUT/files.txt" | nice -n 19 zstd -3 -T2 -q -o "$OUT/pool-inputs.tar.zst.partial"
mv "$OUT/pool-inputs.tar.zst.partial" "$OUT/pool-inputs.tar.zst"
sha256sum "$OUT/pool-inputs.tar.zst" "$OUT/files.txt" | tee "$OUT/pool-inputs.sha256"
# the aggregate (non-entry) inputs individually, for the provenance record
grep -v '/entries/' "$OUT/files.txt" | xargs sha256sum > "$OUT/aggregate-inputs.sha256"
echo done
