# Production launch: frozen split and training pilot

The user authorized using available CPU capacity to proceed after the verified
50-complex report. Existing constraints remain: no scientific work on login
nodes, exclude comp1400, 2 GB per core, and at most 400 user-requested cores
queued plus running.

The versioned `production-1024-v1` campaign covers all 1452 currently usable
frozen complexes: 778 training, 151 validation and 523 test. Before any new
predictions, initialization verifies the frozen artifacts and actual structure
and annotation hashes, then selects 500 training complexes by seeded SHA-256
sequence-group round robin. Selection does not use structure size, labels or
prediction success. Pilot membership is recorded in `pilot.json`; the remaining
training complexes stay in the production universe. No split is changed.

The production JAX wrapper uses 1024 iterations with unchanged numerical
thresholds. It is copied from the verified isolated readout, runs each state
in a fresh process and extracts midpoints from the already computed grid.
A production equivalence gate compares all three states of a smoke complex
with the accepted diagnostic predictions before workers are released. Prior
50-complex validation remains the broad numerical evidence.

There are 8712 complex-method tasks for PypKa, PROPKA, JAX-Ka, pKAI, pKAI+ and
the zero-shift baseline. Compatible completed smoke predictions are reused with
source receipts, content hashes and input-coordinate checks. The 500-pilot
PypKa/PROPKA tasks have highest priority, followed by the pilot comparison
methods and remaining benchmark work. Labels retain the approved uncertainty
masks and existing teacher configuration; historical pKPDB labels are not mixed
into these newly generated labels.

A bounded Slurm array supplies up to 198 persistent workers (396 requested
cores, 4 GB per worker), with two cores reserved for collection. The shared
submission lock counts all user jobs and rejects submissions that would exceed
400 cores. Live scheduler state determines which responsive nodes can start
workers. Each worker claims a task under a filesystem lock; completion frees it
to take another task. Idle workers exit when the queue is exhausted.

Worker allocations last at most 24 hours; workers stop taking new tasks after
21 hours. PypKa and JAX have 5400-second total AB/A/B budgets; other methods
have 600 seconds. Process failures and numerical invalidity are retained, never
converted to valid labels. The collector refuses to publish a complete score
report if any task lacks a receipt (for example after an OOM kills a worker).
Such tasks require explicit scheduler review and a bounded recovery wave.

Runtime root:
`/home/coulson/oc/lina4225/_runtime/jax-Ka/pkabench/campaigns/production-1024-v1`.

Initialization: 730936. Production wrapper gate: 730937. Pool and collector
IDs are recorded in the shared scheduler submission ledger and launch receipt.
The active source, environment locks, weights, native runtime manifest,
configuration, pilot and frozen-mask hashes are pinned in the campaign manifest.

This launch creates labels and benchmark predictions. It does not launch model
training, finish experimental Set 2 curation, or establish independent physical
accuracy. Completing 500 teacher processes is distinct from obtaining 500
complexes with sufficient usable labels; label coverage must be checked before
constructing the training dataset.

## Confirmed launch

Initialization completed and selected 500 training complexes spanning all
277 eligible training sequence groups. The production wrapper gate passed
in 32 seconds. Array 730938 supplies 198 workers; collector 731115 waits for
all array elements to terminate. At the initial scheduler check, 176 workers
were running (352 cores), with 22 workers plus the collector pending (46 cores):
398 total requested cores. Later occupancy follows the scheduler's live capacity.

## User amendment: defer JAX-Ka

The user requested skipping JAX-Ka for the initial pilot and rerunning it later
with more resources. Under the shared queue lock, 952 unclaimed JAX tasks were
removed; the claimed prefix of 3396 tasks was preserved. The original queue is
archived in `queue-before-jax-deferral.json`, with removed tasks recorded in
`deferred-jaxka-queue.json`. Completed/failed JAX predictions remain available.
Already-started JAX tasks finish without interrupting other-method workers.
The original scientific manifest and active implementation hashes are unchanged.

Collector 731115 was cancelled because it included JAX-Ka. Replacement reports
use PypKa, PROPKA, pKAI, pKAI+ and the null baseline, recomputing common support
without JAX. A separate `training-pilot-nojax-v1` report waits for all 500 pilot
complexes to have receipts for these methods, including explicit failure receipts.
A `production-nojax-v1` report follows the full pool. Both verify input/output
receipts, preserve frozen masks and independently recompute group macro scores.
JAX resource recovery is deferred; no larger JAX allocation has been launched.

## Pilot scorer scheduling

The user authorized comp1400 for one scoring run only. Before relocation,
six production workers were retired safely after their final receipts were
written, holding the queue lock so they could not claim another task. This
restored the 400-core cap after unrelated account jobs were submitted. Their
IDs are recorded in `workers-retired-for-core-cap.json`; no in-flight prediction
was interrupted and no claimed task was lost.

Freeing the first four slots let the original pilot scorer 731160 start on
comp0650. It was left running there, so the comp1400 exception was not used.
The general comp1400 exclusion remains unchanged. Full-report collector 731161
still follows the array via afterany, including the intentionally retired workers.

The original pilot scorer subsequently exceeded 4 GB after 2 min 56 s. Its
assembled predictions survived. The user's one-time comp1400 exception was
then used for scoring retry 731178, with 8 CPUs / 16 GB (still 2 GB/core).
The retry resumes scoring from the assembled pilot predictions and validates
source receipts and masks before scoring. No predictions are recomputed.
The exception is confined to the one-time scoring interpreter and recorded
in `one-time-comp1400-scoring.json`; global runtime exclusions are unchanged.
Additional workers were retired only after completing their tasks to make
room under the 400-core cap as unrelated account jobs increased their requests.

Before retry 731178 started, comp1400 became fully allocated (48/48 CPUs).
The same queued job was moved to comp0650/amd96 with unchanged 8 CPUs / 16 GB,
using the normal compute guard on that node. The one-time exception receipt
records `exception_used` from the actual node; comp1400 was not used by this
fallback. Global exclusions remain unchanged.

## Receipt audit and recovery (2026-10-05)

Pilot scoring retry 731178 completed and verified: 2500 receipts, 984 group
metrics, 458/500 usable teacher interface pairs, 24646 teacher paired sites and
23818 all-method common sites. JAX remains excluded.

A later audit found 20 non-JAX tasks without receipts on retired workers.
This corrects the earlier assertion that worker retirement interrupted no
claims. Holding the queue lock while requesting cancellation did not establish
that all workers had terminated before the lock was released; cancellation
completion was not checked. The task/worker ledger is preserved. Recovery jobs
731201–731220 rerun only these missing tasks with unchanged science and retained
old attempts. One other original worker was still active at audit time.
Collector 731161 was cancelled and replaced by a collector dependent on all
recovery jobs and the original array. The full collector requests 16 CPUs /
32 GB, maintains 2 GB/core, excludes comp1400 and enforces the 400-core cap.
Existing failure receipts remain explicit; they are not made valid by recovery.
