# Current DEPPO v2 experiment evaluation protocol

This protocol records the comparison selected during the September 20, 2026
exploratory runs, before inspecting the held-out test outcomes. These runs are
scenario-adapted experiments, not a reproduction of the paper's numerical tables.

## Primary matched comparison

- DEPPO: `outputs/deppo-v2-wide-seed{0,1,2}`.
- MAPPO-no-context: `outputs/mappo-v2-wide-seed{0,1,2}`.
- Each run receives 262,144 environment decision steps (2,048 episodes), not
  multiplied by the number of agents.
- Both use the small physical scenario, business reward with scale 1, independent
  actors, a shared centralized critic, MLP width 256, learning rate 3e-4,
  per-agent advantage normalization, initial latent standard deviation 0.3,
  batch/minibatch 512, five PPO epochs, and four simulation workers.
- DEPPO additionally uses the 128-unit GRU over eight completed observation-action
  pairs. This is a controlled history ablation, not a claim that either method's
  hyperparameters are independently optimal.

## Checkpoint selection and execution

For each training seed, report two explicitly labeled execution modes:

1. Deterministic: choose `best.pt` by deterministic validation success rate,
   breaking ties with lower mean successful-request latency.
2. Sampled policy: choose `best-stochastic.pt` by sampled validation success rate,
   with the same latency tie-breaker; test using `--exploration stochastic`.

Use the same ten validation workload seeds 1,000,000,000 through 1,000,000,009.
Sampled validation was added partway through training; its separate checkpoint
retention was added later. Earlier logged scores without retained models cannot
be retroactively selected. Record the retained checkpoint's actual training step
and code revision. Do not choose an execution mode or change a checkpoint based
on test results. Preserve both modes in the final report.

## Fixed and heuristic baselines

The completed validation grid is in `outputs/deppo-v2-baseline-audit/selection.json`:
three forwarding rules (random, local, forward) times nine fixed backhaul ratios
(0.1 through 0.9). Tuned-fixed selected forward with ratio 0.6. Its validation
success rate was 0.3194097816528257; this value is not a test result.

Also report the original Random (per-request random forwarding and periodic
ratio uniform in [0.4, 0.6]), Always-local (ratio 0.5), Always-forward (ratio 0.5),
and Queue-adaptive definitions. Do not conflate fixed-ratio random forwarding
with the original Random baseline. All methods use identical topology, resource
budgets, caching rules, request workload seeds, deadlines, and drain semantics.

## Held-out evaluation

After full-budget training and validation selection, freeze the selected files
and record their hashes. Evaluate all methods on the same thirty workload seeds
2,000,000,000 through 2,000,000,029, using `evaluate --split test --episodes 30`.
Do not use this set for further hyperparameter or checkpoint selection. Existing
small smoke tests are implementation checks, not evidence of workload performance.

Report individual training-seed results, the mean and spread across all three
seeds, and paired differences against each baseline. Bootstrap by training seed
and by matched workload for uncertainty; do not present a single training seed's
workload interval as cross-training-seed evidence. Any interim validation
intervals remain exploratory because the validation set has informed tuning.

Metrics include success, timeout, rejection and overflow rates; completed and
unfinished requests; successful-request mean/p50/p95/p99 latency; cache hit rate;
backhaul/delivery bytes and utilization; and forwarding ratio. State explicitly
that latency excludes failed requests and that reported mean episode quantiles
are not pooled request quantiles. Evaluation drains pending work; requests still
pending at the training horizon are separately reported rather than counted as
immediate failures. Compare raw business reward across reward-scale variants.

## Exploratory runs and limitations

The original and normalized small-network candidates were pruned with healthy
checkpoints preserved. They are not completed full-budget baselines. The
`deppo-v2-scaled-seed0` candidate uses reward scale 0.1 and is a single-seed
conditioning experiment; do not label it a three-seed result. It is not included
in the primary matched history ablation.

The original serial reward/learning-rate search suite was stopped in favor of
parallel exploratory training at the user's request. Consequently these results
must not be described as completion of the initially proposed exhaustive,
equal-budget hyperparameter grid.
