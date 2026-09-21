# Minimal pretraining on the legacy MAPPO architecture

This path preserves independent two-layer 256-unit ReLU actors, 13 local input
features, five bounded actions, and learned action standard deviations. The
initial standard deviation is 0.3. It adds no shared actor, history encoder,
LayerNorm, dropout, fixed variance, or new reward. The simulator has no learning
dependencies. The historical restoration experiment continues separately.

1. Train a legacy MAPPO teacher and freeze its stochastic-validation-selected
   checkpoint. Teacher training uses seed 17 in the local workflow.
2. Collect complete episodes on separate seeds: 100000000 onward for training,
   200000000 onward for supervised validation. Existing demonstration shards
   contain observations, sampled actions, teacher means/stds/log probabilities,
   next observations, interval rewards/outcomes, termination and truncation,
   workload, and episode seed. Preserve the agent axis. Only the first 13 local
   observation features enter the actor; the existing archive also retains its
   history padding for compatibility.
3. Minimize KL(teacher || student) separately for each agent, averaging across
   samples and agents. Both distributions have the identical invertible
   tanh/affine transform, so pre-transform Gaussian KL equals action-space KL.
   Every epoch shuffles and consumes every training timestep/agent once, including
   the final partial minibatch. Validation consumes only validation episodes.
4. Evaluate the frozen student before RL. For real experiments, failure to match
   the teacher calls for diagnosis before claiming useful pretraining. The local
   smoke deliberately continues to verify the entire pipeline even if it fails
   that performance check; its teacher is itself only a short-run fixture.
5. Initialize independent MAPPO actors from their corresponding student weights.
   Keep the paired scratch critic initialization identical and create fresh RL
   optimizers. Both arms use the same PPO settings and workload seeds. The frozen
   student remains a separate reference. Do not load a different cluster count
   into this versioned independent-actor format.

The `edge-distill-independent-v1` checkpoint contains every actor state, optimizer,
epoch, best validation KL, metric history, manifest hash, and shuffle/Torch RNG
states. `fit(..., resume=True)` resumes after the last complete epoch. Existing
shared `edge-bc-v1` and contextual `edge-bc-v2` loading paths remain supported.
RL checkpoints retain optimizer/RNG states; new runs also save `initial.pt` for
auditing matched critics and initialization.

## Local workflow verification

```sh
.venv/bin/python scripts/smoke_legacy_pretrain.py --output outputs/legacy-distill-smoke-v1
.venv/bin/python -m pytest tests/test_legacy_distill.py tests/test_pretrain.py tests/test_context_rl.py -q
```

Use a fresh output directory. Individual distillation and RL stages support
checkpoint resume; the top-level smoke script is not an automatic job scheduler.

The smoke keeps the 3/10 physical scenario, load 0.75, no retries, and business
reward. To fit local CPU verification it shortens episodes to 16 decisions,
uses two workers and 64-step batches, trains the teacher for 256 environment
steps, collects 8 training and 2 validation episodes, distills for 20 full epochs,
and trains both RL arms for 512 environment steps. PPO still uses learning rate
3e-4, normalized advantages, five epochs, and the original clipping/GAE/gamma.
Closed-loop comparisons use two paired validation seeds (1000000000–1000000001).
No independent final test set is used, and no convergence/benefit claim follows
from this smoke. No server experiment is launched by this script.

Outputs include spec, teacher/student/RL checkpoints, data manifest and shards,
offline W&B files for training, distillation CSV, pre-RL closed-loop results,
complete paired evaluation JSON, and an audited `report.json`.

## Server trial

`scripts/run_legacy_pretrain_trial.py --teacher <best-stochastic.pt> --output <new-dir>`
freezes an existing teacher, collects 128/32 train/validation episodes, and distills
for 40 complete epochs. This small actor's offline distillation runs on CPU;
the paired RL jobs use CUDA. Distribution KL is logged to W&B each epoch.
The frozen student must be within 1 percentage point of teacher success and
within 105% of teacher mean successful-request latency on 10 paired validation
episodes before RL starts. Failure writes `gate.json` and stops before RL.
Passing launches two independent 262144-step old-MAPPO jobs: eight sampling
workers each, batch 1024, original PPO settings and learnable standard deviation.
Their common seed differs from teacher training. Evaluation jobs share a lock.
The same teacher-selection validation seed set is reused for development only;
none of these scores is an independent test result. Teacher training is an
explicit upstream cost, including when the source is an in-progress snapshot.
Partial collections and completed distillation epochs can resume in the same
output. Existing RL checkpoints resume; completed arms are not restarted.
Final comparison checks initial critic equality and matched evaluation workloads.
`--smoke --device cpu --wandb-mode offline` exercises this launcher locally with
reduced data/training budgets; it still enforces the pre-RL gate.
