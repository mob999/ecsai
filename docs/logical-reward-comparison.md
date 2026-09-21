# Fixed-load logical reward comparison

Compare BC initialization against scratch MAPPO at 3 clusters / 10 caches,
fixed delivery load 0.75, seed 0, and 65,536 environment steps per arm.
Use the same checkpoint, workload seeds, retry rules, network, optimization,
and evaluation schedule as `context-rl-small-fixed075-v1`. Only reward changes.

`reward_mode=logical` settles each original request exactly once: on success
or after exhausting two retries. Intermediate failures carry no separate
failure penalty. For terminal requests in the interval:

    reward = (successes - final_failures - 0.1 * sum(total_elapsed / D)) / Z

`total_elapsed` runs from the original arrival through failed attempts and
100 ms retry waits to final completion or failure. D is the configured single
attempt deadline (1 second). Z is expected original arrivals over the episode
divided by decision cycles, clamped to at least 1; it stays fixed across windows.
Elapsed time is charged for both successful and finally failed requests.
This is a success-oriented scalar surrogate, not an exact lexicographic objective.

`train/reward` and `train/logical_reward` report the new reward; the old
`train/business_reward` remains a diagnostic with its original attempt semantics.
Do not compare absolute reward magnitudes between the two objectives.
Compare final logical success rate, full successful-request delay, failed-request
elapsed time, all-request elapsed time, and attempts using paired evaluation seeds.

The 128-cycle horizon remains a truncation with bootstrap. Unfinished requests
receive no fabricated outcome. Evaluation drains all retries; its
`logical_settled_return` includes the full ledger, while `episode_return` retains
only the training horizon. Drain contributes no PPO transitions. The existing
finite training horizon and gamma discount remain limitations of alignment.
This experiment requires max_retries > 0 for SDK attempt-outcome telemetry.
