# v1 三档高负载离线预训练与 RL 对照

这是独立于中期 v1 结果的探索实验，保持原模型、奖励、规模和物理机制不变，不覆盖已完成的实验。

- 规模：3 集群 / 10 缓存节点；负载 0.75 / 0.875 / 1.00，速率 300 / 350 / 400 请求每秒，固定总容量。
- 教师：复用最初 legacy-independent-pretrain-v1 冻结教师（98,304 步），三档均使用同一策略权重。采集策略副本仅覆盖场景到达率和名义负载，并记录原教师 SHA-256；不在新负载上训练教师。
- 混合 base：每档 128 个训练、32 个监督验证 episode，共 384 / 96 个；单负载 base：0.75 下 384 / 96 个，控制总数据量。混合组的 0.75 数据是单负载数据的首 128 / 32 个，数据共享是有意设计。
- 采集使用原 train / validation 独立种子空间，不与闭环验证种子重叠；每个 episode 保持固定负载。三档等决策步数、完整遍历，因此总体等权；不是 episode 内切换负载。
- 两组均使用独立两层 256 ReLU actor、13 维输入、五维动作、可学习方差，KL(teacher||student) 蒸馏 40 个完整 epoch、batch 128、Adam 3e-4、seed 0，按监督验证 KL 保存 best。
- 基础质量检查：每档 10 个固定开发验证 episode，比对冻结教师与学生；混合组全部三档、单负载组仅训练档 0.75 要求成功率差不低于 -1pp、成功延迟不超过教师 105%。不通过则保留诊断并停止后续 RL。通过不等于超过基线或独立测试通过。
- RL：三档分别跑 mixed / single / scratch，seed 0，2,048 episode × 128 周期 = 262,144 环境步。8 worker、batch=minibatch=1024、每批 5 轮、lr 3e-4、advantage normalization，与 legacy 训练一致。优化器和 critic 重新初始化。
- 复用现有 0.75 scratch 最终结果，另运行 8 个 RL 任务；最多两项训练并行，评估使用同一文件锁排队。每 8,192 步评估 10 个固定种子，记录确定性与随机执行，主对比随机执行。
- 本阶段只做开发验证，不使用独立测试集，不承诺混合 base 一定更好。与原始单负载预训练实验比较时，要说明本次样本数不同；公平主对照是本轮等数据量 single base。
- 数据 manifest、完整 epoch checkpoint、RL last.pt 保留；异常后使用同一输出恢复，不能重复启动存活的 launcher。W&B online 同时保留本地日志。

入口：

```sh
.venv/bin/python scripts/run_legacy_mixed_load_trial.py \
  --teacher outputs/legacy-independent-pretrain-v1/teacher.pt \
  --scratch-reference outputs/mappo-legacy-f064-restored-seed0 \
  --output outputs/legacy-mixed-load-night-v1 --wandb-mode online
```

本地 smoke 使用短 episode 的现有教师，加 `--smoke --wandb-mode offline`，跑通两种蒸馏及全部九个短 RL 闭环。smoke 允许质量门槛不通过后继续，仅验证流程，不作性能结论。
