# 行为克隆 base actor 与 MAPPO 微调

## 调研与选择（2026-09-20）

- imitation 的 BC 对专家 observation/action 数据执行监督学习，以负对数似然衡量动作拟合：https://imitation.readthedocs.io/en/stable/_api/imitation.algorithms.bc.html 。无需为此引入第二套 SB3 训练栈。
- Policy Distillation 展示了从一个或多个 RL 教师提取学生策略的思路：https://arxiv.org/abs/1511.06295 。这是本实现采用跨 agent 共享学生的依据，不是对边缘调度效果的保证。
- Minari 的 PyTorch BC 示例展示以 episode 数据构造监督样本：https://minari.farama.org/tutorials/using_datasets/behavioral_cloning/ 。本项目先复用现有环境导出，不引入 Minari 运行依赖。
- TorchRL 当前文档含 BC 相关新接口，但项目锁定 0.10.1；本实现直接复用已安装的 NormalParamExtractor 和 TanhNormal，避免因新 API 升级训练栈。

## 设计先行

目标：训练一个与集群数量无关的 base actor，在未参与预训练的目标规模上做 MAPPO 微调。SDK 和当前三规模实验不变。

1. 教师：已训练的 MAPPO-no-context 五维 checkpoint；保留教师 SHA256、原配置、训练步数和选择依据。教师按固定验证集选择，测试集不参与选择。初次实验使用 small 的已有 best-stochastic 快照，明确是有限预算教师，不能称作最优专家。
2. 数据：使用与 RL 训练、验证、测试隔离的种子区间；每个完整 episode 一份原子写入的 tensor 文件。保存完整观测（含历史）、联合动作、教师 loc/scale/log_prob、next observation、共享 reward、terminated/truncated，以及区间计数和排空后的整集结果。训练时仅取观测前 13 维，不给 actor 全局信息。延迟结果是系统统计，不把当期成功硬归因给当期动作；暂不做结果加权和请求级因果标签。
3. 拆分：按完整 episode/种子拆分监督训练与验证，不能打散相邻步后再拆分。manifest 固定教师、配置、种子、分片校验和；失败重跑跳过已校验完整 episode。
4. base：与现有无历史 actor 相同的两层 256 ReLU MLP、相同有界 TanhNormal 与标准差约束。所有源 agent 共同训练一个学生；多源数据先均匀选源、再选 episode、step、agent，避免大规模源因为样本多占优势。监督目标为动作 NLL；记录 held-out NLL、教师到学生解析 KL、动作边界比例。教师分布用于诊断，不修改动作标签。
5. 局限：13 维归一化观测没有显式拓扑/资源条件，可能存在状态混叠；先量化现有表示的迁移能力。未来新增资源上下文需要对 scratch/BC 同时更改，不能混作预训练收益。
6. 微调：把 base 权重复制到目标场景每个独立 actor，保持参数不共享；critic 按目标 agent 数重新初始化。支持全量微调与仅最后线性层微调。初始化在 BenchMARL functionalization/optimizer 构造前完成；同步采样策略后才采样。已有 checkpoint 的正常 resume 与新初始化区分，不能同时使用。
7. 对照：相同目标场景、架构、训练预算和评估种子，比较 scratch、BC-only、BC+full、BC+head。先 small 教师→medium（5/20/600，rho=.75），目标场景不进入源数据；随后可扩展多源→未见规模。初次验证 seed 0 是可行性实验，不声称统计显著。
8. 初次服务器预算：源数据 128 个训练 episode、32 个监督验证 episode；BC 最多 20 轮，按验证 NLL 保存 best；每个 RL 对照 65,536 环境步。每 8,192 步验证，固定验证请求种子；最终 30 个独立测试 episode。BC-only 只评估，不计为 RL 学习曲线。
9. 成本与日志：保存源采样时间/环境步、BC 更新和墙钟时间、微调环境步/墙钟、reward、成功率、成功请求延迟、超时拒绝率。教师训练成本另列，不能把专家免费处理。沿用 W&B offline 和本地 CSV，后续同步。
10. 运行：本地实现与 CPU smoke（真实采样、NLL 更新、跨规模加载、全量/冻结微调、断点恢复）通过后提交推送；服务器只拉取。新实验使用独立输出目录，串行运行额外实验并控制 worker 数，避免干扰正在运行的三规模任务。

## 验收

- 数据的步连续性、终止标志、奖励和区间计数、排空总账可核对；校验和及种子拆分验证。
- 学生分布与 BenchMARL 分布的 log probability 一致，极端合法动作有限；NLL 训练梯度有限且能下降。
- 2→3、3→5 等数量变化可以加载同一 base；actor 独立、critic 新建；head-only 的 backbone 不变。
- CPU 保存/恢复跑通，原有测试通过；部署后确认新实验有真实数据和参数更新。
- 如 BC 模仿或在线收益不佳如实保留，不通过筛除失败样本、测试集调参或更改基线保证优势。

## 使用方式

```sh
# 独立导出和训练（多个 manifest 可训练同一个 base）
.venv/bin/python -m edge_sim_learning.pretrain collect \
  --teacher outputs/source/best-stochastic.pt --output outputs/bc/data --workers 2
.venv/bin/python -m edge_sim_learning.pretrain fit \
  --manifests outputs/bc/data/manifest.json --output outputs/bc/base --mode offline
# 中断后的 BC：保持其它配置一致，加 --resume 并提供尚未完成的总 epoch 预算

# 独立微调：目标配置可以拥有不同的集群数量；--head-only 为可选消融
.venv/bin/python -m edge_sim_learning.cli train --method MAPPO-no-context \
  --scenario target.json --actor-init outputs/bc/base/best.pt --hidden-size 256 \
  --output outputs/bc/finetune --wandb-mode offline
# PPO 中断续训只传 --resume last.pt，不再传 --actor-init；自动恢复冻结模式。

# 完整四分支实验，包含独立测试集、成本账和结果表
.venv/bin/python scripts/run_pretrain_transfer.py \
  --teacher outputs/source/best-stochastic.pt --output outputs/bc-transfer-v1 \
  --device cuda --workers 2 --wandb-mode offline
```

`--smoke` 使用 2 集群 smoke 教师迁移到 3 集群，每个 RL 分支 32 步、监督采样 3+2 集、BC 2 轮；正式默认教师应为 small 三集群，目标为 medium 五集群。先冻结教师文件，再运行；恢复 suite 时继续使用同一教师快照。`complete.json` 表示四分支全部评估完成，不代表学到的策略优于基线。

输出包括 `data/manifest.json` 与分集 `.pt`、`base/{last,best}.pt`、三组 PPO checkpoint 和 `*-test/evaluation.json`、`BC-only/evaluation.json`、`comparison.csv/json`、`costs.json`。数据导出按完整 manifest 条目恢复；BC 保留最近完整 epoch，PPO 保留最近完整采样批次。

## 本机验证记录

2026-09-20：完整测试 196 passed / 1 skipped（本机无 NVIDIA）；新增测试包含真实教师训练、完整 episode 导出、手算奖励、log probability 一致性、BC 保存恢复、2→3 集群迁移、head-only 续训后 backbone 不变。`outputs/transfer-smoke-v2/complete.json` 的四分支流程完成；相同测试请求和排空检查通过。全量微调的 backbone 参数实际变化，head-only 变化为零。

另用已有 small 教师采集 3 个训练集 episode 和 2 个独立验证 episode，在 90 个 BC minibatch 更新后，验证 NLL 从 6.600 降至 5.944，教师 KL 从 0.843 降至 0.211。这是管道验证数据，不能作为正式迁移收益或收敛证明。正式实验保持前述预先确定的预算和评价口径。
