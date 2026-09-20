# DEPPO-adapted：内容服务场景与 BenchMARL

实现分三层：`edge-sim-models` 定义领域 DTO，`edge-sim-simgrid` 在独立进程中运行物理仿真，`edge-sim-learning` 负责观测、历史、奖励、采样和 BenchMARL MAPPO。SDK 不导入 NumPy、Torch、PettingZoo 或 BenchMARL；只安装 SDK 不会安装学习栈。

验收测试、三种子小预算对照和 1/2/4 worker 数据见 [本机验收记录](deppo-validation.md)。完整 JSON 场景示例位于 `configs/learning/`。后续租用机器的 CUDA 检查和训练结果见 [Linux / 4090 验证记录](linux-validation.md)。最新本地优化、正确性对照和吞吐数据见 [采样性能验证](sampling-performance.md)。

## 安装与入口

先按根 README 构建 SimGrid，再执行 `uv sync --locked --all-packages`。学习栈固定为 BenchMARL 1.5.2、TorchRL 0.10.1、TensorDict 0.10.0、PyTorch 2.9.1、PettingZoo 1.25.0，完整解析结果见 `uv.lock`。TorchRL 会提示其上游测试版本是 PettingZoo 1.24.3；本项目另行执行了 1.25.0 的 Parallel API 和 TorchRL spec 检查。

CPU smoke（16 个环境步，包含采样、更新、排空评估、checkpoint、CSV 和 W&B offline）：

```sh
uv run --package edge-sim-learning edge-learn train \
  --profile smoke --output outputs/deppo-smoke \
  --episodes 2 --workers 1 --batch 16 --epochs 1 --minibatch 8 \
  --eval-interval 16 --eval-episodes 1 --wandb-mode offline
```

正式实验在 Linux + NVIDIA GPU 上执行，每个种子分别运行：

```sh
uv run --package edge-sim-learning edge-learn train \
  --profile small --seed 0 --device cuda --wandb-mode online \
  --output outputs/deppo-small-seed0
```

种子用 0、1、2；对照加 `--method MAPPO-no-context`。W&B 项目 `ecsai-deppo`，entity 取 `WANDB_ENTITY`，online 使用已登录账号或标准 `WANDB_API_KEY` 配置。必须显式指定 online/offline，不会默默切换。未在未配置账号的机器上执行 online 验证。

默认预算 2048 episodes × 128 cycles = 262144 **环境步**，4 workers、每批 512 环境步、10 轮 minibatch 64 更新。agent 步数单独记为环境步数乘调度集群数。批次要求每个 worker 包含整数个 episode，保证保存/恢复边界明确。模拟和策略采样在 CPU；loss、GAE、反向传播与优化器在 `--device`。Torch 单线程避免每个采样进程重复争抢 CPU。

```sh
# 使用同一固定评估请求种子集，末尾继续排空；读取 checkpoint 自带场景和模型
uv run --package edge-sim-learning edge-learn evaluate \
  --checkpoint outputs/deppo-smoke/best.pt --wandb-mode offline --output outputs/deppo-eval

# 启发式，与学习方法使用相同评估种子和统计口径
uv run --package edge-sim-learning edge-learn evaluate \
  --profile small --method local --wandb-mode offline --output outputs/local-eval
# --method random / forward

# 恢复参数、优化器、计数器、随机状态；episodes 表示新的总预算
uv run --package edge-sim-learning edge-learn train \
  --profile smoke --episodes 4 --workers 1 --batch 16 --epochs 1 --minibatch 8 \
  --eval-interval 16 --eval-episodes 1 --wandb-mode offline \
  --resume outputs/deppo-smoke/last.pt --output outputs/deppo-resumed

uv run --package edge-sim-learning edge-learn benchmark \
  --profile small --workers 1 2 4 --steps 128 --wandb-mode offline --output outputs/deppo-benchmark

# 自动跑两个学习方法各三种子、三种启发式、10 个相同评估 episode，以及 1/2/4 workers
uv run --package edge-sim-learning python scripts/validate_learning.py
```

checkpoint 恢复在完整 episode 边界重建冷缓存物理进程，不恢复进行中的 native SimGrid 活动。训练场景/方法必须匹配；工作负载 episode 序号从已完成环境步继续。checkpoint 是本地 PyTorch 对象，只加载可信文件。最后和最佳 checkpoint 各保存一份，最佳依据排空后成功率，平局按成功请求平均延迟。

## 场景与控制合同

`small/medium/large` 分别为 3/5/7 个集群、10/20/30 个缓存、300/1000/2000 请求/s。`--scenario file.json` 可以直接提供完整或部分 `ScenarioConfig`（未写字段取默认值）。目录、拓扑、到达流各用独立 NumPy SeedSequence；动作随机数与生成请求无关。日志保存每个实际评估 episode 的 workload seed、请求数和理论交付负载比。

默认内容 1000 项，Zipf 指数 1.0，大小均匀 0.5–7.5 **十进制 MB**，请求 Poisson 到达，截止 1s。每缓存 100 MB，回源和交付分别为 125000000 byte/s，即 1 Gbit/s，传播时延 20ms/5ms。配置可通过 `backhaul_bandwidth_bytes_s` / `delivery_bandwidth_bytes_s` 分别覆盖容量；未指定时使用公共默认 `bandwidth_bytes_s`。`backhaul_concurrency` / `delivery_concurrency`、`backhaul_waiting` / `delivery_waiting` 可分别覆盖并发和等待上限。每个缓存有自己的回源链路和交付链路；同一缓存到不同区域的交付共享该缓存交付链路。没有隐藏骨干瓶颈。理论交付负载比为 λ × 按内容热度加权的平均对象大小 / 所有交付链路总容量；数值大于 1 时不会降低请求率。

每个调度器 1ms 服务时间、100 个 FIFO 等待位；每缓存每方向 8 个活动传输、50 个 FIFO 等待位。容量不含活动项。传输进度、共享链路速率及传播延迟均由 SimGrid `Comm` 决定。未设置 `TransferPoolSpec` 限制时保留无限并发；已有 DAG 模式保持原行为。

每个缓存的同对象回源合并，各请求保留自己的截止时间、结果和交付流。最后一个等待者超时才取消合并回源。完整对象回源后才开始交付；不模拟分片或播放。LRU 在请求命中时刷新，冷缓存开局；尚在交付的对象暂时固定，不能驱逐。没有空间容纳新对象时明确拒绝 `storage_capacity`，计入拒绝率而非队列溢出率。回源等待项按唯一对象传输计数，交付等待项按请求计数。

SDK 的 `RunSpec(control_mode="window", content=ContentServiceSpec(...))` 选择周期控制。`Session.advance_window(until_s, WindowControl(...))` 或 `BatchRunner.submit_window/recv_ready` 驱动；同一 run 禁止混用 `advance/apply`。进程内控制器用 `Serve(request_id, cache_node)` 内容服务操作，保持对象 ID 不变，无虚构 DAG 计算阶段。控制 DTO 不包含 reward 或网络参数张量。

边界先处理已有传输完成、截止事件与已开始的调度服务完成；到达时间恰等于边界的新请求在下一次调用中接纳。精确截止时交付完成判成功。新参数作用于下一周期的服务决策；已提交给某个传输池的工作不重新选择节点。转发不重置截止时间，最多一次，目标队列满立即拒绝。邻居负载广播只在周期开始更新，按 `0.5 × 缓存平均服务负载 + 0.5 × 调度队列占用率` 排序，同负载按 ID。缓存分配使用交付容量加权的 smooth weighted round robin。

`WindowResult.view` 只返回存活请求与本窗口新结束请求，避免每步传输完整历史；`latencies_s` 也只包含本窗口成功延迟。`Session.result()` 可读取全体请求结果。链路字节/容量时间与队列计数为累计值；`WindowResult.link_bytes`、completed/timed_out/rejected 为区间增量。trace 开启时记录进入服务、等待、完成、取消、溢出与转发事件。

## 学习合同

每 agent 当前观测 13 维：调度等待占用率、所属缓存平均回源负载/交付负载、大小直方图 5 档、剩余期限直方图 5 档。负载分母为对应活动上限加等待上限。直方图只统计调度等待请求，区间为归一化后的 `[0,1]` 等宽五档，按队列总数归一化；空队列为零。

PettingZoo observation 实际打包为 150 维：当前 13 + 最近 8 个 `(13 observation,4 action)` 对 + 1 有效长度。历史按时间顺序从前填入，剩余位置补零；每次 reset 清空。critic 的 `state` 单独提供全体 agent **当前**观测的拼接，没有历史。actor 仅读取自己的 150 维。

DEPPO actor 每 agent 独立：GRU(17,64)，取有效长度位置输出；空历史强制零；拼接当前 13 维后经 ReLU MLP(128,128)。不把 GRU 隐状态保存在 collector 中，PPO minibatch 可以精确重算同一个窗口。MAPPO-no-context 接收同一观测但只使用前 13 维。两者共享一个中心 critic ReLU MLP(128,128)。动作由 TanhNormal 限制在 `[-5,5]^4`。阈值用 `w·x+b >= 0`，等价于 sigmoid ≥ 0.5。

学习率 3e-4，gamma .99，GAE λ .95，PPO clip .2；BenchMARL 默认 entropy coefficient 0、critic coefficient 1、Adam epsilon 1e-6、梯度范数裁剪 5。初期未针对本场景调参。

奖励 = .5 × 本周期成功数/结束数 + .5 × 本周期链路实际发送字节/容量时间。结束数包括成功、超时、拒绝，无结束请求时第一项零。所有 agent 获得同一奖励。128 周期后 `truncated=True, terminated=False`；TorchRL GAE 使用 terminated 处理 bootstrap。训练截断的未完成请求单独记录，不伪装超时。评估使用同一截断前 return，同时停止新到达、沿用最后动作排空至全部完成/截止，再计算服务指标。延迟仅统计成功请求。

## 产物与计时

每次训练目录包含 `config.json`、长表 `metrics.csv`、`last.pt`、`best.pt`、`evaluation-*.json`、BenchMARL CSV 和 `wandb/offline-run-*`。W&B 标量默认横轴为 `env_steps`，同时保存 `wall_s`，可在 UI 切换。每 8192 步以及训练结束做固定评估，默认 10 个 episode，种子从 1000000000 起，与训练种子隔离。

指标覆盖 reward/return、actor/critic loss、entropy、成功/超时/拒绝/溢出、未完成、成功延迟 p50/p95/p99、命中率、回源/交付字节和利用率、队列与转发、采样与请求吞吐。

`simulation_wall_s` 为 worker 内周期执行耗时；`ipc_wall_s` 为提交到收到回应减去 worker 执行耗时，包含进程等待与串行打包，不是纯网络传输时间。推理计时来自 collector policy forward hooks，训练计时覆盖 GAE、buffer/minibatch 与优化。多 worker 的 worker-seconds 可以大于墙钟秒，不应相加当作端到端时间。benchmark 单独记录启动耗时，其采样耗时包含 episode 重建冷缓存进程的成本；smoke episode 很短，进程重启成本可能主导吞吐。

## 论文—实现差异

| 项目 | 本实现及解释 |
| --- | --- |
| 五维动作 | 四维参数化转发，移除带宽比例动作；物理链路容量固定配置 |
| 拓扑与链路 | 有线回源、独立用户交付，显式链路竞争，无隐式全局无线瓶颈 |
| 请求数据 | 合成 Poisson + Zipf；不声称复现论文表格数值 |
| 传输/队列 | FIFO 等待、并发共享带宽、合并回源、独立交付、显式超时取消 |
| 缓存 | 所有方法同一冷启动 LRU；完整对象，交付中对象固定 |
| reward | 团队周期成功率与实际链路利用率各 .5，非原奖励逐项复写 |
| 历史网络 | 显式 8 对窗口、GRU64、ReLU MLP128×2，是补充的可复现实验定义 |
| 对照命名 | `MAPPO-no-context`，不称为原论文 DD；Random 为进程内逐请求随机本地/转发 |
| 截断 | 训练正确 bootstrap；评估排空，单独报告截断未完成数 |

本阶段不实现监督预训练、完整离线 rollout 导出或新 UI。未来采样/预训练可复用 13 维观测、4 维动作、8 对窗口和独立环境，不改物理 SDK。
