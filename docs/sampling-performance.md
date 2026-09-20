# 采样性能优化与本地验证

2026-09-20。本次只在本机修改、验证和提交；没有连接或启动租用服务器。
优化前基准是 `3a695df`，包含此前已完成的 DEPPO 场景适配实现。

## 已完成的优化

- **批量传输完成通知**：维护持久 SimGrid `ActivitySet`，消费 `wait_any_for` 返回的完成活动，用 `test_any` 收集同一时刻其余完成项。去掉每个事件逐个 `Comm.test()` 的跨语言调用与集合重建。仍按传输创建顺序处理同时完成项，先释放全部完成项，再判截止、接纳新工作。
- **按需读取真实字节进度**：在窗口边界、传输完成或取消前，从 SimGrid 读取剩余量并累计差值。无需每个内部事件扫描全部传输，未完成与取消传输的已发送字节仍计入利用率。没有改变带宽、时间步、请求量或网络求解器。
- **索引代替全表扫描**：缓存主机、集群归属、加权轮转权重；维护请求到传输的索引、内容被交付占用的计数；转发目标每次广播计算一次。
- **精简窗口响应**：SDK 的 `advance_window(..., scope="scheduling")` 和 `submit_window` 同名参数返回调度等待请求、池状态、完整计数及区间完成延迟，省去传输明细和无关请求。响应 `view.scope` 显式标记范围。默认仍为 `full`，`inspect()` 保留完整实时查询。
- **采样与编码重叠**：每个 worker 返回后立即编码，其他 worker 同时继续仿真，最终按原 worker 槽位组装 TensorDict。没有重排 PPO/GAE 时间轴或改变策略版本。
- **减少封装开销**：空队列跳过直方图计算，缓存静态映射，直接填充历史窗口，一次计算三个精确延迟分位数，每个 worker 一次分配指标 tensor。
- **避免重复冷启动**：TorchRL 规格探测与第一次采样 reset 若指向同一个尚未推进的 episode，复用它；不同种子、episode 或已经推进的仿真仍重新创建进程和冷缓存。

## 测量口径

`simulation_wall_s` 是 worker 内推进和快照生成的墙钟时间，包含 Python 调度，不能称作纯 SimGrid 求解时间。
`rpc_overhead_wall_s` 是发起请求到父进程收到响应的时间减去 worker 时间，仍包含序列化、进程调度和父进程未及时读取造成的等待，不能称作纯 IPC 时间。`ipc_wall_s` 保留为兼容别名。
`encoding_wall_s` 单独记录环境观测、历史、奖励和指标编码时间。跨 worker 相加是 worker 秒，不是并行任务的墙钟。

benchmark 分别报告启动、采样、episode reset 耗时以及包含启动的吞吐；最后一步结束后不再启动无用的新 episode。
同步批次仍在最慢 worker 完成后返回，这是当前 BenchMARL on-policy 收集边界。没有宣称已经实现完全异步 actor/learner。
SimGrid 每个真实 episode 仍用独立进程，保留引擎生命周期、时间原点与冷缓存隔离；没有为省启动开销复用污染状态。

## 正确性验证

- 完整测试 **174 passed**（16.78 秒）；Ruff、`git diff --check`、`uv lock --check` 通过。含 PettingZoo、TorchRL 合同、局部观测隔离、历史重置、GRU 梯度、截断 bootstrap、CPU 更新、保存恢复与 W&B offline 文件检查。
- 新增部分传输/取消字节、重复查询不重复计数、精简/完整快照一致性、多个同时交付精确截止、冷启动复用和 batch/独立环境轨迹对齐测试。
- 使用旧实现源码快照和相同场景/动作，逐窗口对比状态、完整请求结果与事件序列：small 的 0/1/2 种子各 128 周期（threshold/random/local），large 的种子 0、32 周期（random，2,000 请求/秒）。
- 全部离散状态、队列、事件顺序一致；浮点计数改变累加顺序，对照使用绝对容差：字节 `1e-3`、时间 `1e-11` 秒。实际最大差异约 `7.5e-5` 字节、`2.9e-14` 秒。不能以 JSON 哈希不同判定物理行为改变。
- large 对照到截断共到达 6,343 请求，2,457 成功、879 超时、1,731 拒绝，其余未完成；没有靠减少压力得到加速。
- 本地加载此前服务器的 CUDA checkpoint，映射到 CPU 后从 512 步续训到 1,024 步：4 workers、small/128 周期、每批 512 步、10 轮更新、minibatch64。36 个保存的参数张量变化，全部有限；完成评估、last/best checkpoint 和 W&B offline 日志。该次新增批次采样 3.37 秒，推理 0.11 秒，更新阶段 0.40 秒。不能拿本机时间直接除以前服务器的 51.60 秒作为加速倍数。
- `MAPPO-no-context` CPU smoke（2 workers、32 步、一次更新）也完成训练、评估及 checkpoint，产物位于 `mappo-smoke/`。

## 本机吞吐

macOS 26.5.1 / arm64，CPU，锁定依赖；small 配置、固定种子 0、每 worker 32 周期，相同随机参数动作。优化前后各运行三遍，以下取中位数；测速时没有并行运行本项目的其他测试。采样吞吐不含初次启动，包含环境编码/TorchRL 封装；不是仅测仿真内核。

| workers | 优化前 环境步/s | 优化后 环境步/s | 倍数 | 优化前/后启动秒 |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 14.88 | 112.73 | 7.58× | 0.378 / 0.163 |
| 2 | 20.04 | 205.67 | 10.26× | 0.695 / 0.343 |
| 4 | 22.59 | 334.66 | 14.82× | 1.449 / 0.648 |

4 workers 共 128 步，采样由 5.667 秒降为 0.382 秒；将启动计入约为 7.116 秒对 1.030 秒（约 6.9×）。吞吐区分环境步和 agent 步，不乘 agent 数。不同 worker 行的总工作量不同，不能据此声称严格线性扩展。
原始三次记录为 `before-{0,1,2}/benchmark.json` 和 `after-final-{0,1,2}/benchmark.json`，汇总为 `benchmark-summary.json`。

## 复现

原始本地结果保存在 `outputs/sampling-optimization/`（测试日志、before/after benchmark、逐窗口语义文件、对照报告、训练产物）；这些大文件不进入 Git。

```sh
.venv/bin/python -m pytest -q
.venv/bin/python -m edge_sim_learning.cli benchmark --profile small \
  --workers 1 2 4 --steps 32 --seed 0 --wandb-mode offline \
  --output outputs/my-sampling-benchmark
.venv/bin/python scripts/profile_sampling.py --profile small --cycles 128 \
  --seed 0 --policy threshold --output outputs/my-kernel-profile --cprofile
```

对照旧版本时，将旧版本各 `packages/*/src` 目录放入 `PYTHONPATH`，运行相同的 `profile_sampling.py` 生成 `semantics.json`；新版本命令加 `--compare 旧结果/semantics.json` 会验证每个字段，失败以非零状态退出。开启 cProfile 的运行只用于归因，不混入性能中位数。

服务器已拉取提交并完成 Linux 1/2/4 worker benchmark、CUDA 训练与续训，见 [Linux 采样优化验收](linux-sampling-validation.md)。macOS 本地结果不能直接代替租用 Linux 的加速倍数或 GPU 利用率测量。依赖锁文件没有因本次性能优化改变。
