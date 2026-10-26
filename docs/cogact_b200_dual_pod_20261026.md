# CogACT B200：80,000步从单Pod切换到双Pod

用户要求在第80,000步检查点完整保存后，将已有训练扩展为两个Pod共同恢复训练。
整个过程保留Pod、Job及PID1；训练命令直接在现有容器中执行。

**已完成切换**：2026-10-26北京时间09:58检查时，双Pod已从80,000步恢复至
80,140步，16张B200共同训练，W&B沿用原run。下方包含配置、恢复方式和实测回执。

## 资源与训练口径

| 角色 | Pod | IP | 物理节点 |
|---|---|---|---|
| node rank 0 | liluo-agibot-2-server-0 | 10.45.152.131 | slc01-cl02-hgx-0061 |
| node rank 1 | liluo-agibot-1-server-0 | 10.44.59.231 | slc01-cl02-hgx-0034 |

Context为`oidc@msr02`，namespace为`bonete05`，container为`server`。
每个Pod有8张B200；代码和独立环境均在各Pod本地的
`/workspace/internw0-delta-yuming-20261023`，共享结果和准备文件在`/data/yuming/internw0-delta`。
原始数据`/data/liluo/msra_process`始终只读。

新配置`configs/task/cogact_bs256_100k_2node.yaml`继承原正式训练参数：
每卡batch从32改为16，共16卡、梯度累积1，**全局batch仍为256**。
总更新数100,000、每5,000步保存、学习率5e-5及原有cosine调度、BF16、ZeRO2、
梯度检查点均保持。继续32步绝对关节动作、关闭RTC及4D蒸馏、全部修正版12份数据训练。
固定训练样本loss评估由8卡×每卡2个样本改为16卡×每卡1个样本，
保留相同16个样本及各样本评估随机种子，每1,000步评估一次。

## 完整状态恢复

ZeRO2将FP32主权重和Adam动量分片保存在各个GPU的检查点中。
当前DeepSpeed0.19.2不支持直接用8卡分片恢复16卡，必须先离线重新划分。
转换工具`tools/cogact_reshard_zero2.py`：

- 只接受已停止写入的纯数据并行ZeRO2状态，拒绝不支持的参数排序或混合分片。
- 根据参数布局重新划分FP32主权重、一阶和二阶动量，并重新计算对齐填充。
- 保留Adam更新计数、学习率、调度器、trainer step、epoch和数据位置。
- 每个目标分片保存后重新读取并检查三组张量的SHA256，全部完成才写`reshard_complete.json`。
- 转换源保持原样，输出目录必须不存在；训练器拒绝未完成验证的转换结果。
- GPU数量变化后，随机流无法与原8卡逐位接续；使用记录在状态文件中的基础种子，
  为新拓扑的各rank确定性重设随机流。后续相同拓扑恢复照常恢复保存的随机状态。

由于全局batch保持256，`batch_in_epoch × 256`仍指向同一数据位置。
本次恢复使用同一个W&B run：
https://wandb.ai/xymliu/cogact_baseline/runs/cgb200-20261023-100k 。
实际新配置会更新至W&B；API key仅通过进程环境注入，不写入仓库或回执。

## 已完成的准备验证

- 两个Pod均为Running、重启次数0、PID1为sleep；新Pod的8张B200原本空闲。
- 新Pod已安装独立环境，核心版本与原训练一致：torch2.10.0+cu128、
  torchvision0.25.0、DeepSpeed0.19.2、transformers5.13.0、PyAV16.0.1、wandb0.23.1。
- 新Pod检查12份数据的起点、尾部和跨视频分片窗口，共36个真实样本，全部通过。
- 15项本地测试通过，包含分片边界、对齐填充、源文件不变、损坏状态拒绝、数据位置保持。
- `tests/integration/zero2_resume_probe.py`在新Pod上用真实DeepSpeed运行：
  2卡更新3步后保存，转换为4卡恢复，再更新1步。
  两个参数组的FP32权重及两组Adam动量，与原2卡连续更新至第4步的结果**逐位一致**。
- 对正式第75,000步检查点只读预检通过：8个源分片、6,211,551,888个参数、全局batch256。
- 两Pod原有NetworkPolicy只允许各自Job内通信。新增
  `configs/k8s/cogact_dual_pod_networkpolicy.yaml`，仅对这两个Job的server-0开放彼此的TCP通信。
  未创建或修改Pod/Job规格。
- 原Pod缺少libibverbs，已安装libibverbs1、ibverbs-providers、ibverbs-utils及两个libnl依赖。
  两端选择8个400Gb/s InfiniBand适配器，NCCL GPU直接通信验证通过；
  每端1张卡、128MiB all-reduce十次，平均约6.6毫秒。该数值不是完整16卡训练步耗时。

准备、检查与日志保存在共享目录`transition-20261026`。
小型验证回执另存`docs/results/cogact_b200_dual_20261026/`。

## 切换执行方式

原Pod中的监控脚本等待`trainer_state.json`显示80,000步，且训练日志出现完整状态保存后的
`[ckpt] step=80000`。随后检查模型、8个优化器分片、调度器和各rank随机状态文件，
只向核对过身份的训练launcher PID192734发送SIGTERM，不操作PID1或Pod生命周期。
监督进程和停训回执保存在`transition-20261026/watcher_*.json`。

确认原训练所有GPU worker退出后，保留8卡原始完整状态并转换至：
`/data/yuming/internw0-delta/transition-20261026/state16-step80000`。
两个Pod分别在各自代码目录执行同一脚本，`NODE_RANK`分别为0和1：

```bash
export NODE_RANK=0  # 另一个Pod设为1
export MASTER_ADDR=10.45.152.131
export MASTER_PORT=29526
export RUN_ID=b200-joint-bs256-100k-20261023
export RESUME_STATE_DIR=/data/yuming/internw0-delta/transition-20261026/state16-step80000
export WANDB_RUN_ID=cgb200-20261023-100k
export WANDB_RESUME=allow
bash scripts/cogact_b200_dual_node.sh
```

后台监督进程会将命令输出、退出码与启动回执持久化到共享盘。
脚本使用各Pod独立的文件锁防止重复启动，并在发现已有GPU计算进程时拒绝启动。

## 实际切换与恢复结果

以下时间均为2026-10-26北京时间（UTC+8）：

- 09:39:55达到80,000步，09:42:01日志确认完整状态保存完成。
  09:42:03检查文件和进程身份后，仅停止原训练launcher及其worker。
- 原8卡完整状态保留为
  `/data/yuming/internw0-delta/transition-20261026/state8-step80000`。
  该目录保留模型、优化器、调度器和随机状态，可用于回退；转换没有修改源张量。
  `checkpoint_80000_ready.json`记录的是移动前的检查路径，原`state/latest`将在85k保存时重建。
- 09:48完成16个新优化器分片的写入、回读和SHA256校验；09:49两个Pod同时启动。
- 09:52:54实际训练日志确认模型、优化器、调度器及数据位置恢复到80,000步。
  数据位置为`epoch=2, batch_in_epoch=26146, sample_offset=6693376`；
  基础随机种子为1280042。训练代码为`bfd1d67b0dc3c01891abc6a0fb99978ca53ee8fb`。
- 09:58检查至80,140步，16个rank的GPU指标均记录至此步，全部训练loss和梯度范数有限。
  W&B云端已确认接收至80,130步，状态为running，配置为每卡16、总100k步、每5k保存。
  两Pod均为Running、重启次数0、PID1为sleep。

| 恢复后指标 | 实测值 |
|---|---|
| 第80,140步总loss / action loss | 0.077890 / 0.001153 |
| 第80,140步梯度范数 / 学习率 | 0.04926 / 5.6486e-6 |
| 第80,010→80,140步平均耗时 | 2.009秒/步，127.44样本/秒 |
| 每卡PyTorch峰值已分配 / 保留显存 | 56.47 / 71.89 GiB |
| 每卡nvidia-smi进程总占用 | 约76.48 GiB |

耗时排除恢复后的前10步编译预热，保留之后所有慢步；尚未覆盖双Pod的5k保存周期。
按这130步的平均速度，剩余19,860步纯训练约11.1小时，另需加上检查点保存和评估时间；
长时间数据读取及共享盘波动可能改变预计完成时间。

完整实测见[dual_resume_health.json](results/cogact_b200_dual_20261026/dual_resume_health.json)，
同目录保存完整80k检查、分片转换及两端启动回执。

当前运行目录为
`/data/yuming/internw0-delta/runs/cogact/b200-joint-bs256-100k-20261023`。
**双Pod日志使用`train_dual_rank0.log`和`train_dual_rank1.log`**；
`train.log`仅保留单Pod阶段记录。JSON/CSV指标沿用原路径，GPU指标扩展至rank0–15。
共享目录`transition-20261026`中的`dual_rank0_launch.json`和`dual_rank1_launch.json`
记录当前launcher PID718980和2731395；两端退出时分别写`dual_rank{0,1}.exitcode`，
rank0另写运行目录的`train.exitcode`和`completion.json`。训练正常完成后仍保留Pod。
