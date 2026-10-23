# CogACT 八卡 A100 训练验证

本实验使用 `eaiws` 工作区、`palisades33` 的单节点 8 张 A100 40GB。
amlt project 为 `internw0-delta`，存储为 `eaidatashare/v-xingyumliu`。
目标是验证完整模型能否更新、测量当前配置的 batch 上限，并检查短程拟合的 loss 趋势。
不启用 W&B；不改变已经确认的动作语义；不是正式训练配方。

## 输入和模型

从已经核对的 CogACT 录制 `15/20260506_162713` 的片段 `[3142,3447)`
选取 64 个不同、完整且有历史图像的窗口。它们有重叠，来自同一片段。
归一化仍使用该片段的统计。来源、帧索引和文件 SHA-256 校验值保存在输入 manifest。
这是一项小数据拟合检查，不能据此报告验证集效果或推广到全部 12 份数据。

每个样本包含 32 步、80 维动作，9 帧 384×256 三视角画布，起点及历史帧，
以及视觉语言模型的三张当前图像。关节为下一帧绝对目标，夹爪为当前命令；不输入外参。
实际加载完整预训练模型，视频编码器和视觉语言模型在线运行并冻结权重。
文本使用已经生成的真实指令缓存。没有替换成小模型或模拟编码器。

训练使用 bf16、梯度检查点、PyTorch AdamW，以及 DeepSpeed ZeRO-2。
ZeRO-2 将梯度和优化器状态分摊到 8 张卡，模型参数仍在每卡保留副本。
梯度累积为 1，因此全局 batch = 每卡 batch × 8。未使用 CPU 卸载。
必要的策略权重必须严格匹配，缺失或形状不符会报错。

## 三个测试阶段

1. 从每卡 batch=1 开始，每个候选运行 3 次完整前向、反向和优化器更新。
   检查 loss、梯度范数和更新后的参数是否有限，并检测实际参数变化。
2. batch 翻倍直到 CUDA 显存不足，再二分定位相邻的成功与失败整数。
   每次尝试都启动新的 8 进程训练，避免复用 OOM 后的 CUDA 状态。
   超时或代码异常不会被当成显存上限。搜索最多到每卡 64；如果仍成功，只报告下界。
   三步探测得到的是本配置的短程容量，不能保证所有数据或长程训练都稳定。
3. 取留有余量的每卡 batch：`max(1, min(4, 最大成功 batch // 2))`，训练 200 步。
   第 100 步保存完整训练状态并退出工作进程，再启动新进程恢复到第 200 步。
   每 25 步在固定的 8 个训练窗口上，用固定的 3 组噪声测量各项 loss，减少随机采样
   对趋势比较的影响。报告初末值和随步数变化的斜率；训练成功不要求 loss 一定下降。

检查点包含可训练模型、分片优化器、学习率调度器、步数及每个 rank 的随机状态。
冻结的编码器由原始权重恢复，不重复保存。所有分片关闭后再发布完成标记；恢复时核对
配置、数据指纹和文件大小。采样由步数确定，恢复后延续相同的样本序列。

## 结果和排错

结果先写入节点本地 `$COGACT_LOCAL_ROOT/results/probe/`，所有训练工作进程退出后，
由 `scripts/cogact_a100_probe_stage.sh` 顺序复制到 `$AMLT_OUTPUT_DIR/probe/`。
可用 `COGACT_RESULT_SUBDIR` 设置独立的恢复结果目录。挂载存储的上传延迟不再参与
GPU 进程之间的同步；检查点完成标记在对应文件复制完成后才上传。
训练过程中通过平台日志或 SSH 查看进度；本地文件在上传前不具备节点抢占保护。

结果目录包含：

- `capacity.csv`：各 batch 的状态、PyTorch 峰值分配/保留显存、硬件使用量峰值和耗时。
- `capacity/batch_*/summary.json`：每张卡的详细结果；硬件显存每秒采样一次，可能漏掉瞬时峰值。
- `training.csv`、`evaluation.csv`：逐步训练指标和固定噪声评估曲线。
- `result.json`：容量、各项 loss 趋势和断点恢复是否通过。
- `status.json`：当前阶段及错误；只有所有要求的阶段成功才会标记 `passed`。
- `training/checkpoints/`：可恢复的完整训练状态；`complete.json` 标记保存完成。

工作进程日志先写节点本地，关闭后复制到结果目录；主日志每 30 秒输出阶段进度。
每次工作进程最多运行 4 小时，超时保留日志并报错。
环境安装和训练均在前台包装脚本内；结束或失败后按用户已有偏好保留节点 20 小时，
再返回原始退出码。此时平台显示 Running 可能仅表示排错等待，实际进度以 `status.json`
和 `debug_hold.json` 为准。保留期间仍占用 8 张卡，平台仍可抢占。

输入权重预先上传，通过大小和 SHA-256 校验后复制到节点本地；运行时不下载 Hugging Face
权重。Python 依赖安装在节点独立虚拟环境，实际版本写入 `pip-freeze.txt`。

本地准备入口：

```bash
.venv/bin/python tools/cogact_probe.py prepare
.venv/bin/python -m pytest -q tests/test_cogact_joint.py tests/test_cogact_probe.py
```

集群配置为仓库根目录的 `amlt_cogact_a100_probe.yaml`，由 `amlt template` 生成后填写。
模型与运行代码均作为本次作业的代码快照上传。

## 2026-10-22 首次提交记录

实验 `cogact-a1008-probe-20261022-073733` 的作业 ID 为
`amusing_cabbage_37y1v6vv6z`（[平台页面](https://aka.ms/amlt?q=lbbpl)）。
提交命令返回成功，但平台随后在创建运行环境时失败；训练入口没有执行，也没有训练指标。
后端错误为 `AuthorizationError`：
`Authentication failed for container registry msrcommonacr.azurecr.io`。

镜像漏洞检查已经通过；这里失败的是集群拉取镜像的身份认证。
工作区主身份为 `eaiws4storage`，其 principal ID 为
`66cb1243-0045-4b88-a555-a972a96263bd`。只读查询未查到它或工作区系统身份
在 `msrcommonacr` 上的直接或继承角色，也没有匹配的旧镜像凭证连接。
`eaiws` 自带的镜像仓库为 `eaiacr.azurecr.io`；上述两种身份均在其资源组上有
Contributor 角色。恢复前先检查这个已授权仓库内的镜像能否通过同样的安全检查。
首次失败记录保留，不能将它当作模型训练失败或 batch=1 显存不足。

已检查 `eaiacr` 中的两个候选镜像：

- `aifx/acpt/stable-ubuntu2204-cu126-py310-torch28x@sha256:22c27788061d07934e54ab5672e8697bdbfe053933b132b8aa4aded0ad1d48a0`
- `semantic-wam/acpt-pytorch-2.13-cuda12.6-sfi@sha256:35d647084ba03ed664f503bbe02bccc8c8978f2073ec082dcc06cf57625eb0a5`

两次 `amlt scan-image ... --max-job-duration-days 2` 均返回非零：
`No vulnerability scan found`。没有使用这些镜像提交，也没有关闭或降低扫描要求。

原镜像的权限修复命令如下，**尚未执行**。需要用户批准，并由对该仓库有角色授权能力
的账号执行。`AcrPull` 为只读拉取角色；由于仓库使用传统角色模式，此授权覆盖整个仓库，
不只是本次镜像。

```bash
az role assignment create \
  --assignee-object-id 66cb1243-0045-4b88-a555-a972a96263bd \
  --assignee-principal-type ServicePrincipal \
  --role AcrPull \
  --scope /subscriptions/de12e4e9-1d99-4689-a06a-45bb7a59f083/resourceGroups/msr-common-acr/providers/Microsoft.ContainerRegistry/registries/msrcommonacr \
  --subscription de12e4e9-1d99-4689-a06a-45bb7a59f083
```

权限修复后沿用相同模型、数据和实验配置，以新的实验名提交并继续监控。

### 在已有权限内恢复

进一步查询当前账号的实际权限后确认：它能读取 `msrcommonacr` 镜像并向 `eaiacr`
推送镜像，但不能修改原仓库角色。因此不执行上述角色授权命令，改为将原镜像的同一
digest 连同 OCI referrer 附件（包含安全扫描记录）复制到工作区仓库：

`eaiacr.azurecr.io/internw0-delta/probe-base:ml-common-py312-20261022-4481c5e4`

复制工具使用 ORAS 1.3.5，官方二进制包经过发布校验和验证。复制保持镜像和扫描记录
的原始内容；凭证只通过标准输入交给客户端，临时认证文件在操作结束后删除。
恢复配置 `amlt_cogact_a100_probe_eaiacr.yaml` 从原模板生成的配置复制，仅替换镜像地址。
实际重提仍通过 amlt，保留完整的默认安全检查。复制成功和重提成功应分别核实，
不能把镜像传输进度当作训练进度。

镜像复制及全部附件复制均已成功（退出码 0），目标仓库再次读取到的 digest 为
`sha256:4481c5e4dc45e9a0ea3ce4f4b93ae4edb65697c936a6e9d5ca560df06e465569`，
与原始提交一致。恢复实验名为 `cogact-a1008-probe-20261022-eaiacr`。
本地复制凭据已随临时目录删除；未修改任何云端角色授权。

用户提供的 `/home/yuming/projects/openpi` checkout 中，
`workspaces/agent/jobs/openpi-cogact-train-b200x8.yaml` 使用
`pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel`，调度器为 Volcano，命名空间为 `bonete05`。
该基础环境可用于准备 InternW0 依赖，但本实验仍需安装项目锁定的 PyTorch 2.10。
当前安装的 amlt 扫描器对公开非 ACR 仓库仅支持 `mcr.microsoft.com`；尝试检查完整
Docker Hub 镜像地址时错误调用了 ACR 元数据接口并返回 404。这是扫描工具兼容问题，
不是发现了该镜像的漏洞。直接切换至这个 Docker Hub 镜像尚未得到可用性验证。

恢复提交于 2026-10-22 08:21 UTC 成功返回（退出码 0），默认镜像安全检查通过。
作业 ID 为 `epic_quince_bjc5wkd8f2`，
[恢复作业页面](https://aka.ms/amlt?q=lbbrs)。首次监控状态为 `queued`；
这表示已进入调度队列，尚不能据此确认训练入口或模型更新成功。

### 同一节点上的检查点 I/O 修复

恢复作业成功分配到 8×A100-SXM4-40GB。每卡 batch=1、2、3 均完成 3 次参数更新，
batch=4 在第 3 次更新时 CUDA OOM，当前配置的短程边界为每卡 3、全局 24。

首轮拟合完成 100 步，固定评估总 loss 从 1.327727 降至 0.478962，动作 loss 从
1.124757 降至 0.422030。随后直接写入挂载存储的一个优化器分片迟迟未完成，
其他进程等待时触发 300 秒 NCCL 同步超时；2026-10-22 09:01 UTC 工作进程退出。
原始失败结果保留在作业输出的 `probe/` 中，不能据此声称已完成 200 步或恢复验证。

新增本地写盘、进程退出后顺序上传的脚本，避免在分布式同步期间等待远端大文件上传。
原训练进程全部退出后，在同一作业节点的排错保留期内启动恢复，没有重新申请 GPU。
模型、输入、优化器、工作进程代码和容量测试签名不变；容量结果复用，拟合阶段从
预训练权重重跑。未完成检查点不作为恢复来源。

本次节点内恢复的结果子目录为 `probe-localio-20261022`，恢复日志为节点上的
`/tmp/cogact_localio_recovery.log`。上传成功后，作业输出根目录的 `recovery_exit.json`
记录这次恢复的退出码；原始 `debug_hold.json` 保留首次训练失败的退出码 1。
原前台包装进程已进入 20 小时保留期，其平台状态不能代替恢复结果。

本地恢复轮最终完成 200 步，两个检查点均在本地保存成功；8 个新进程均从第 100 步恢复。
最终固定评估总 loss 为 0.425981、动作 loss 为 0.383707。
顺序复制到挂载存储的大文件仍出现长时间等待，因此停止该复制进程后改用
`tools/cogact_publish.py` 调用节点已有的 AzCopy 直传。本次归档包含 37 个检查点相关文件，
总计 175,959,462,362 字节；完成标记最后上传，逐文件以远端 HEAD 核对大小。
使用当前账号生成的逐 blob、限时用户委托 SAS，仅授权本次目标文件的读取、创建和写入。
凭证通过 SSH 标准输入传入，不写凭证文件或输出到日志，未修改云端角色或工作区身份。
恢复汇总以 `probe-localio-20261022/result.json` 和根目录的 `recovery_exit.json` 为准。

结果表、曲线及 CSV 见 [测试报告](cogact_a100_results_20261022.md)。
