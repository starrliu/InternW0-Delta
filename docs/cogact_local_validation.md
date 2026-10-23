# CogACT 数据格式核对与 RTX 5080 本机验证

本页记录 2026-10-22 的单卡、batch size=1（每次处理一个样本）验证。
目标是检查真实数据能否接入完整 InternW0-Δ，并测量本机显存限制。
这不是全量数据适配或正式微调配方，未提交集群任务。

## 动作格式以 InternW0 为准

依据：

- [`projections/robot.yaml`](../configs/pretrain/projections/robot.yaml)：80 维布局；末端位姿状态使用位置和 6D 旋转，末端动作使用平移和旋转增量。
- [`embodiments/agibot.yaml`](../configs/pretrain/embodiments/agibot.yaml)：关节使用下一时刻的绝对目标，夹爪使用绝对命令。
- [`embodiments/robomind.yaml`](../configs/pretrain/embodiments/robomind.yaml)：支持只提供关节、夹爪而屏蔽末端位姿的机器人。
- `pretrain_lerobot_loader.py` 同时支持 `action_from_next_action` 和 `action_from_next_state`，即从下一行命令或实测状态构造目标。

因此，本次使用符合仓库约定的 joint-only 表示，而不是把所有通道都转换成增量：

| 通道 | 状态 | 动作 | 80 维位置（左闭右开） |
|---|---|---|---|
| 左臂 | 当前实测关节角 | 下一帧实测关节目标 | `[0,7)` |
| 左夹爪 | 实测值 / 120 | 当前帧夹爪命令 | `[16,17)` |
| 右臂 | 当前实测关节角 | 下一帧实测关节目标 | `[40,47)` |
| 右夹爪 | 实测值 / 120 | 当前帧夹爪命令 | `[56,57)` |

其他通道填零，维度掩码中 `True` 表示排除。没有输入相机外参。
关节保留数据源原生单位；本次没有推断单位，也没有进行度/弧度转换。
实机部署前仍须核对采集与控制接口的单位和频率。

原始 `action.npy` 是每行 56 个 float32 的裸二进制文件，使用
`meta_data.json` 中的长度和 `dim_list` 验证文件大小后，以 `np.memmap` 读取。
本次下载的是 `actions_cmd_rfix_v1` 中的修正版文件。

本次实际检查的完整录制 `15/20260506_162713` 有 5,832 行；其中命令关节
`2:16` 与实测关节 `38:52` 逐元素差值最大为 **0**。
因此，在该录制上，下一帧命令与下一帧实测目标完全相同。
该结论尚未推广到全部 12 份数据，未来全量接入时必须重新检查。

窗口沿用当前仓库默认值：32 步动作，RGB 帧为 `t,t+4,...,t+32`，
共 9 帧；另加片段起点和 `t-32` 的历史图像。三视角按照仓库的
`robotwin` 方式拼接为高 384、宽 256 的画布。
序列末尾重复最后一帧，但所有越界动作均排除损失；未沿用旧 OpenPi
额外 3 个重复动作仍参与损失的规则。

## 已实现的本机入口

- [`cogact_joint.py`](../src/wam/datasets/cogact_joint.py)：裸记录到关节/夹爪目标的映射、80 维投影、窗口边界处理。
- [`cogact_smoke.py`](../tools/cogact_smoke.py)：从一条真实示范构造输入、生成真实文本编码、尝试完整模型前向/反向/优化器更新，并记录失败阶段。
- [`cogact_5080.yaml`](../configs/smoke/cogact_5080.yaml)：本机测试配置，模型结构继承 `configs/model/wam.yaml`；开启视觉语言分支、bf16 和梯度检查点，不使用 CPU 卸载。
- [`test_cogact_joint.py`](../tests/test_cogact_joint.py)：6 项测试，覆盖绝对目标、关节/夹爪时间对齐、80 维槽位和跨片段边界，均通过。

测试样本来自：

```text
/data/cogact_dataset/cogact_pickplace_combined_20260730_161821_split_500_240h/
  episodic_dataset_20260730_235631.npy
  lmdb/frames.lmdb/
```

采用 metadata 第 0 段，区间 `[3142,3447)`，当前原始帧为 3206。
指令为 `pick up the blue-green spray bottle with the right arm`。

归一化仅使用该片段的统计，并用仓库的 z-score 归一化器缩放。
这是本次可运行性检查的局部统计，**不能用于正式训练或报告泛化性能**。

## 环境与复现

已在仓库 `.venv` 安装 Python 3.11.15、PyTorch 2.10.0+cu128，
CUDA wheel 包含 RTX 5080 所需的 `sm_120` 支持。

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python --torch-backend cu128 -e '.[cogact]' pytest
```

权重分别下载到以下目录：

```text
checkpoints/base/pretrain.pt
checkpoints/RynnBrain1.1-2B/                         # Alibaba-DAMO-Academy/RynnBrain1.1-2B
checkpoints/Wan-AI/Wan2.2-TI2V-5B/
  Wan2.2_VAE.pth
  models_t5_umt5-xxl-enc-bf16.pth
checkpoints/Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl/
```

本次访问 README 引用的转换版 Wan 编码器仓库返回 HTTP 401，
因此使用官方原始 `.pth` 权重，并配置 `redirect_common_files=false`。
现有加载器已支持这些文件，无需修改模型代码。

修正版动作 metadata 和该条录制的 `action.npy` 从已授权的 Blob 副本
`eaidatashare/v-xingyumliu/dataset/msra_process/<dataset_folder>/actions_cmd_rfix_v1/`
下载到 `.cache/cogact_smoke/source/actions_cmd_rfix_v1/`；图像直接读本地 LMDB。

在仓库根目录执行：

```bash
.venv/bin/python tools/cogact_smoke.py prepare \
  --dataset-dir /data/cogact_dataset/cogact_pickplace_combined_20260730_161821_split_500_240h \
  --metadata /data/cogact_dataset/cogact_pickplace_combined_20260730_161821_split_500_240h/episodic_dataset_20260730_235631.npy \
  --action-dir .cache/cogact_smoke/source/actions_cmd_rfix_v1

CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  .venv/bin/python tools/cogact_smoke.py encode-text

CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  .venv/bin/python tools/cogact_smoke.py run
```

`CUDA_VISIBLE_DEVICES=1` 选择空闲的第二张物理 GPU；进程内显示为 `cuda:0`。
`prepare` 输出真实样本、当前画布和来源报告；`encode-text` 复用仓库的
文本缓存工具。`run` 输出 `runs/cogact_5080_smoke/result.json`，失败返回非零状态。
只有前向、反向及优化器更新全部完成，才会记录 `status=passed`。

## 实测结论

**单张 16GB RTX 5080，完整模型、不使用 CPU 卸载时，无法完成 batch=1 验证。**

真实图像读取、数据窗口构造、6 项单元测试和文本编码均已完成。
实际模型初始化在把 RynnBrain 视觉语言模型移入 GPU 时发生显存不足
（CUDA OOM），约耗时 26.4 秒。未进入预训练策略权重加载、前向、反向或优化器更新。

PyTorch 报告可用设备总容量为 15.47 GiB，峰值已分配 14.91 GiB，
峰值保留 15.15 GiB。GiB 是以 2 的 30 次方字节为单位的容量。

根据下载权重中的张量形状计算，不包括运行时缓存和中间结果：

| 部分 | bf16 权重容量 |
|---|---:|
| 视频/动作专家及状态编码器 | 11.57 GiB |
| 冻结的视觉语言模型 | 4.12 GiB |
| 冻结的视频编码器 | 1.31 GiB |
| 合计 | **17.00 GiB** |

仅可训练部分的权重与同精度梯度就至少需要约 **23.14 GiB**，
还未包含优化器状态和计算中间结果。因此，减小 batch 无法解决此容量问题。
若继续在本机验证完整训练，需要另行验证参数/梯度/优化器的 CPU 卸载或分片方案；
双卡普通数据并行会各自复制模型，也不会消除这个加载限制。
本轮未尝试这些方案，正式训练参数也尚未确定。
