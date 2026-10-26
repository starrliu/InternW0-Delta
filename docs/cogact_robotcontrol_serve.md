# InternW0 CogACT 关节模型接入 RobotControl-G1

`scripts/serve_robotcontrol.sh` 启动一个 HTTP 模型服务，接收三路图像、任务指令及当前关节/夹爪状态，返回 **32 步双臂绝对关节目标**。机器人读取、动作规划和执行仍由 RobotControl-G1 负责。RTC 关闭。

对接依据为 [RobotControl-G1](https://github.com/msraeai/RobotControl-G1/tree/b37c68391ec4a9ffcf6e6d08ceeb16c1f9f3ff20) 的 `g1_platform/policy/cogact.py`、`perception/state_builder.py` 和 `action/interpreter.py`。原有 CogACT HTTP 客户端无需修改，只需切换输入/输出格式。

## 启动模型服务

使用已安装本仓库模型依赖的环境，额外安装 HTTP 依赖。B200 环境直接用现有虚拟环境解释器，避免重新解析依赖而替换 CUDA PyTorch：

```bash
uv pip install --python .venv/bin/python 'flask>=3.1,<4' 'waitress>=3,<4' 'requests>=2.32,<3'
```

从仓库根目录执行（以下是当前训练文件的路径）：

```bash
export CUDA_VISIBLE_DEVICES=0
bash scripts/serve_robotcontrol.sh \
  --run-dir /data/yuming/internw0-delta/runs/cogact/b200-joint-bs256-100k-20261023 \
  --checkpoint latest \
  --stats /data/yuming/internw0-delta/datasets/corrected12-all-20261023/normalization.json \
  --model-root /data/yuming/internw0-delta/checkpoints \
  --vlm-path /data/yuming/internw0-delta/checkpoints/RynnBrain1.1-2B \
  --port 5509
```

服务使用一张 GPU，默认 BF16、SDPA attention、10 次去噪。先在同一命令末尾加 `--check-config` 可仅用 CPU 核对配置、权重 step 和归一化；不会加载 GPU 模型。`WAM_PYTHON` 可指定其他虚拟环境解释器。

`latest` 根据训练的 `weights_manifest.json` 选择已完成的权重，启动时固定加载一次，不自动热更新。也可用 `--checkpoint /path/to/completed_policy.pt` 指定独立副本。训练权重 `slot_N.pt` 会循环覆盖；需要固定版本时应保留独立文件。服务读取的是策略 `.pt`，不是 DeepSpeed 优化器状态目录。

模型加载时严格检查全部可训练模块；冻结的 Wan VAE、T5 和 RynnBrain 从 `--model-root` / `--vlm-path` 加载。归一化必须来自该 checkpoint 对应的数据准备目录；若原目录仍可访问，还会核对其完成标记与 SHA256。迁移到其他机器时一起复制 `config.yaml`、权重和匹配的 `normalization.json`。

此脚本只启动服务进程，没有停止训练或操作 Pod 的命令。当前两 Pod 正在训练，服务尚未在它们的 GPU 上启动。

## RobotControl 端配置

将本仓库 [policy_config.yaml](../deploy/robotcontrol/policy_config.yaml) 的内容用于 RobotControl 的 `g1_platform/config/policy_config.yaml`：

```yaml
policy_type: cogact
action_format: joint_angles
input_format: joint_angles
```

在 RobotControl 的 `g1_platform/config/run_config.yaml` 中设置服务机器的可达 IP 与端口：

```yaml
policy_ip: "<模型服务机器的 IP>"
policy_port: 5509
```

[task_params.yaml](../deploy/robotcontrol/task_params.yaml) 提供对应训练节奏的默认值：32 步、间隔 1、每步 `1/30` 秒，左右夹爪缩放均为 `1.0`。RobotControl 当前仓库默认缩放为 `0.7`，且 task/subtask 有覆盖项；使用本模型时需一并调整这些覆盖项，否则会再次放大夹爪命令。`process_length` 也可设为 16，只执行前半个 chunk 再同步重规划，这不属于 RTC。TOPP-RA 等轨迹规划可能进一步改变实际执行耗时。

现有客户端先把图像等比缩到高 240 再 JPEG 编码，服务接受这种请求。服务端再按训练接口分别缩放三视角到 `192×256`，并生成 `384×256` 的 Wan 画布；VLM 使用同一组已缩放视图。原始训练任务 3173 的头部图像有上下各 5% 边框，可用 `--crop-head-3173` 对应裁剪；其他任务默认不裁剪。控制端额外的任务裁剪也需与相应训练视野一致。

## HTTP 输入和输出

`POST /api/inference` 使用 `multipart/form-data`，含 `image_0`（头部）、`image_1`（左腕）、`image_2`（右腕）三个 JPEG/PNG 文件，以及名为 `json` 的 JSON 文件或表单字段：

```json
{
  "task_description": "Pick up the cup.",
  "has_left": true,
  "has_right": true,
  "state": {
    "ROBOT_LEFT_JOINTS": [0, 0, 0, 0, 0, 0, 0],
    "ROBOT_LEFT_GRIPPER": [0.25],
    "ROBOT_RIGHT_JOINTS": [0, 0, 0, 0, 0, 0, 0],
    "ROBOT_RIGHT_GRIPPER": [0.75]
  }
}
```

这些零关节值仅说明字段形状，实际请求使用 RobotControl `StateBuilder` 读取的实时状态。关节顺序为每侧 `j1…j7`，单位弧度。RobotControl 已将实测夹爪除以 120，输入范围为 `[0,1]`；服务不会再除一次。外参字段允许附带，但模型不使用它。

返回 JSON 只含以下四个动作数组，第一维均为 32：

| 字段 | 形状 | 语义 |
|---|---|---|
| `ROBOT_LEFT_JOINTS` | `[32,7]` | 左臂绝对关节角，弧度 |
| `ROBOT_RIGHT_JOINTS` | `[32,7]` | 右臂绝对关节角，弧度 |
| `ROBOT_LEFT_GRIPPER` | `[32,1]` | 左夹爪目标，截断至 `[0,1]` |
| `ROBOT_RIGHT_GRIPPER` | `[32,1]` | 右夹爪目标，截断至 `[0,1]` |

内部状态顺序为 `[左关节7,左夹爪,右关节7,右夹爪]`，使用训练的 z-score（减均值、除标准差，截断至 `[-5,5]`），再放入 80 维的 `0–6、16、40–46、56` 槽。预测后按 action 统计反归一化，不与当前关节角相加。`has_left/has_right=false` 的一侧返回重复的当前状态，但输入仍需完整双臂状态和三视角。

服务不返回末端位姿或虚构任务完成率。RobotControl 的 `joint_angles` 解释分支直接读取这些关节数组，无需逆运动学。非法字段返回 HTTP 400；并发推理/重置返回 409；模型失败返回 500 并清空历史。

## 历史观测与新回合

每个服务进程对应一台机器人。缓存首个观测作为 anchor（本回合首帧），并寻找当前时刻前 32 个控制周期的 recent（历史帧）。这两项是模型的视觉条件，与 RTC 动作前缀不同。

现有客户端不发送帧号，因此默认用请求到达时间按 30 Hz 估算；仅使用目标时刻之前、相差不超过 4 帧的真实历史观测，找不到则屏蔽 recent。网络/推理等待会使这种时间对齐近似，不能声称与训练的逐帧历史完全一致。

需要精确历史时，可在 JSON 增加从 0 开始、严格递增的 `frame_index`，表示真实相机采样在 30 Hz 时间轴上的索引。不要用请求次数乘 chunk 长度代替。可用相同 multipart 请求调用 `POST /api/observe` 补充中间观测，仅更新历史而不推理；请求需串行发送，遇到 409 时不要堆积过时观测。显式帧号模式只使用准确的 `t−32`，缺失就屏蔽；一个回合内不能混用显式帧号和默认时间估计。

开始新回合（包括同一句指令重试）时调用：

```bash
curl -X POST http://<服务 IP>:5509/api/reset
```

或在下一次请求 JSON 中加 `"reset": true`，显式模式同时传 `"frame_index": 0`。这只重置模型历史，不移动机器人。指令改变或超过 60 秒没有请求也会自动清空历史。RobotControl 原有 `CogACTPolicy.reset()` 是空操作，不会自动调用此接口。

`GET /health` 返回已加载的 checkpoint step 和数据签名。每次推理响应的 `X-WAM-Frame-Index`、`X-WAM-Recent-Frame`、`X-WAM-Inference-Seconds` header 分别报告当前帧、历史帧（或 `masked`）、请求处理耗时。

## 验证范围

```bash
ROBOTCONTROL_G1_ROOT=/path/to/RobotControl-G1 \
  .venv/bin/python -m pytest -q tests/test_robotcontrol_server.py
```

测试包含训练状态/图像/归一化一致性、历史掩码与重置、缺字段/非法数值、单臂保持，以及使用**未修改的 RobotControl HTTP 客户端和动作解释器**通过本地 HTTP 服务收发 32 步动作。模型调用测试使用替身，以验证输入张量和关闭 RTC 的调用路径；完整 checkpoint 的 GPU 推理延迟及实机动作效果还未验证。

2026-10-26 已在独立代码目录以 `CUDA_VISIBLE_DEVICES=''` 执行 `--check-config`，通过当前第 80,000 步权重及全量归一化检查。数据签名为 `2320fa375ded473df404986bcb388b55d50f6e9bd9e5ce5567dce19599a95e6f`，未更改正在训练的代码目录。
