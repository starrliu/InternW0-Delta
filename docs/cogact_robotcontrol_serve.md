# InternW0 CogACT 关节模型接入 RobotControl-G1

`scripts/serve_robotcontrol.sh` 启动一个 HTTP 模型服务，接收三路图像、任务指令及当前关节/夹爪状态，返回 **32 步双臂绝对关节目标**。机器人读取、动作规划和执行仍由 RobotControl-G1 负责。RTC 关闭。

对接依据为 [RobotControl-G1](https://github.com/msraeai/RobotControl-G1/tree/b37c68391ec4a9ffcf6e6d08ceeb16c1f9f3ff20) 的 `g1_platform/policy/cogact.py`、`perception/state_builder.py` 和 `action/interpreter.py`。原有 CogACT HTTP 客户端切换输入/输出格式后可收发动作；若要匹配训练的历史观测和回合边界，还需按下文接入帧号、历史观测和重置。

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

服务默认使用一张 GPU，采用 BF16、SDPA attention、10 次去噪。先在同一命令末尾加 `--check-config` 可仅用 CPU 核对配置、权重 step 和归一化；不会加载 GPU 模型。`WAM_PYTHON` 可指定其他虚拟环境解释器。

可选 `--text-device cpu` 将 T5 文本编码器放在 CPU，按指令缓存编码结果，减少 GPU 显存占用。`--encoder-device cuda:1` 可将冻结的 RynnBrain 放到另一张可见 GPU；`--device cuda:0` 指定策略模型所在 GPU。拆分设备时先在 CPU 加载权重，再移动完整模块，不使用量化。单张 5090 的显存容量和速度、跨 GPU 运行均尚未实测。

`latest` 根据训练的 `weights_manifest.json` 选择已完成的权重，启动时固定加载一次，不自动热更新。也可用 `--checkpoint /path/to/completed_policy.pt` 指定独立副本。训练权重 `slot_N.pt` 会循环覆盖；需要固定版本时应保留独立文件。服务读取的是策略 `.pt`，不是 DeepSpeed 优化器状态目录。

模型加载时严格检查全部可训练模块；冻结的 Wan VAE、T5 和 RynnBrain 从 `--model-root` / `--vlm-path` 加载。归一化必须来自该 checkpoint 对应的数据准备目录；若原目录仍可访问，还会核对其完成标记与 SHA256。迁移到其他机器时一起复制 `config.yaml`、权重和匹配的 `normalization.json`。

此脚本只启动服务进程，没有停止训练或操作 Pod 的命令。2026-10-26 按用户要求取消本地启动，仅完成接口测试和[训练/推理一致性审计](cogact_inference_parity_20261026.md)，未运行完整模型服务，也未占用训练 Pod 的 GPU。

## 在 5090 上准备环境与文件

代码在 `starrliu/InternW0-Delta` 的 `cogact_baseline` 分支。以下命令在一份新目录建立 Python 3.11 环境，选择支持 Blackwell 的 CUDA 12.8 PyTorch 构建；需要已安装兼容驱动和 `uv`。这是部署起点，尚未在 5090 上验证完整模型的显存及速度。

```bash
git clone --branch cogact_baseline git@github.com:starrliu/InternW0-Delta.git
cd InternW0-Delta
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python --torch-backend cu128 -e '.[serve]'
```

Git 不含训练权重和冻结编码器。将匹配的部署文件单独放入以下目录（文件名可自行调整）：

```text
deploy-artifacts/
  run/
    config.yaml                 # 该训练运行保存的配置
    normalization.json          # 对应全量训练集的统计
    policy-step80000.pt          # 独立的已完成策略权重
  checkpoints/
    Wan-AI/
      Wan2.2-TI2V-5B/
        Wan2.2_VAE.pth
        models_t5_umt5-xxl-enc-bf16.pth
      Wan2.1-T2V-1.3B/
        google/umt5-xxl/         # 完整 tokenizer 文件
    RynnBrain1.1-2B/             # 完整模型、配置和 processor 文件
```

这些文件已在 Azure Blob 中备齐。账户为 `eaidatashare`，容器为 `v-xingyumliu`，共同 URL 前缀为 `https://eaidatashare.blob.core.windows.net/v-xingyumliu/`：

| 内容 | Blob 路径（相对共同前缀） |
|---|---|
| 80k 策略、训练配置、全量归一化和下载清单 | `internw0-delta/serving/b200-joint-bs256-100k-20261023/step80000/` 下的 `policy.pt`、`config.yaml`、`normalization.json`、`assets_manifest.json` |
| Wan VAE | `internw0-delta/assets/cogact-probe-20261022-v1/checkpoints/Wan-AI/Wan2.2-TI2V-5B/Wan2.2_VAE.pth` |
| RynnBrain 全目录 | `internw0-delta/assets/cogact-probe-20261022-v1/checkpoints/RynnBrain1.1-2B/` |
| T5 权重 | `internw0-delta/assets/text-encoder-20261023-v1/checkpoints/Wan-AI/Wan2.2-TI2V-5B/models_t5_umt5-xxl-enc-bf16.pth` |
| T5 tokenizer 全目录 | `internw0-delta/assets/text-encoder-20261023-v1/checkpoints/Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl/` |

这是固定的 **第 80,000 步**部署版本，不会自动指向训练中的新检查点。共 16 个文件、31.07 GB（28.94 GiB）。准确文件列表及 SHA256 在 [assets_step80000.json](../deploy/robotcontrol/assets_step80000.json)。归一化必须使用上表 `serving/.../step80000/normalization.json`，早期 `assets/.../data/window_*/normalization.json` 是小样本容量测试用统计，不能用于本模型部署。

安装 AzCopy 并使用有该容器读取权限的 Azure 账号登录后，在仓库根目录运行：

```bash
azcopy login
python tools/cogact_download_serve_assets.py --output deploy-artifacts
```

若该机器已经通过 Azure CLI 的 `az login` 登录，也可以让 AzCopy 使用该登录：

```bash
AZCOPY_AUTO_LOGIN_TYPE=AZCLI python tools/cogact_download_serve_assets.py --output deploy-artifacts
```

下载工具无需模型依赖，使用 Python 标准库和外部 `azcopy`。它将 Blob 的 `policy.pt` 保存为上图的 `run/policy-step80000.pt`，自动创建编码器目录；每个文件校验大小和 SHA256 后才改为正式文件名。已有文件会校验并复用，大小或哈希不符时报错而不覆盖。`--list` 只显示计划；`--verify-only` 只检查本地文件。URL 不含访问凭证，读取需要账号本身具有权限。

本次已用独立安装的 AzCopy 10.32.8 实际下载配置和统计并通过 SHA256 校验。若 Snap 版 AzCopy 的 Azure CLI 自动登录超时，可用 `--azcopy /path/to/azcopy` 指定独立安装版本。

策略 `.pt` 已包含训练后的专家权重，服务不会用原始 DiT 初始化覆盖它，也不需要 DeepSpeed 优化器分片。当前已验证的 80k 独立副本及数据签名见审计报告。

先进行 CPU 配置检查，再删除命令末尾的 `--check-config` 启动模型。示例将 T5 放到 CPU，降低单卡显存占用，但会占用相应主机内存：

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/serve_robotcontrol.sh \
  --run-dir "$PWD/deploy-artifacts/run" \
  --checkpoint "$PWD/deploy-artifacts/run/policy-step80000.pt" \
  --stats "$PWD/deploy-artifacts/run/normalization.json" \
  --model-root "$PWD/deploy-artifacts/checkpoints" \
  --vlm-path "$PWD/deploy-artifacts/checkpoints/RynnBrain1.1-2B" \
  --device cuda:0 --text-device cpu --port 5509 \
  --check-config
```

`--check-config` 只核对训练配置、策略文件元数据及归一化，不能代替 GPU 模型加载或冻结编码器文件检查。实际服务启动后执行 `curl http://127.0.0.1:5509/health`，应看到加载步数及 `"supported_protocol_versions": [1, 2]`。

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

每个服务进程对应一台机器人。anchor 是本回合首帧，recent 是当前时刻前 32 个采样周期的历史帧。这两项是模型的视觉条件，与 RTC 动作前缀不同。服务支持下述显式历史 v2，以及旧客户端使用的服务端缓存 v1。

### 显式历史 v2：推荐用于新客户端

客户端在本地按真实采样时间维护图像缓存，推理期间继续采集；一次 `POST /api/inference` 同时发送当前、首帧和历史图像，不需要逐帧调用 `/api/observe`。客户端启动时检查 `/health` 的 `supported_protocol_versions` 包含 `2`，再启用此协议。

在上文 JSON 的任务、状态与双臂启用字段之外，添加以下字段：

```json
{
  "protocol_version": 2,
  "episode_id": "episode-20261026-001",
  "frame_index": 96,
  "sampling_hz": 30,
  "anchor_frame_index": 0,
  "recent_frame_index": 64
}
```

`episode_id` 为 1–128 个 ASCII 字母、数字或 `_.:-` 字符；每个新回合生成新的 ID。同一回合的 `frame_index` 必须严格递增，不能用请求次数推算。`sampling_hz` 必须是数字 30；`recent_frame_index` 必须为 `frame_index−32`，或缺失时的 `null`。可附带采样时间用于客户端诊断，服务不将其转换为帧号。

multipart 图像字段如下，三个视角的顺序始终为头部、左腕、右腕：

| 观测 | 文件字段 | 要求 |
|---|---|---|
| 当前 | `image_0`、`image_1`、`image_2` | 每次完整提供 |
| 首帧 | `anchor_image_0`、`anchor_image_1`、`anchor_image_2` | 每次完整提供，同一回合保持不变 |
| 历史 | `recent_image_0`、`recent_image_1`、`recent_image_2` | `recent_frame_index` 非空时完整提供；为空时全部省略 |

建议使用 PNG 并省去客户端额外缩图，服务对三组观测应用相同的训练预处理。JPEG 也被接受；同一回合的 anchor 应采用固定编码配置。同为第 0 帧的图像必须相同：`frame_index=0` 时当前图像等于 anchor，`recent_frame_index=0` 时历史图像等于 anchor；服务对预处理后的画布做一致性检查。单个请求上限为 32 MiB。

第 0、20 帧没有合法的 `t−32`，传 `null`；第 96 帧应传第 64 帧。若后者确实丢失，也可传 `null`，服务会屏蔽该视觉条件并记录告警，不把首帧冒充有效历史。首个 HTTP 请求可以从任意非负帧开始，因为请求内已提供第 0 帧。历史状态不在此协议中传输，模型使用当前关节与夹爪状态。

新 `episode_id` 会开始新的服务端回合，且客户端仍应在任务重试时串行调用 `/api/reset`、清空自己的图像及动作队列。相同 ID 内改变任务指令或首帧会返回 400，需新 ID 或显式 reset。客户端必须丢弃旧回合的迟到响应，不得回传旧回合请求。v2 不使用请求到达时间或 60 秒空闲超时来重建时间轴。

响应动作格式保持四个数组不变；额外的 `X-WAM-Protocol-Version: 2`、`X-WAM-Episode-Id` header 标识协议和回合。`/api/observe` 仅用于 v1，向该接口发送 v2 会返回 400。切换 v1/v2 前先 reset；历史文件或元数据未声明 v2 时也会返回 400，避免静默忽略。

### 服务端缓存 v1：兼容原客户端

省略 `protocol_version` 或明确设为 `1` 时，服务缓存首个观测，并从已接收观测寻找 recent。

现有客户端不发送帧号，因此默认用请求到达时间按 30 Hz 估算；仅使用目标时刻之前、相差不超过 4 帧的真实历史观测，找不到则屏蔽 recent。网络/推理等待会使这种时间对齐近似，不能声称与训练的逐帧历史完全一致。

需要精确历史时，可在 JSON 增加从 0 开始、严格递增的 `frame_index`，表示真实相机采样在 30 Hz 时间轴上的索引。不要用请求次数乘 chunk 长度代替。可用相同 multipart 请求调用 `POST /api/observe` 补充中间观测，仅更新历史而不推理；请求需串行发送，遇到 409 时不要堆积过时观测。显式帧号模式只使用准确的 `t−32`，缺失就屏蔽；一个回合内不能混用显式帧号和默认时间估计。

开始新回合（包括同一句指令重试）时调用：

```bash
curl -X POST http://<服务 IP>:5509/api/reset
```

或在下一次请求 JSON 中加 `"reset": true`，显式模式同时传 `"frame_index": 0`。这只重置模型历史，不移动机器人。指令改变或超过 60 秒没有请求也会自动清空历史。RobotControl 原有 `CogACTPolicy.reset()` 是空操作，不会自动调用此接口。

两种协议下，`GET /health` 都返回已加载的 checkpoint step、数据签名和支持的协议版本。每次推理响应的 `X-WAM-Frame-Index`、`X-WAM-Recent-Frame`、`X-WAM-Inference-Seconds` header 分别报告当前帧、历史帧（或 `masked`）、请求处理耗时。

## 验证范围

```bash
ROBOTCONTROL_G1_ROOT=/path/to/RobotControl-G1 \
  OMP_NUM_THREADS=2 .venv/bin/python -m pytest -q \
  tests/test_robotcontrol_server.py tests/test_cogact_joint.py \
  tests/test_cogact_video.py tests/test_cogact_inference_parity.py
```

测试包含训练状态/图像/归一化一致性、历史掩码与重置、缺字段/非法数值、单臂保持，以及使用**未修改的 RobotControl HTTP 客户端和动作解释器**通过本地 HTTP 服务收发 32 步动作。新增独立审计测试比较真实训练和在线推理的 attention mask（允许哪些 token 相互读取的规则），以及视觉语言条件和当前状态 token。昂贵的模型计算使用替身；完整 checkpoint 的 GPU 数值结果、推理延迟及实机动作效果还未验证。

v2 增加了 PNG 三组图像传输、真实模型输入适配函数、首帧与准确历史帧选择、缺失历史掩码、回合重置、单调帧号、协议协商和错误请求检查。2026-10-26 上述命令共 **59 项通过**；v2 使用模拟客户端测试，尚未与修改后的 RobotControl 客户端或 5090 完成实机联调。

2026-10-26 已在独立代码目录以 `CUDA_VISIBLE_DEVICES=''` 执行 `--check-config`，通过当前第 80,000 步权重及全量归一化检查。数据签名为 `2320fa375ded473df404986bcb388b55d50f6e9bd9e5ce5567dce19599a95e6f`，未更改正在训练的代码目录。
