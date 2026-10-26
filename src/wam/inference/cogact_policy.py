"""CogACT joint policy with the same 16/80-D normalization and images as training."""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
from torchvision.transforms import functional as TF

from wam.datasets.cogact_joint import JOINT_INDICES, scatter80
from wam.datasets.cogact_video import VIEWS, canvas
from wam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
from wam.datasets.lerobot.utils.normalizer import SingleFieldLinearNormalizer


class CogACTNormalizer:
    def __init__(self, path):
        stats = json.loads(Path(path).read_text())
        if stats.get("mode") != "z-score":
            raise ValueError("Expected CogACT z-score normalization.json")
        self.signature = stats["signature"]
        for name in ("state", "action"):
            values = {k: torch.tensor(stats[name][k], dtype=torch.float32) for k in ("mean", "std")}
            if any(v.shape != (16,) or not torch.isfinite(v).all() for v in values.values()):
                raise ValueError(f"Invalid 16-D {name} statistics")
            if (values["std"] < 0).any():
                raise ValueError(f"Negative {name} standard deviation")
            setattr(self, name, SingleFieldLinearNormalizer(values, "z-score"))

    def encode_state(self, state):
        return scatter80(self.state.forward(torch.as_tensor(state, dtype=torch.float32)))[0]

    def decode_action(self, action):
        values = action.detach().to(device="cpu", dtype=torch.float32)
        if values.shape == (1, 32, 80):
            values = values[0]
        if values.shape != (32, 80) or not torch.isfinite(values).all():
            raise RuntimeError("Model must return finite actions [1,32,80] or [32,80]")
        return self.action.backward(values[:, JOINT_INDICES]).numpy()


@dataclass
class Observation:
    image: torch.Tensor  # [3,1,384,256], float32 [-1,1], CPU
    vlm_images: torch.Tensor  # [3,3,192,256], uint8 RGB
    proprio: torch.Tensor  # [80], normalized and masked


def prepare_images(images, *, crop_head_3173=False):
    """Resize each RGB view before composing the Wan canvas, as in the dataset."""
    views = []
    if len(images) != 3:
        raise ValueError("Expected head, left wrist and right wrist images")
    for index, image in enumerate(images):
        image = np.asarray(image)
        if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
            raise ValueError("Expected RGB uint8 HWC images")
        if index == 0 and crop_head_3173:
            # Task 3173 has 5% top/bottom borders (40:760 at H=800).
            border = round(image.shape[0] * .05)
            if border:
                image = image[border:-border]
        tensor = torch.from_numpy(image.copy()).permute(2, 0, 1).float() / 255
        views.append(TF.resize(tensor, [192, 256], antialias=True))
    views = torch.stack(views)
    return canvas(views[:, None]), (views * 255).round().clamp(0, 255).to(torch.uint8)


def prepare_observation(images, state, normalizer, *, crop_head_3173=False):
    image, vlm_images = prepare_images(images, crop_head_3173=crop_head_3173)
    return Observation(image, vlm_images, normalizer.encode_state(state))


class ObservationHistory:
    """Bounded real observations; never invent a frame at t-32."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.anchor = None
        self.frames = OrderedDict()
        self.last_step = -1

    def add(self, step, observation):
        if type(step) is not int or step < 0 or step <= self.last_step:
            raise ValueError("frame_index must strictly increase; reset at a new episode")
        if self.anchor is None:
            if step != 0:
                raise ValueError("The first observation after reset must have frame_index=0")
            self.anchor = observation
        self.frames[step] = observation
        self.last_step = step
        while next(iter(self.frames)) < step - 64:
            self.frames.popitem(last=False)

    def inputs(self, step, *, tolerance=0):
        target = step - 32
        candidates = [k for k in self.frames if target - tolerance <= k <= target]
        recent_step = max(candidates) if candidates else None
        recent = self.frames[recent_step] if recent_step is not None else self.anchor
        result = {}
        for name, obs, padded in (("anchor", self.anchor, False), ("recent", recent, recent_step is None)):
            key = "memory_video_" + name
            result[key] = obs.image
            result[key + "_is_pad"] = torch.tensor([padded])
            result[key + "_proprio"] = obs.proprio[None]
            result[key + "_proprio_is_pad"] = torch.tensor([padded])
        return result, recent_step


def read_run(args):
    """Validate a saved CogACT run without instantiating the GPU model."""
    from omegaconf import OmegaConf
    from wam.utils.config_resolvers import register_default_resolvers

    if args.model_root:
        os.environ["DIFFSYNTH_MODEL_BASE_PATH"] = str(args.model_root.resolve())
        os.environ["WAM_CHECKPOINT_ROOT"] = str(args.model_root.resolve())
    if args.vlm_path:
        os.environ["WAM_VLM_PATH"] = str(args.vlm_path.resolve())
    register_default_resolvers()
    cfg = OmegaConf.load(args.run_dir / "config.yaml")
    train = cfg.data.train
    if train.get("_target_") != "wam.datasets.cogact_dataset.CogACTDataset":
        raise ValueError("This server requires a CogACTDataset training run")
    if (int(train.num_frames), int(train.action_video_freq_ratio), tuple(train.video_size),
        int(train.memory_video_anchor_size), int(train.memory_recent_frame_offset), int(train.context_len),
        int(train.processor.action_output_dim), int(train.processor.proprio_output_dim)) != (33, 4, (384, 256), 1, 32, 128, 80, 80):
        raise ValueError("Unsupported training data contract")
    stats = args.stats or Path(str(train.pretrained_norm_stats))
    normalizer = CogACTNormalizer(stats)
    marker = Path(str(train.prepared_dir)) / "complete.json"
    if marker.is_file():
        receipt = json.loads(marker.read_text())
        if (receipt["signature"] != normalizer.signature or
            receipt["files"]["normalization.json"] != hashlib.sha256(stats.read_bytes()).hexdigest()):
            raise ValueError("Normalization differs from the prepared training dataset")

    folder = args.run_dir / "checkpoints/weights"
    expected_step = None
    if args.checkpoint == "latest":
        manifest = json.loads((folder / "weights_manifest.json").read_text())
        entry = manifest["latest"]
        path, expected_step = folder / entry["path"], int(entry["step"])
        if not path.resolve().is_relative_to(folder.resolve()):
            raise ValueError("Checkpoint manifest path escapes checkpoints/weights")
    else:
        path = Path(args.checkpoint).expanduser()
        if not path.is_file():
            path = folder / args.checkpoint
    if not path.is_file() or path.suffix != ".pt" or ".partial" in path.name:
        raise ValueError("Select a completed policy .pt, not a DeepSpeed state directory")
    payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    step = payload.get("step")
    if type(step) is not int or step < 0 or (expected_step is not None and step != expected_step):
        raise ValueError("Checkpoint step differs from the completion manifest")
    if not isinstance(payload.get("mot"), dict):
        raise ValueError("Checkpoint has no policy weights")
    del payload
    cfg.model.load_text_encoder = True
    cfg.model.mot_attention_backend = "sdpa"
    cfg.model.mot_checkpoint_mixed_attn = False
    cfg.model.video_dit_config.use_gradient_checkpointing = False
    cfg.model.action_dit_config.use_gradient_checkpointing = False
    cfg.model.action_dit_pretrained_path = None
    cfg.model.skip_dit_load_from_pretrain = True
    if args.vlm_path:
        cfg.model.understanding.vlm_model_path = str(args.vlm_path.resolve())
    report = dict(checkpoint=str(path.resolve()), step=step, stats=str(stats.resolve()),
                  normalization_signature=normalizer.signature, action_horizon=32, action_hz=30,
                  input_format="joint_angles", action_format="joint_angles", rtc=False,
                  gripper_range=[0, 1], cameras=list(VIEWS))
    return cfg, normalizer, path, report


class CogACTPolicy:
    def __init__(self, model, normalizer, *, denoise_steps=10, seed=42, tiled=False, text_device=None):
        self.model, self.normalizer = model, normalizer
        self.denoise_steps, self.seed, self.tiled = denoise_steps, seed, tiled
        self.text_device = None if text_device is None else torch.device(text_device)
        self.action_dim_is_pad = scatter80(torch.zeros(16))[1]
        self.text = {}

    @torch.inference_mode()
    def predict(self, observation, instruction, memory):
        from wam.inference.online_action_policy import infer_online_action_chunk
        prompt = DEFAULT_PROMPT.format(task=instruction)
        if prompt not in self.text:
            if self.text_device is None:
                encoded = self.model.encode_prompt(prompt)
            else:
                ids, mask = self.model.tokenizer(prompt, return_mask=True, add_special_tokens=True)
                ids = ids.to(self.text_device)
                mask = mask.to(device=self.text_device, dtype=torch.bool)
                context = self.model.text_encoder(ids, mask).masked_fill(~mask.unsqueeze(-1), 0)
                encoded = context, torch.ones_like(mask)
            self.text = {prompt: encoded}
        context, mask = self.text[prompt]
        action = infer_online_action_chunk(
            self.model, prompt=None, context=context, context_mask=mask,
            understanding_prompt=prompt, input_image=observation.image[:, 0][None],
            proprio=observation.proprio[None], action_dim_is_pad=self.action_dim_is_pad,
            action_horizon=32, memory_inputs=memory, vlm_current_images=observation.vlm_images,
            vlm_view_names="|".join(VIEWS), video_layout="single", video_view_names="canvas",
            num_inference_steps=self.denoise_steps, seed=self.seed, tiled=self.tiled,
        )["action"]
        return self.normalizer.decode_action(action)


def load_policy(args, prepared):
    from hydra.utils import instantiate
    from wam.training.checkpoint import load_wam_checkpoint_strict
    cfg, normalizer, path, _ = prepared
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "no": torch.float32}[args.precision]
    encoder_device = getattr(args, "encoder_device", None)
    text_device = getattr(args, "text_device", None)
    split = encoder_device is not None or text_device is not None
    # Loading on CPU first avoids briefly placing all frozen backbones on a
    # 16-GiB policy GPU before they can be distributed across devices.
    model = instantiate(cfg.model, model_dtype=dtype, device="cpu" if split else args.device).eval()
    load_wam_checkpoint_strict(model, path)
    if split:
        place_model(model, args.device, encoder_device or args.device, text_device or args.device)
    else:
        model.to(args.device)
    model.requires_grad_(False)
    return CogACTPolicy(model, normalizer, denoise_steps=args.denoise_steps, seed=args.seed,
                        tiled=args.tiled, text_device=text_device)


def place_model(model, policy_device, encoder_device, text_device):
    """Place complete BF16 modules; no quantization or weights are changed."""
    model.device = torch.device(policy_device)
    for name, module in model.named_children():
        device = text_device if name == "text_encoder" else encoder_device if name == "understanding" else policy_device
        module.to(device=device, dtype=model.torch_dtype)
    # These are plain attributes rather than registered buffers.
    for name in ("mean", "std"):
        if torch.is_tensor(getattr(model.vae, name, None)):
            setattr(model.vae, name, getattr(model.vae, name).to(device=policy_device, dtype=model.torch_dtype))
    if hasattr(model.vae, "mean") and hasattr(model.vae, "std"):
        model.vae.scale = [model.vae.mean, 1.0 / model.vae.std]
    if model.understanding is not None:
        model.understanding.vlm.device = torch.device(encoder_device)
        model.understanding.vlm.dtype = model.torch_dtype
