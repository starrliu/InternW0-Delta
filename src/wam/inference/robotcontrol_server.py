"""HTTP multipart interface consumed by RobotControl-G1's CogACTPolicy."""
from __future__ import annotations

import argparse
import io
import json
import logging
import os
from pathlib import Path
import threading
import time

import numpy as np
from PIL import Image, UnidentifiedImageError

from .cogact_policy import ObservationHistory, prepare_observation

LOG = logging.getLogger(__name__)


def parse_payload(payload):
    if not isinstance(payload, dict):
        raise ValueError("json must contain an object")
    instruction = payload.get("task_description")
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("task_description must be a non-empty string")
    state = payload.get("state")
    if not isinstance(state, dict):
        raise ValueError("state is required; set RobotControl input_format: joint_angles")
    values = []
    for side in ("LEFT", "RIGHT"):
        for field, size in (("JOINTS", 7), ("GRIPPER", 1)):
            key = f"ROBOT_{side}_{field}"
            try:
                array = np.asarray(state[key], dtype=np.float32)
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"state.{key} must contain {size} numbers") from exc
            if array.shape != (size,) or not np.isfinite(array).all():
                raise ValueError(f"state.{key} must be finite with shape [{size}]")
            if field == "GRIPPER" and ((array < 0).any() or (array > 1).any()):
                raise ValueError(f"state.{key} must be in [0,1]; RobotControl already divides by 120")
            values.extend(array)
    for key in ("has_left", "has_right", "reset"):
        if key in payload and type(payload[key]) is not bool:
            raise ValueError(f"{key} must be boolean")
    if not payload.get("has_left", True) and not payload.get("has_right", True):
        raise ValueError("At least one arm must be enabled")
    if "frame_index" in payload and (type(payload["frame_index"]) is not int or payload["frame_index"] < 0):
        raise ValueError("frame_index must be a non-negative integer at 30 Hz")
    return instruction.strip(), np.asarray(values, dtype=np.float32)


def action_response(action, state, payload):
    action = np.asarray(action, dtype=np.float32).copy()
    if action.shape != (32, 16) or not np.isfinite(action).all():
        raise RuntimeError("Policy returned invalid robot actions; expected finite [32,16]")
    result = {}
    for side, start in (("LEFT", 0), ("RIGHT", 8)):
        if not payload.get("has_" + side.lower(), True):
            action[:, start:start + 8] = state[start:start + 8]
        result[f"ROBOT_{side}_JOINTS"] = action[:, start:start + 7].tolist()
        result[f"ROBOT_{side}_GRIPPER"] = action[:, start + 7:start + 8].clip(0, 1).tolist()
    # The RobotControl interpreter applies len() to EVERY returned value.
    # Metadata belongs in HTTP headers or /health, never in this action dict.
    return result


class PolicyService:
    """One robot/episode per server; callers serialize observe/infer/reset."""
    def __init__(self, policy, *, report=None, crop_head_3173=False, idle_reset_seconds=60):
        self.policy = policy
        self.report = report or {}
        self.crop_head_3173 = crop_head_3173
        self.idle_reset_seconds = idle_reset_seconds
        self.lock = threading.Lock()
        self.history = ObservationHistory()
        self.reset()

    def reset(self):
        self.history.reset()
        self.instruction = None
        self.started_at = self.last_arrival = None
        self.explicit_clock = None

    def process(self, images, payload, *, observe_only=False, now=None):
        instruction, state = parse_payload(payload)
        observation = prepare_observation(images, state, self.policy.normalizer,
                                          crop_head_3173=self.crop_head_3173)
        now = time.monotonic() if now is None else now
        if (payload.get("reset", False) or instruction != self.instruction or
            (self.last_arrival is not None and now - self.last_arrival > self.idle_reset_seconds)):
            self.reset()
        explicit = "frame_index" in payload
        if self.explicit_clock is not None and self.explicit_clock != explicit:
            raise ValueError("Do not mix frame_index and arrival-time clocks without reset")
        start = now if self.started_at is None else self.started_at
        step = payload["frame_index"] if explicit else round((now - start) * 30)
        self.history.add(step, observation)
        self.instruction, self.started_at, self.last_arrival = instruction, start, now
        self.explicit_clock = explicit
        # With the stock client, request arrival time only approximates capture time.
        # Use an actually observed past frame within 4 ticks, otherwise mask it.
        memory, recent_step = self.history.inputs(step, tolerance=0 if explicit else 4)
        if observe_only:
            return {"frame_index": step}, step, recent_step
        try:
            action = self.policy.predict(observation, instruction, memory)
        except Exception as exc:
            # A model ValueError is a server failure, not malformed input.
            raise RuntimeError("Model prediction failed") from exc
        return action_response(action, state, payload), step, recent_step


def create_app(service):
    from flask import Flask, jsonify, request
    from werkzeug.exceptions import HTTPException
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024

    @app.errorhandler(HTTPException)
    def http_error(exc):
        return jsonify(error=exc.description), exc.code

    @app.get("/health")
    def health():
        return jsonify(status="ready", **service.report)

    @app.post("/api/reset")
    def reset():
        if not service.lock.acquire(blocking=False):
            return jsonify(error="Inference busy; retry reset after it completes"), 409
        try:
            service.reset()
            return jsonify(status="reset")
        finally:
            service.lock.release()

    def read_request():
        part = request.files.get("json")
        raw = part.read() if part is not None else request.form.get("json", "")
        try:
            payload = json.loads(raw)
        except (ValueError, UnicodeError) as exc:
            raise ValueError("Missing or invalid multipart json part") from exc
        parse_payload(payload)
        images = []
        for index in range(3):
            name = f"image_{index}"
            if name not in request.files:
                raise ValueError(f"Missing {name}; all three cameras are required")
            try:
                with Image.open(io.BytesIO(request.files[name].read())) as image:
                    if image.width * image.height > 16_000_000:
                        raise ValueError(f"{name} exceeds 16 megapixels")
                    images.append(np.asarray(image.convert("RGB")))
            except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
                raise ValueError(f"Invalid encoded image: {name}") from exc
        return images, payload

    @app.post("/api/inference")
    @app.post("/api/observe")
    def inference():
        if not service.lock.acquire(blocking=False):
            return jsonify(error="Inference busy; serialize requests for this robot"), 409
        started = time.perf_counter()
        try:
            images, payload = read_request()
            result, step, recent = service.process(images, payload, observe_only=request.path == "/api/observe")
            response = jsonify(result)
            response.headers["X-WAM-Frame-Index"] = str(step)
            response.headers["X-WAM-Recent-Frame"] = "masked" if recent is None else str(recent)
            response.headers["X-WAM-Inference-Seconds"] = f"{time.perf_counter() - started:.4f}"
            return response
        except ValueError as exc:
            return jsonify(error=str(exc)), 400
        except Exception:
            service.reset()
            LOG.exception("Policy inference failed")
            return jsonify(error="Policy inference failed; history reset, see server log"), 500
        finally:
            service.lock.release()
    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", default="latest", help="Completed .pt path or latest manifest entry")
    parser.add_argument("--stats", type=Path, help="Matching CogACT normalization.json")
    parser.add_argument("--model-root", type=Path, default=os.environ.get("DIFFSYNTH_MODEL_BASE_PATH"))
    parser.add_argument("--vlm-path", type=Path, default=os.environ.get("WAM_VLM_PATH"))
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5509)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--precision", choices=("bf16", "fp16", "no"), default="bf16")
    parser.add_argument("--denoise-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tiled", action="store_true", help="Tile VAE encoding to reduce memory")
    parser.add_argument("--crop-head-3173", action="store_true", help="Crop 5%% top/bottom head-camera borders")
    parser.add_argument("--check-config", action="store_true", help="Validate config/checkpoint/stats without loading GPU model")
    args = parser.parse_args()
    if args.denoise_steps < 1 or not 1 <= args.port <= 65535:
        parser.error("denoise-steps must be positive; port must be 1..65535")
    if args.device == "cuda":
        args.device = "cuda:0"
    from .cogact_policy import load_policy, read_run
    logging.basicConfig(level=logging.INFO)
    prepared = read_run(args)
    print(json.dumps(prepared[-1], indent=2), flush=True)
    if args.check_config:
        return
    policy = load_policy(args, prepared)
    service = PolicyService(policy, report=prepared[-1], crop_head_3173=args.crop_head_3173)
    from waitress import serve
    serve(create_app(service), host=args.host, port=args.port, threads=4,
          max_request_body_size=32 * 1024 * 1024)


if __name__ == "__main__":
    main()
