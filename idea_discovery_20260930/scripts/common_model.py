"""Strict released V-JEPA2 adapter. Model calls belong exclusively in sbatch.

Only supplied visible tubelets reach the context transformer. Each observed
image occupies a tubelet of two identical frames, at its explicit time index.
This is an exploratory sparse-observation protocol, not a reproduction of the
authors' 48-frame IntPhys2 protocol or a trained causal tracking head.
"""
import copy
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


class FrozenVJEPA:
    def __init__(self, source, checkpoint, device="cuda"):
        sys.path.insert(0, str(Path(source).resolve()))
        from src.hub.backbones import _make_vjepa2_model
        self.device = torch.device(device)
        self.encoder, self.predictor = _make_vjepa2_model(model_name="vit_giant", pretrained=False, num_frames=64)
        self.target_encoder = copy.deepcopy(self.encoder)
        state = torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=False)
        self.load_report = {}
        for name, module in (("encoder", self.encoder), ("target_encoder", self.target_encoder), ("predictor", self.predictor)):
            if name not in state:
                raise RuntimeError(f"Released checkpoint lacks {name}; random fallback forbidden")
            clean = {key.replace("module.", "").replace("backbone.", ""): value for key, value in state[name].items()}
            result = module.load_state_dict(clean, strict=True)
            self.load_report[name] = str(result)
            module.eval().requires_grad_(False).to(self.device)
        if self.predictor.mask_tokens is None or self.predictor.num_mask_tokens == 0:
            raise RuntimeError("Predictor must use mask tokens, never target feature conditioning")
        del state
        self.encoder_calls = 0
        self.predictor_calls = 0
        self.encoded_visible_tubelets = 0
        self.predicted_tubelets = 0
        self.gpu_seconds = 0.0

    @staticmethod
    def _image(value):
        image = np.asarray(value)
        if image.ndim != 3:
            raise ValueError(f"Expected RGB image, received {image.shape}")
        if image.shape[0] == 3 and image.shape[-1] != 3:
            image = np.moveaxis(image, 0, -1)
        if image.dtype != np.uint8:
            if float(image.max()) <= 1.0:
                image = image * 255
            image = np.clip(image, 0, 255).astype(np.uint8)
        image = np.asarray(Image.fromarray(image).resize((256, 256), Image.Resampling.BILINEAR)).copy()
        return image

    def _inputs(self, times, images, targets=()):
        if len(times) != len(images) or not times or len(set(times)) != len(times):
            raise ValueError("Require unique visible time indices and one RGB image/time")
        origin = min(times)
        visible = [int(t) - origin for t in times]
        future = [int(t) - origin for t in targets]
        if set(visible).intersection(future):
            raise ValueError("Prediction target must not be an observed tubelet")
        length = max(visible + future) + 1
        if length > 32 or min(visible + future) < 0:
            raise ValueError("Sparse window must fit the released 64-frame/32-tubelet support")
        video = torch.zeros((1, 3, length * 2, 256, 256), dtype=torch.float32, device=self.device)
        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device)[:, None, None]
        std = torch.tensor([0.229, 0.224, 0.225], device=self.device)[:, None, None]
        for t, image in zip(visible, images):
            tensor = torch.from_numpy(self._image(image)).to(self.device).permute(2, 0, 1).float() / 255
            tensor = (tensor - mean) / std
            video[0, :, 2*t:2*t+2] = tensor[:, None]
        mask_x = torch.tensor([256*t + p for t in visible for p in range(256)], device=self.device, dtype=torch.long)[None]
        mask_y = torch.tensor([256*t + p for t in future for p in range(256)], device=self.device, dtype=torch.long)[None]
        return video, mask_x, mask_y

    @torch.inference_mode()
    def encode_sparse(self, times, images):
        video, mask, _ = self._inputs(times, images)
        torch.cuda.synchronize(self.device)
        start = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            features = self.target_encoder(video, masks=[mask])
        torch.cuda.synchronize(self.device)
        self.gpu_seconds += time.perf_counter() - start
        self.encoder_calls += 1
        self.encoded_visible_tubelets += len(times)
        return features[0].float().reshape(len(times), 16, 16, -1)

    @torch.inference_mode()
    def predict_sparse(self, context_times, images, target_times):
        if min(target_times) <= max(context_times):
            raise ValueError("This causal probe predicts only after the last observed image")
        video, mask_x, mask_y = self._inputs(context_times, images, target_times)
        torch.cuda.synchronize(self.device)
        start = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            context = self.encoder(video, masks=[mask_x])
            prediction = self.predictor(context, [mask_x], [mask_y])
        torch.cuda.synchronize(self.device)
        self.gpu_seconds += time.perf_counter() - start
        self.encoder_calls += 1
        self.predictor_calls += 1
        self.encoded_visible_tubelets += len(context_times)
        self.predicted_tubelets += len(target_times)
        if isinstance(prediction, list):
            prediction = prediction[0]
        return prediction[0].float().reshape(len(target_times), 16, 16, -1)

    def counters(self):
        return {"encoder_calls": self.encoder_calls, "predictor_calls": self.predictor_calls,
                "encoded_visible_tubelets": self.encoded_visible_tubelets,
                "predicted_tubelets": self.predicted_tubelets, "gpu_seconds": self.gpu_seconds}

    @torch.inference_mode()
    def score_clip(self, frames, context_frames=12):
        """Native adjacent-frame tubelets; actual targets used only for scoring.

        Context masks remove future patches before transformer attention. The
        full-window EMA target matches the published surprise interface; this
        is retrospective physical-plausibility classification, not warning.
        """
        if len(frames) % 2 or context_frames % 2 or not 0 < context_frames < len(frames):
            raise ValueError("Require even context and full frame counts")
        images = np.stack([self._image(frame) for frame in frames])
        video = torch.from_numpy(images).to(self.device).permute(3, 0, 1, 2)[None].float() / 255
        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device)[None, :, None, None, None]
        std = torch.tensor([0.229, 0.224, 0.225], device=self.device)[None, :, None, None, None]
        video = (video - mean) / std
        cutoff = context_frames // 2 * 256
        count = len(frames) // 2 * 256
        mask_x = torch.arange(cutoff, device=self.device)[None]
        mask_y = torch.arange(cutoff, count, device=self.device)[None]
        torch.cuda.synchronize(self.device)
        start = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            target = self.target_encoder(video)
            context = self.encoder(video, masks=[mask_x])
            prediction = self.predictor(context, [mask_x], [mask_y])
        torch.cuda.synchronize(self.device)
        self.gpu_seconds += time.perf_counter() - start
        self.encoder_calls += 2
        self.predictor_calls += 1
        self.encoded_visible_tubelets += len(frames) // 2 + context_frames // 2
        self.predicted_tubelets += (len(frames) - context_frames) // 2
        target = F.layer_norm(target[:, cutoff:].float(), (target.shape[-1],))
        prediction = prediction.float()
        future = (len(frames) - context_frames) // 2
        return prediction[0].reshape(future, 256, -1), target[0].reshape(future, 256, -1)
