"""One-seed, label-free matched trajectory-target pilot (step-3 protocol).

This module deliberately does not import the old comp_pilot models or labels.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import platform
import random
import time
import traceback
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .signatures import chen, flatten, integrated_lift, signature


ARMS = ("ordered_frames", "learned_summary", "signature_d2")


def write_json(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    tmp.replace(path)


def sinusoid(position: torch.Tensor, dim: int) -> torch.Tensor:
    if dim % 2:
        raise ValueError("sinusoid dimension must be even")
    frequency = torch.exp(-math.log(10000.0) * torch.arange(dim // 2, device=position.device) / (dim // 2))
    angle = position.float()[..., None] * frequency
    return torch.cat([angle.sin(), angle.cos()], -1)


def transformer(dim: int, depth: int, heads: int = 4) -> nn.TransformerEncoder:
    layer = nn.TransformerEncoderLayer(dim, heads, 4 * dim, 0.0, batch_first=True,
                                       norm_first=True, activation="gelu")
    return nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)


class FrameEncoder(nn.Module):
    def __init__(self, latent: int) -> None:
        super().__init__()
        width = 64
        self.visual = nn.Sequential(nn.LayerNorm(384), nn.Linear(384, width))
        self.camera = nn.Parameter(torch.randn(1, 3, 1, width) * 0.02)
        self.patch = nn.Parameter(torch.randn(1, 1, 16, width) * 0.02)
        self.query = nn.Parameter(torch.randn(1, 1, width) * 0.02)
        self.attn = nn.MultiheadAttention(width, 4, batch_first=True)
        self.proprio = nn.Linear(16, width)
        self.out = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, latent), nn.LayerNorm(latent))

    def forward(self, visual: torch.Tensor, proprio: torch.Tensor) -> torch.Tensor:
        x = self.visual(visual.float()) + self.camera + self.patch
        x = x.flatten(1, 2)
        q = self.query.expand(len(x), -1, -1)
        pooled = self.attn(q, x, x, need_weights=False)[0][:, 0]
        return self.out(pooled + self.proprio(proprio.float()))


class FrameAE(nn.Module):
    def __init__(self, latent: int) -> None:
        super().__init__()
        self.encoder = FrameEncoder(latent)
        self.visual_decoder = nn.Sequential(nn.Linear(latent, 512), nn.GELU(), nn.Linear(512, 3 * 16 * 384))
        self.proprio_decoder = nn.Sequential(nn.Linear(latent, 64), nn.GELU(), nn.Linear(64, 16))

    def loss(self, visual: torch.Tensor, proprio: torch.Tensor) -> tuple[torch.Tensor, dict]:
        z = self.encoder(visual, proprio)
        visual_hat = self.visual_decoder(z).reshape(-1, 3, 16, 384)
        proprio_hat = self.proprio_decoder(z)
        lv = F.mse_loss(visual_hat.float(), F.layer_norm(visual.float(), (384,)))
        lp = F.mse_loss(proprio_hat.float(), proprio.float())
        return lv + lp, {"visual": float(lv.detach()), "proprio": float(lp.detach())}


class SummaryAE(nn.Module):
    def __init__(self, latent: int, width: int, tokens: int, samples: int) -> None:
        super().__init__()
        self.width, self.tokens, self.samples = width, tokens, samples
        self.inp = nn.Linear(latent, width)
        self.queries = nn.Parameter(torch.randn(1, tokens, width) * 0.02)
        self.encoder = transformer(width, 3)
        layer = nn.TransformerDecoderLayer(width, 4, 4 * width, 0.0, batch_first=True,
                                           norm_first=True, activation="gelu")
        self.decoder = nn.TransformerDecoder(layer, 2)
        self.out = nn.Linear(width, latent)
        self.norm = nn.LayerNorm(width)

    def encode(self, frames: torch.Tensor) -> torch.Tensor:
        pos = sinusoid(torch.arange(self.samples, device=frames.device), self.width)
        x = self.inp(frames) + pos[None]
        y = self.encoder(torch.cat([self.queries.expand(len(x), -1, -1), x], 1))
        return self.norm(y[:, :self.tokens])

    def decode(self, summary: torch.Tensor, length: int) -> torch.Tensor:
        q = sinusoid(torch.arange(self.samples, device=summary.device), self.width)[None].expand(len(summary), -1, -1)
        q = q + sinusoid(torch.full((len(summary), 1), float(length), device=summary.device), self.width)
        return self.out(self.decoder(q, summary))


class SignatureDecoder(nn.Module):
    def __init__(self, rep_dim: int, latent: int, samples: int) -> None:
        super().__init__()
        self.samples = samples
        self.net = nn.Sequential(nn.Linear(rep_dim + latent + 32, 512), nn.GELU(),
                                 nn.Linear(512, 512), nn.GELU(), nn.Linear(512, samples * latent))
        self.latent = latent

    def forward(self, normalized_signature: torch.Tensor, endpoint: torch.Tensor, length: int) -> torch.Tensor:
        duration = sinusoid(torch.full((len(endpoint),), float(length), device=endpoint.device), 32)
        return self.net(torch.cat([normalized_signature, endpoint, duration], -1)).reshape(-1, self.samples, self.latent)


class HistoryEncoder(nn.Module):
    def __init__(self, latent: int, dim: int) -> None:
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(latent + 12, dim), nn.GELU())
        self.gru = nn.GRU(dim, dim, batch_first=True)

    def forward(self, z: torch.Tensor, incoming_actions: torch.Tensor) -> torch.Tensor:
        return self.gru(self.inp(torch.cat([z, incoming_actions], -1)))[1][0]


class ActionPredictor(nn.Module):
    def __init__(self, latent: int, dim: int, rep_dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.h = nn.Linear(dim, dim)
        self.z = nn.Linear(latent, dim)
        self.a = nn.Linear(12, dim)
        self.query = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.body = transformer(dim, 4)
        self.norm = nn.LayerNorm(dim)
        self.rep = nn.Linear(dim, rep_dim)
        self.end = nn.Linear(dim, latent)

    def forward(self, h: torch.Tensor, z: torch.Tensor, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pos = sinusoid(torch.arange(actions.shape[1], device=actions.device), self.dim)
        act = self.a(actions) + pos[None]
        x = torch.cat([self.h(h)[:, None], self.z(z)[:, None], act,
                       self.query.expand(len(h), -1, -1)], 1)
        q = self.norm(self.body(x)[:, -1])
        return self.rep(q), self.end(q)


class TrajectoryArm(nn.Module):
    def __init__(self, kind: str, latent: int, dim: int, rep_dim: int) -> None:
        super().__init__()
        self.kind = kind
        self.history = HistoryEncoder(latent, dim)
        self.predictor = ActionPredictor(latent, dim, rep_dim)
        self.update = nn.Sequential(nn.Linear(dim + rep_dim + latent, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))

    def advance(self, h, rep, endpoint):
        return h + self.update(torch.cat([h, rep, endpoint], -1))


class PackedData:
    def __init__(self, root: Path, cfg: dict, profile: bool) -> None:
        manifest = json.loads((root / "manifest.json").read_text())
        if not math.isclose(manifest["encoder"]["frame_rate_hz"], 1 / cfg["dt_seconds"]):
            raise ValueError("feature cadence mismatch")
        entries = [m for m in manifest["episodes"] if m["split"] != "test"]
        if profile:
            rng = random.Random(cfg["seed"])
            selected = []
            for split, count in (("train", cfg["profile"]["max_train_episodes"]),
                                 ("val", cfg["profile"]["max_val_episodes"])):
                pool = [m for m in entries if m["split"] == split]
                selected += rng.sample(pool, count)
            entries = sorted(selected, key=lambda m: m["episode"])
        visual, proprio, action = [], [], []
        self.entries, self.offsets = [], {}
        offset = 0
        for i, m in enumerate(entries):
            p = torch.load(root / m["path"], map_location="cpu", weights_only=True, mmap=True)
            if not torch.equal(p["frame_index"], torch.arange(m["frames"])):
                raise ValueError("non-consecutive cached frames")
            if len(p["visual"]) != m["frames"] or len(p["proprio"]) != m["frames"] or len(p["actions"]) != m["frames"]:
                raise ValueError("unaligned cached episode")
            self.offsets[m["episode"]] = offset
            self.entries.append(m)
            offset += m["frames"]
            visual.append(p["visual"])
            proprio.append(p["proprio"].float())
            action.append(p["actions"].float())
            if (i + 1) % 50 == 0:
                print(f"loaded {i+1}/{len(entries)} cached episodes", flush=True)
        self.visual = torch.cat(visual)
        p_raw, a_raw = torch.cat(proprio), torch.cat(action)
        train_rows = torch.cat([torch.arange(self.offsets[m["episode"]], self.offsets[m["episode"]] + m["frames"])
                                for m in self.entries if m["split"] == "train"])
        self.train_rows = train_rows
        self.val_rows = torch.cat([torch.arange(self.offsets[m["episode"]], self.offsets[m["episode"]] + m["frames"])
                                   for m in self.entries if m["split"] == "val"])
        self.proprio_stats = (p_raw[train_rows].mean(0), p_raw[train_rows].std(0, unbiased=False).clamp_min(1e-4))
        self.action_stats = (a_raw[train_rows].mean(0), a_raw[train_rows].std(0, unbiased=False).clamp_min(1e-4))
        self.proprio = (p_raw - self.proprio_stats[0]) / self.proprio_stats[1]
        self.actions = (a_raw - self.action_stats[0]) / self.action_stats[1]
        self.pools = {}
        minimum = cfg["history_stride"] * (cfg["history_tokens"] - 1) + 1
        for split in ("train", "val"):
            for length in set(cfg["train_lengths"] + [sum(x) for x in cfg["eval_splits"].values()]):
                starts = []
                for m in self.entries:
                    if m["split"] != split:
                        continue
                    off = self.offsets[m["episode"]]
                    starts.extend(off + t for t in range(minimum, m["frames"] - length, 4))
                self.pools[(split, length)] = torch.tensor(starts, dtype=torch.long)

    def choose(self, split: str, length: int, batch: int, generator: torch.Generator) -> torch.Tensor:
        pool = self.pools[(split, length)]
        if not len(pool):
            raise ValueError(f"no {split} windows of length {length}")
        return pool[torch.randint(len(pool), (batch,), generator=generator)]

    def frame_batch(self, split: str, batch: int, generator: torch.Generator, device) -> tuple:
        rows = self.train_rows if split == "train" else self.val_rows
        idx = rows[torch.randint(len(rows), (batch,), generator=generator)]
        return self.visual[idx].to(device), self.proprio[idx].to(device)

    def window(self, starts: torch.Tensor, length: int, z: torch.Tensor, cfg: dict, device) -> dict:
        starts = starts.long()
        hpos = starts[:, None] - cfg["history_stride"] * torch.arange(cfg["history_tokens"] - 1, -1, -1)[None]
        end = starts + length
        epos = end[:, None] - cfg["history_stride"] * torch.arange(cfg["history_tokens"] - 1, -1, -1)[None]
        step = torch.arange(length)[None]
        path = torch.arange(length + 1)[None]
        return {
            "hist_z": z[hpos].to(device), "hist_actions": self.actions[hpos - 1].to(device),
            "end_hist_z": z[epos].to(device), "end_hist_actions": self.actions[epos - 1].to(device),
            "z0": z[starts].to(device), "actions": self.actions[starts[:, None] + step].to(device),
            "z_path": z[starts[:, None] + path].to(device),
        }


def sample_future(z_path: torch.Tensor, samples: int) -> torch.Tensor:
    length = z_path.shape[1] - 1
    pos = torch.linspace(1, length, samples, device=z_path.device).round().long()
    return z_path[:, pos]


def resample(sequence: torch.Tensor, samples: int) -> torch.Tensor:
    pos = torch.linspace(0, sequence.shape[1] - 1, samples, device=sequence.device).round().long()
    return sequence[:, pos]


def raw_signature(z_path: torch.Tensor, dt: float) -> torch.Tensor:
    return flatten(signature(integrated_lift(z_path.float(), dt), 2))


def as_signature(raw: torch.Tensor, channels: int) -> tuple:
    if raw.shape[-1] != channels + channels * channels:
        raise ValueError("wrong degree-2 signature size")
    return (torch.ones_like(raw[..., :1]), raw[..., :channels], raw[..., channels:])


class Targets:
    def __init__(self, cfg, summary, sig_decoder, sig_mean, sig_std):
        self.cfg, self.summary, self.sig_decoder = cfg, summary, sig_decoder
        self.sig_mean, self.sig_std = sig_mean, sig_std

    @torch.no_grad()
    def get(self, kind, z_path, length):
        sampled = sample_future(z_path, self.cfg["samples_per_segment"])
        if kind == "ordered_frames":
            rep = sampled.flatten(1)
        elif kind == "learned_summary":
            rep = self.summary.encode(sampled).flatten(1)
        elif kind == "signature_d2":
            rep = (raw_signature(z_path, self.cfg["dt_seconds"]) - self.sig_mean) / self.sig_std
        else:
            raise KeyError(kind)
        return rep, z_path[:, -1], sampled

    def decode(self, kind, rep, endpoint, length):
        if kind == "ordered_frames":
            return rep.reshape(len(rep), self.cfg["samples_per_segment"], self.cfg["frame_latent_dim"])
        if kind == "learned_summary":
            return self.summary.decode(rep.reshape(len(rep), self.cfg["summary_tokens"], self.cfg["summary_dim"]), length)
        return self.sig_decoder(rep, endpoint, length)

    def signature_raw(self, normalized):
        return normalized * self.sig_std + self.sig_mean

    def signature_normalized(self, raw):
        return (raw - self.sig_mean) / self.sig_std


def optimize(loss, optimizer, params, grad_clip):
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(params, grad_clip)
    optimizer.step()


def mean_dict(rows):
    return {k: sum(r[k] for r in rows) / len(rows) for k in rows[0]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise RuntimeError("step-3 pilot requires an sbatch GPU job")
    base_cfg = json.loads(args.config.read_text())
    cfg = dict(base_cfg)
    if args.profile:
        cfg.update({k: v for k, v in cfg["profile"].items() if not k.startswith("max_")})
    args.run_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.run_dir / "result.json"
    result = {"verdict": "RUNNING", "job_id": os.environ["SLURM_JOB_ID"], "node": platform.node(),
              "profile": args.profile, "protocol": "docs/TRAJECTORY_SSL_STEP3_PROTOCOL.md", "config": cfg,
              "labels_loaded": False, "test_episodes_loaded": 0, "grounded_checkpoints_loaded": False}
    write_json(result_path, result)
    started = time.monotonic()

    def budget():
        if time.monotonic() - started > cfg["time_budget_seconds"]:
            raise TimeoutError("internal step-3 time budget exceeded")

    try:
        device = torch.device("cuda")
        torch.manual_seed(cfg["seed"])
        random.seed(cfg["seed"])
        generator = torch.Generator().manual_seed(cfg["seed"])
        data = PackedData(Path(cfg["feature_root"]), base_cfg, args.profile)
        result["data"] = {"episodes": len(data.entries), "train_episodes": sum(m["split"] == "train" for m in data.entries),
                          "val_episodes": sum(m["split"] == "val" for m in data.entries),
                          "frames": len(data.visual)}
        write_json(result_path, result)

        # Stage A: shared label-free frame bottleneck.
        frame_ae = FrameAE(cfg["frame_latent_dim"]).to(device)
        opt = torch.optim.AdamW(frame_ae.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
        frame_log = []
        ae_batch = min(256, max(cfg["batch_size"], 64))
        for step in range(cfg["frame_ae_steps"]):
            visual, proprio = data.frame_batch("train", ae_batch, generator, device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss, terms = frame_ae.loss(visual, proprio)
            optimize(loss, opt, frame_ae.parameters(), cfg["grad_clip"])
            if step % max(1, cfg["frame_ae_steps"] // 5) == 0:
                print(f"frame_ae {step}/{cfg['frame_ae_steps']} loss={float(loss):.5f}", flush=True)
                frame_log.append({"step": step, "loss": float(loss.detach()), **terms})
                budget()
        frame_ae.eval().requires_grad_(False)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            train_v, train_p = data.frame_batch("train", min(1024, len(data.train_rows)), generator, device)
            val_v, val_p = data.frame_batch("val", min(1024, len(data.val_rows)), generator, device)
            train_loss, _ = frame_ae.loss(train_v, train_p)
            val_loss, _ = frame_ae.loss(val_v, val_p)
        result["frame_autoencoder"] = {"train_loss": float(train_loss), "val_loss": float(val_loss), "log": frame_log,
                                       "parameters": sum(p.numel() for p in frame_ae.parameters())}
        torch.save(frame_ae.state_dict(), args.run_dir / "frame_ae.pt")
        del train_v, train_p, val_v, val_p, opt

        # Encode all non-test frames, then release the 8GB patch cache.
        z_chunks = []
        with torch.no_grad():
            for start in range(0, len(data.visual), 1024):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    z_chunks.append(frame_ae.encoder(data.visual[start:start+1024].to(device),
                                                     data.proprio[start:start+1024].to(device)).float().cpu())
                if start % (1024 * 50) == 0:
                    print(f"encoded frame latents {start}/{len(data.visual)}", flush=True)
                    budget()
        z = torch.cat(z_chunks)
        del z_chunks, data.visual
        gc.collect()

        # Stage B1: freely learned summary target autoencoder.
        summary = SummaryAE(cfg["frame_latent_dim"], cfg["summary_dim"], cfg["summary_tokens"],
                            cfg["samples_per_segment"]).to(device)
        opt = torch.optim.AdamW(summary.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
        for step in range(cfg["target_ae_steps"]):
            length = cfg["train_lengths"][step % len(cfg["train_lengths"])]
            starts = data.choose("train", length, cfg["batch_size"], generator)
            wb = data.window(starts, length, z, base_cfg, device)
            frames = sample_future(wb["z_path"], cfg["samples_per_segment"])
            with torch.autocast("cuda", dtype=torch.bfloat16):
                encoded = summary.encode(frames)
                reconstructed = summary.decode(encoded, length)
                loss = F.mse_loss(reconstructed.float(), frames.float())
            optimize(loss, opt, summary.parameters(), cfg["grad_clip"])
            if step % max(1, cfg["target_ae_steps"] // 5) == 0:
                print(f"summary_ae {step}/{cfg['target_ae_steps']} loss={float(loss):.5f}", flush=True)
                budget()
        summary.eval().requires_grad_(False)
        torch.save(summary.state_dict(), args.run_dir / "summary_ae.pt")
        del opt

        # Stage B2: train-only normalization and decoder for exact degree-2 targets.
        collected = []
        for i in range(32 if args.profile else 128):
            length = cfg["train_lengths"][i % len(cfg["train_lengths"])]
            starts = data.choose("train", length, min(128, cfg["batch_size"] * 2), generator)
            collected.append(raw_signature(data.window(starts, length, z, base_cfg, device)["z_path"], cfg["dt_seconds"]).cpu())
        collected = torch.cat(collected)
        sig_mean = collected.mean(0).to(device)
        sig_std = collected.std(0, unbiased=False).clamp_min(1e-4).to(device)
        sig_dim = sig_mean.numel()
        expected = (cfg["frame_latent_dim"] + 1) + (cfg["frame_latent_dim"] + 1) ** 2
        if sig_dim != expected:
            raise ValueError(f"signature size {sig_dim}, expected {expected}")
        sig_decoder = SignatureDecoder(sig_dim, cfg["frame_latent_dim"], cfg["samples_per_segment"]).to(device)
        opt = torch.optim.AdamW(sig_decoder.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
        for step in range(cfg["target_ae_steps"]):
            length = cfg["train_lengths"][step % len(cfg["train_lengths"])]
            starts = data.choose("train", length, cfg["batch_size"], generator)
            wb = data.window(starts, length, z, base_cfg, device)
            frames = sample_future(wb["z_path"], cfg["samples_per_segment"])
            target = (raw_signature(wb["z_path"], cfg["dt_seconds"]) - sig_mean) / sig_std
            with torch.autocast("cuda", dtype=torch.bfloat16):
                reconstructed = sig_decoder(target, wb["z_path"][:, -1], length)
                loss = F.mse_loss(reconstructed.float(), frames.float())
            optimize(loss, opt, sig_decoder.parameters(), cfg["grad_clip"])
            if step % max(1, cfg["target_ae_steps"] // 5) == 0:
                print(f"signature_decoder {step}/{cfg['target_ae_steps']} loss={float(loss):.5f}", flush=True)
                budget()
        sig_decoder.eval().requires_grad_(False)
        torch.save({"state_dict": sig_decoder.state_dict(), "mean": sig_mean.cpu(), "std": sig_std.cpu()},
                   args.run_dir / "signature_decoder.pt")
        del opt, collected

        targets = Targets(cfg, summary, sig_decoder, sig_mean, sig_std)
        rep_dims = {"ordered_frames": cfg["samples_per_segment"] * cfg["frame_latent_dim"],
                    "learned_summary": cfg["summary_tokens"] * cfg["summary_dim"],
                    "signature_d2": sig_dim}

        # Observed-target decoder ceilings, including unseen length 128.
        ceilings = {}
        for length in (32, 64, 128):
            starts = data.choose("val", length, min(cfg["eval_windows_per_length"], len(data.pools[("val", length)])), generator)
            wb = data.window(starts, length, z, base_cfg, device)
            for kind in ARMS:
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    rep, endpoint, frames = targets.get(kind, wb["z_path"], length)
                    decoded = targets.decode(kind, rep, endpoint, length)
                ceilings[f"{kind}|{length}"] = float(F.mse_loss(decoded.float(), frames.float()))
        result["target_ceiling_mse"] = ceilings
        write_json(result_path, result)

        # Stage C: matched action-conditioned predictors and memory updates.
        arms = {}
        training_logs = {}
        for arm_index, kind in enumerate(ARMS):
            torch.manual_seed(cfg["seed"] + 100)  # identical shared-module initialization where shapes permit
            arm = TrajectoryArm(kind, cfg["frame_latent_dim"], cfg["model_dim"], rep_dims[kind]).to(device)
            opt = torch.optim.AdamW(arm.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
            rows = []
            for step in range(cfg["predictor_steps"]):
                length = cfg["train_lengths"][step % len(cfg["train_lengths"])]
                starts = data.choose("train", length, cfg["batch_size"], generator)
                wb = data.window(starts, length, z, base_cfg, device)
                with torch.no_grad():
                    rep_t, endpoint_t, _ = targets.get(kind, wb["z_path"], length)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    h = arm.history(wb["hist_z"], wb["hist_actions"])
                    h_end = arm.history(wb["end_hist_z"], wb["end_hist_actions"]).detach()
                    rep, endpoint = arm.predictor(h, wb["z0"], wb["actions"])
                    pred_loss = F.smooth_l1_loss(rep.float(), rep_t.float()) + F.mse_loss(endpoint.float(), endpoint_t.float())
                    update_loss = F.mse_loss(arm.advance(h, rep, endpoint).float(), h_end.float())
                    rollout_loss = rep.new_zeros(())
                    if length == 64:
                        left_path, right_path = wb["z_path"][:, :33], wb["z_path"][:, 32:]
                        with torch.no_grad():
                            left_t, left_end_t, _ = targets.get(kind, left_path, 32)
                            right_t, right_end_t, _ = targets.get(kind, right_path, 32)
                        left, left_end = arm.predictor(h, wb["z0"], wb["actions"][:, :32])
                        h_roll = arm.advance(h, left, left_end)
                        right, right_end = arm.predictor(h_roll, left_end, wb["actions"][:, 32:])
                        rollout_loss = (F.smooth_l1_loss(left.float(), left_t.float()) + F.mse_loss(left_end.float(), left_end_t.float())
                                        + F.smooth_l1_loss(right.float(), right_t.float()) + F.mse_loss(right_end.float(), right_end_t.float()))
                    loss = pred_loss + update_loss + rollout_loss
                optimize(loss, opt, arm.parameters(), cfg["grad_clip"])
                if step % max(1, cfg["predictor_steps"] // 6) == 0:
                    row = {"step": step, "loss": float(loss.detach()), "prediction": float(pred_loss.detach()),
                           "update": float(update_loss.detach()), "rollout": float(rollout_loss.detach())}
                    rows.append(row)
                    print(f"{kind} {step}/{cfg['predictor_steps']} loss={row['loss']:.5f}", flush=True)
                    budget()
            arm.eval().requires_grad_(False)
            torch.save(arm.state_dict(), args.run_dir / f"arm_{kind}.pt")
            arms[kind] = arm
            training_logs[kind] = rows
            del opt
        result["training_log"] = training_logs
        result["parameters"] = {k: sum(p.numel() for p in v.parameters()) for k, v in arms.items()}

        # Direct development evaluation.
        direct = {}
        with torch.no_grad():
            for length in (32, 64, 128):
                count = min(cfg["eval_windows_per_length"], len(data.pools[("val", length)]))
                starts = data.choose("val", length, count, generator)
                wb = data.window(starts, length, z, base_cfg, device)
                for kind, arm in arms.items():
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        h = arm.history(wb["hist_z"], wb["hist_actions"])
                        rep, endpoint = arm.predictor(h, wb["z0"], wb["actions"])
                        rep_t, endpoint_t, frames = targets.get(kind, wb["z_path"], length)
                        decoded = targets.decode(kind, rep, endpoint, length)
                    direct[f"{kind}|{length}"] = {
                        "common_future_mse": float(F.mse_loss(decoded.float(), frames.float())),
                        "endpoint_mse": float(F.mse_loss(endpoint.float(), endpoint_t.float())),
                        "normalized_target_mse": float(F.mse_loss(rep.float(), rep_t.float())), "windows": count}

        # Deployment-style split rollout; no future observation in right prediction.
        split_metrics = {}
        channels = cfg["frame_latent_dim"] + 1
        with torch.no_grad():
            for split_name, parts in cfg["eval_splits"].items():
                total = sum(parts)
                count = min(cfg["eval_windows_per_length"], len(data.pools[("val", total)]))
                starts = data.choose("val", total, count, generator)
                wb = data.window(starts, total, z, base_cfg, device)
                truth = sample_future(wb["z_path"], cfg["samples_per_segment"])
                for kind, arm in arms.items():
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        h = arm.history(wb["hist_z"], wb["hist_actions"])
                        current, cursor = wb["z0"], 0
                        decoded_parts, raw_parts, last_endpoint = [], [], None
                        for length in parts:
                            act = wb["actions"][:, cursor:cursor+length]
                            rep, endpoint = arm.predictor(h, current, act)
                            decoded_parts.append(targets.decode(kind, rep, endpoint, length))
                            if kind == "signature_d2":
                                raw_parts.append(targets.signature_raw(rep))
                            h, current, last_endpoint = arm.advance(h, rep, endpoint), endpoint, endpoint
                            cursor += length
                        sequential = resample(torch.cat(decoded_parts, 1), cfg["samples_per_segment"])
                    entry = {"windows": count, "sequential_common_mse": float(F.mse_loss(sequential.float(), truth.float()))}
                    if kind == "signature_d2":
                        composed = as_signature(raw_parts[0].float(), channels)
                        for raw in raw_parts[1:]:
                            composed = chen(composed, as_signature(raw.float(), channels))
                        normalized = targets.signature_normalized(flatten(composed))
                        with torch.autocast("cuda", dtype=torch.bfloat16):
                            whole = targets.decode(kind, normalized, last_endpoint, total)
                        entry["fixed_composed_common_mse"] = float(F.mse_loss(whole.float(), truth.float()))
                    split_metrics[f"{kind}|{split_name}"] = entry

        # Locked development decision.
        ceiling_ratio = ceilings["signature_d2|128"] / max(ceilings["learned_summary|128"], 1e-12)
        comparisons = {}
        conditions = [ceiling_ratio <= 1.20]
        for name in ("64=24+40", "128=64+64"):
            sig = split_metrics[f"signature_d2|{name}"]["fixed_composed_common_mse"]
            seq = split_metrics[f"signature_d2|{name}"]["sequential_common_mse"]
            best = min(split_metrics[f"ordered_frames|{name}"]["sequential_common_mse"],
                       split_metrics[f"learned_summary|{name}"]["sequential_common_mse"])
            comparisons[name] = {"signature": sig, "signature_sequential": seq, "best_baseline": best,
                                 "relative_improvement_vs_best": (best - sig) / best,
                                 "relative_change_vs_own_sequential": (sig - seq) / seq}
            conditions += [sig <= best, (sig - seq) / seq <= 0.05]
        conditions.append(any(v["relative_improvement_vs_best"] >= 0.10 for v in comparisons.values()))
        verdict = "SIGNATURE_MECHANISM_LEAD" if all(conditions) else "NO_SIGNATURE_MECHANISM_LEAD_IN_STEP3_PILOT"
        result.update({"direct": direct, "split_rollout": split_metrics, "decision": {"verdict": verdict,
                       "signature_to_summary_ceiling_ratio_128": ceiling_ratio, "comparisons": comparisons,
                       "conditions": conditions}, "elapsed_seconds": time.monotonic() - started,
                       "verdict": "PROFILE_PASS" if args.profile else verdict,
                       "interpretation": "Development representation/prediction result only; no task-value, candidate-ranking, memory-VLA or control result."})
        write_json(result_path, result)
        print(json.dumps(result["decision"], indent=2), flush=True)
    except Exception:
        result.update({"verdict": "FAILED", "error": traceback.format_exc(), "elapsed_seconds": time.monotonic() - started})
        write_json(result_path, result)
        raise


if __name__ == "__main__":
    main()
