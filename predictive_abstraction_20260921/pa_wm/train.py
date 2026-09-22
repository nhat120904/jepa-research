"""One-seed development pilot with matched query supervision. GPU/Slurm only."""
import argparse
import json
import os
import random
import time
from pathlib import Path

from .runtime import require_slurm


def atomic_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def main():
    require_slurm()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if "ENCODE_RUN" in config["feature_root"]:
        raise ValueError("Resolve feature_root to a completed encoder artifact first")
    args.run_dir.mkdir(parents=True, exist_ok=False)
    checkpoints = args.run_dir / "checkpoints"
    checkpoints.mkdir()

    import numpy as np
    import torch
    import torch.nn.functional as F
    from .data import EpisodeStore, WindowSampler
    from .models import (DirectQueryPredictor, EndpointPredictor,
                         FrameSequencePredictor, GenericSummaryCodec, QueryReader,
                         SpatialActionSummary, SpatialObservedCodec)

    if not torch.cuda.is_available():
        raise RuntimeError("Training requires requested GPU")
    seed = config["seed"]
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda")
    train_store = EpisodeStore(config["feature_root"], config["dataset_root"], "train")
    val_store = EpisodeStore(config["feature_root"], config["dataset_root"], "val")
    if (train_store.patch_count, train_store.patch_dim) != (val_store.patch_count, val_store.patch_dim):
        raise RuntimeError("Train/val feature mismatch")
    pc, pd = train_store.patch_count, train_store.patch_dim
    probe_sampler = WindowSampler(train_store, train_store, config["history"],
                                  config["queries_per_window"], seed)
    query_dim, width = probe_sampler.query_dim, config["width"]
    scale = config["profile_scale"] if args.profile else 1.0
    steps = lambda key: max(2, int(round(config[key] * scale)))
    batch_size = min(config["batch_size"], 4) if args.profile else config["batch_size"]

    def move(batch):
        history, actions, future, queries, answers, types = batch
        return (history.to(device), actions.to(device), future.to(device),
                queries.to(device), answers.to(device), types.to(device))

    def freeze(module):
        module.eval().requires_grad_(False)
        return module

    def train_loop(name, modules, step_count, loss_fn, sampler_seed):
        parameters = [p for module in modules for p in module.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(parameters, lr=config["learning_rate"],
                                      weight_decay=config["weight_decay"])
        sampler = WindowSampler(train_store, train_store, config["history"],
                                config["queries_per_window"], sampler_seed)
        horizon_rng = np.random.default_rng(sampler_seed + 1)
        history = []
        for step in range(step_count):
            horizon = int(horizon_rng.choice(config["train_horizons"]))
            batch = move(sampler.batch(batch_size, horizon))
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss, pieces = loss_fn(batch)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite loss in {name} step {step}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 5.0)
            optimizer.step()
            history.append({"step": step + 1, "loss": float(loss.detach()),
                            **{k: float(v.detach()) for k, v in pieces.items()}})
            interval = max(1, step_count // 10)
            if (step + 1) % interval == 0 or step == 0:
                print(f"{name} {step+1}/{step_count} {history[-1]}", flush=True)
        return history

    def eval_reader(name, summary_fn, batches=None):
        output = {}
        for horizon in config["eval_horizons"]:
            sampler = WindowSampler(val_store, train_store, config["history"],
                                    config["queries_per_window"], seed + 9000 + horizon)
            errors = [[] for _ in WindowSampler.OPS]
            predictions, targets = [[] for _ in WindowSampler.OPS], [[] for _ in WindowSampler.OPS]
            count = max(2, int(config["eval_batches"] * scale)) if batches is None else batches
            with torch.inference_mode():
                for _ in range(count):
                    batch = move(sampler.batch(batch_size, horizon))
                    history, actions, future, queries, answers, types = batch
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        predicted = summary_fn(history, actions, future, queries).float()
                    for op in range(len(WindowSampler.OPS)):
                        mask = types == op
                        values = (predicted[mask] - answers[mask]).square().cpu().numpy()
                        errors[op].extend(values.tolist())
                        predictions[op].extend(predicted[mask].cpu().numpy().tolist())
                        targets[op].extend(answers[mask].cpu().numpy().tolist())
            metrics = {}
            for op, label in enumerate(WindowSampler.OPS):
                x, y = np.asarray(predictions[op]), np.asarray(targets[op])
                correlation = float(np.corrcoef(x, y)[0, 1]) if x.std() > 0 and y.std() > 0 else 0.0
                metrics[label] = {"mse": float(np.mean(errors[op])), "r": correlation,
                                  "n": len(errors[op])}
            output[str(horizon)] = metrics
        print(f"evaluation {name}: {output}", flush=True)
        return output

    # A: query-sufficient observed codec.
    query_codec = SpatialObservedCodec(pc, pd, width, config["summary_tokens"]).to(device)
    query_reader = QueryReader(query_dim, width).to(device)
    def query_codec_loss(batch):
        _, _, future, queries, answers, _ = batch
        prediction = query_reader(query_codec(future), queries)
        answer = F.mse_loss(prediction, answers)
        return answer, {"answer": answer}
    logs = {"query_codec": train_loop("query_codec", [query_codec, query_reader],
            steps("codec_steps"), query_codec_loss, seed + 100)}
    torch.save({"codec": query_codec.state_dict(), "reader": query_reader.state_dict()},
               checkpoints / "query_codec.pt")
    freeze(query_codec); freeze(query_reader)

    # B: matched generic token codec, no query target in representation training.
    generic_codec = GenericSummaryCodec(pc, pd, width, config["summary_tokens"]).to(device)
    def generic_codec_loss(batch):
        _, _, future, _, _, _ = batch
        reconstruction = generic_codec.reconstruct(generic_codec(future), future.shape[1])
        target = future.mean(dim=2)
        latent = F.mse_loss(reconstruction, target)
        return latent, {"latent": latent}
    logs["generic_codec"] = train_loop("generic_codec", [generic_codec],
            steps("generic_steps"), generic_codec_loss, seed + 200)
    freeze(generic_codec)
    generic_reader = QueryReader(query_dim, width).to(device)
    def generic_reader_loss(batch):
        _, _, future, queries, answers, _ = batch
        with torch.no_grad(): summary = generic_codec(future)
        answer = F.mse_loss(generic_reader(summary, queries), answers)
        return answer, {"answer": answer}
    logs["generic_reader"] = train_loop("generic_reader", [generic_reader],
            steps("reader_steps"), generic_reader_loss, seed + 300)
    torch.save({"codec": generic_codec.state_dict(), "reader": generic_reader.state_dict()},
               checkpoints / "generic_codec.pt")
    freeze(generic_reader)

    results = {"observed_query_codec": eval_reader("observed_query_codec",
        lambda h, a, f, q: query_reader(query_codec(f), q)),
        "observed_generic_codec": eval_reader("observed_generic_codec",
        lambda h, a, f, q: generic_reader(generic_codec(f), q))}

    # Predictor arms. Seeds make every arm see exactly the same stochastic batch stream.
    predictor_seed = seed + 1000
    summary_predictor = SpatialActionSummary(pc, pd, width=width,
                                             tokens=config["summary_tokens"]).to(device)
    def summary_loss(batch):
        history, actions, future, queries, answers, _ = batch
        predicted = summary_predictor(history, actions)
        answer = F.mse_loss(query_reader(predicted, queries), answers)
        with torch.no_grad(): target = query_codec(future)
        latent = F.mse_loss(predicted, target)
        return answer + config["latent_weight"] * latent, {"answer": answer, "latent": latent}
    logs["query_summary_predictor"] = train_loop("query_summary_predictor",
        [summary_predictor], steps("predictor_steps"), summary_loss, predictor_seed)
    results["query_summary"] = eval_reader("query_summary",
        lambda h, a, f, q: query_reader(summary_predictor(h, a), q))
    torch.save(summary_predictor.state_dict(), checkpoints / "query_summary_predictor.pt")

    generic_predictor = SpatialActionSummary(pc, pd, width=width,
                                             tokens=config["summary_tokens"]).to(device)
    def generic_predictor_loss(batch):
        history, actions, future, queries, answers, _ = batch
        predicted = generic_predictor(history, actions)
        answer = F.mse_loss(generic_reader(predicted, queries), answers)
        with torch.no_grad(): target = generic_codec(future)
        latent = F.mse_loss(predicted, target)
        return answer + config["latent_weight"] * latent, {"answer": answer, "latent": latent}
    logs["generic_summary_predictor"] = train_loop("generic_summary_predictor",
        [generic_predictor], steps("predictor_steps"), generic_predictor_loss, predictor_seed)
    results["generic_summary"] = eval_reader("generic_summary",
        lambda h, a, f, q: generic_reader(generic_predictor(h, a), q))
    torch.save(generic_predictor.state_dict(), checkpoints / "generic_summary_predictor.pt")

    direct = DirectQueryPredictor(query_dim, pc, pd, width).to(device)
    def direct_loss(batch):
        history, actions, _, queries, answers, _ = batch
        answer = F.mse_loss(direct(history, actions, queries), answers)
        return answer, {"answer": answer}
    logs["direct_query"] = train_loop("direct_query", [direct], steps("predictor_steps"),
                                      direct_loss, predictor_seed)
    results["direct_query"] = eval_reader("direct_query",
        lambda h, a, f, q: direct(h, a, q))
    torch.save(direct.state_dict(), checkpoints / "direct_query.pt")

    frame = FrameSequencePredictor(pc, pd, width).to(device)
    frame_reader = QueryReader(query_dim, width).to(device)
    def frame_loss(batch):
        history, actions, future, queries, answers, _ = batch
        predicted = frame(history, actions)
        answer = F.mse_loss(frame_reader(predicted, queries), answers)
        # Deterministic DINO channel slice: no train/test-fitted projection.
        target = future.mean(dim=2)[..., :width]
        latent = F.mse_loss(predicted, target)
        return answer + config["latent_weight"] * latent, {"answer": answer, "latent": latent}
    logs["frame_wm"] = train_loop("frame_wm", [frame, frame_reader],
                                  steps("predictor_steps"), frame_loss, predictor_seed)
    results["frame_wm"] = eval_reader("frame_wm",
        lambda h, a, f, q: frame_reader(frame(h, a), q))
    torch.save({"model": frame.state_dict(), "reader": frame_reader.state_dict()},
               checkpoints / "frame_wm.pt")

    endpoint = EndpointPredictor(pc, pd, width).to(device)
    endpoint_reader = QueryReader(query_dim, width).to(device)
    def endpoint_loss(batch):
        history, actions, future, queries, answers, _ = batch
        predicted = endpoint(history, actions)
        answer = F.mse_loss(endpoint_reader(predicted, queries), answers)
        latent = F.mse_loss(predicted[:, 0], future.mean(dim=2)[:, -1, :width])
        return answer + config["latent_weight"] * latent, {"answer": answer, "latent": latent}
    logs["endpoint_wm"] = train_loop("endpoint_wm", [endpoint, endpoint_reader],
                                     steps("predictor_steps"), endpoint_loss, predictor_seed)
    results["endpoint_wm"] = eval_reader("endpoint_wm",
        lambda h, a, f, q: endpoint_reader(endpoint(h, a), q))
    torch.save({"model": endpoint.state_dict(), "reader": endpoint_reader.state_dict()},
               checkpoints / "endpoint_wm.pt")

    parameter_counts = {name: sum(p.numel() for p in module.parameters()) for name, module in {
        "query_codec": query_codec, "query_reader": query_reader,
        "generic_codec": generic_codec, "generic_reader": generic_reader,
        "query_summary_predictor": summary_predictor,
        "generic_summary_predictor": generic_predictor, "direct_query": direct,
        "frame_wm": frame, "frame_reader": frame_reader,
        "endpoint_wm": endpoint, "endpoint_reader": endpoint_reader}.items()}
    atomic_json(args.run_dir / "result.json", {
        "status": "COMPLETED", "job_id": os.environ["SLURM_JOB_ID"],
        "profile": args.profile, "seed": seed, "test_split_read": False,
        "feature_root": config["feature_root"], "train_episodes": len(train_store.episodes),
        "val_episodes": len(val_store.episodes), "query_types": WindowSampler.OPS,
        "results": results, "parameter_counts": parameter_counts,
        "steps": {key: steps(key) for key in ("codec_steps", "generic_steps", "reader_steps",
                                                   "predictor_steps")},
        "interpretation": "development pilot; offline query accuracy is not control evidence"
    })
    torch.save(logs, args.run_dir / "training_logs.pt")
    print("COMPLETED train/val pilot; sealed test not read.", flush=True)


if __name__ == "__main__":
    main()
