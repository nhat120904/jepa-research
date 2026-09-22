"""Bounded development pilot, prospective queries, matched candidate banks.

No policy/control claim: evaluation here is offline query prediction and selection.
Profile numbers only exercise plumbing, never decide the research hypothesis.
"""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from .runtime import require_slurm
from .branch_data import BranchStore, QUERY_NAMES
from .branch_models import ReconstructionCodec, SpatialForecast, QueryOnly
from .models import SpatialObservedCodec, SpatialActionSummary, QueryReader, DirectQueryPredictor


def freeze(module):
    module.eval().requires_grad_(False)
    return module


def train_steps(label, modules, store, steps, batch_size, seed, loss_fn):
    for module in modules:
        module.train()
    parameters = [p for m in modules for p in m.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=3e-4, weight_decay=.01)
    rng = np.random.default_rng(seed)
    start = time.monotonic()
    first = last = None
    for step in range(steps):
        batch = store.sample(rng, batch_size, "cuda")
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(batch)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"{label}: nonfinite loss")
        loss.backward()
        norm = nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
        if step == 0 and not norm.item() > 0:
            raise AssertionError(f"{label}: zero gradient")
        optimizer.step()
        last = float(loss.detach())
        if first is None:
            first = last
        if step == 0 or (step+1) % max(1, steps//4) == 0:
            print(f"{label} {step+1}/{steps} loss={last:.6f}", flush=True)
    torch.cuda.synchronize()
    return {"steps": steps, "seconds": time.monotonic()-start,
            "first_loss": first, "last_loss": last,
            "trainable_parameters": sum(p.numel() for p in parameters)}


def bootstrap(values, seed=19):
    values = np.asarray(values, float)
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, (2000, len(values)), replace=True).mean(1)
    return [float(x) for x in np.quantile(samples, [.025, .975])]


def main():
    require_slurm()
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--seed", type=int, default=20260922)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("GPU sbatch required")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    store = BranchStore(args.feature_root)
    p, d, w, m = store.patches, store.dim, 128, 4
    batch_size = 16
    codec_steps, reader_steps, predictor_steps = (20,20,20) if args.profile else (1200,600,1800)
    # Feature scale from TRAIN only, shared by both spatial reconstruction arms.
    scale = float(torch.stack([store.rows[i]["future"].float().square().mean()
                               for i in store.train[:16]]).mean().sqrt().clamp_min(1e-6))
    logs = {}
    codec = SpatialObservedCodec(p,d,w,m).cuda()
    reader = QueryReader(store.query_dim,w).cuda()
    logs["query_codec"] = train_steps("query_codec", [codec,reader],store,codec_steps,
        batch_size,args.seed,lambda b: nn.functional.mse_loss(reader(codec(b["future"]),
                                                                  b["queries"]),b["answers"]))
    freeze(codec); freeze(reader)
    generic = ReconstructionCodec(p,d,w,m).cuda()
    logs["generic_codec"] = train_steps("generic_codec",[generic],store,codec_steps,
        batch_size,args.seed,lambda b: nn.functional.mse_loss(
            generic.reconstruct(generic(b["future"]),b["actions"].shape[1]),b["future"]/scale))
    freeze(generic)
    generic_reader = QueryReader(store.query_dim,w).cuda()
    logs["generic_reader"] = train_steps("generic_reader",[generic_reader],store,reader_steps,
        batch_size,args.seed,lambda b: nn.functional.mse_loss(
            generic_reader(generic(b["future"]),b["queries"]),b["answers"]))
    freeze(generic_reader)

    models, readers = {}, {}
    for name, target, readout in (("query_summary",codec,reader),
                                  ("generic_summary",generic,generic_reader)):
        torch.manual_seed(args.seed+1)  # matched student initialization and batches
        model = SpatialActionSummary(p,d,width=w,tokens=m).cuda()
        def summary_loss(b, model=model, target=target, readout=readout):
            z = model(b["history"],b["actions"])
            with torch.no_grad():
                goal = target(b["future"])
            return (nn.functional.mse_loss(readout(z,b["queries"]),b["answers"])
                    + .1 * nn.functional.mse_loss(z,goal))
        logs[name] = train_steps(name,[model],store,predictor_steps,batch_size,args.seed,summary_loss)
        models[name], readers[name] = freeze(model), readout

    for name, endpoint in (("frame",False),("endpoint",True)):
        torch.manual_seed(args.seed+1)
        model = SpatialForecast(p,d,w,endpoint).cuda()
        readout = QueryReader(store.query_dim,w).cuda()
        def frame_loss(b, model=model, readout=readout, endpoint=endpoint):
            z = model(b["history"],b["actions"])
            future = b["future"][:, -1:] if endpoint else b["future"]
            return (nn.functional.mse_loss(readout(z,b["queries"]),b["answers"])
                    + .1*nn.functional.mse_loss(model.reconstruct(z),future/scale))
        logs[name] = train_steps(name,[model,readout],store,predictor_steps,batch_size,args.seed,frame_loss)
        models[name], readers[name] = freeze(model), freeze(readout)
    for name in ("direct", "no_action"):
        torch.manual_seed(args.seed+1)
        model = DirectQueryPredictor(store.query_dim,p,d,w).cuda()
        def direct_loss(b, model=model, name=name):
            actions = b["actions"] if name == "direct" else torch.zeros_like(b["actions"])
            return nn.functional.mse_loss(model(b["history"],actions,b["queries"]),b["answers"])
        logs[name] = train_steps(name,[model],store,predictor_steps,batch_size,args.seed,direct_loss)
        models[name] = freeze(model)
    torch.manual_seed(args.seed+1)
    query_only = QueryOnly(store.query_dim,w).cuda()
    logs["query_only"] = train_steps("query_only",[query_only],store,predictor_steps,batch_size,
        args.seed,lambda b: nn.functional.mse_loss(query_only(b["queries"]),b["answers"]))
    models["query_only"] = freeze(query_only)
    means = torch.stack([store.rows[i]["answers"] for i in store.train]).mean(0).numpy()
    predictions = []
    with torch.inference_mode():
        for horizon in (32,48,64):
            selected = [i for i in store.val if store.rows[i]["horizon"] == horizon]
            # Whole candidate groups: shuffling actions within prefix, not across scenes.
            for begin in range(0,len(selected),8):
                indices = selected[begin:begin+8]
                assert len({store.rows[i]["prefix_id"] for i in indices}) == 1
                b = store.batch(indices,"cuda")
                outputs = {"constant": np.tile(means,(len(indices),1)),
                           "observed_query_codec": reader(codec(b["future"]),b["queries"]).cpu().numpy(),
                           "observed_generic_codec": generic_reader(generic(b["future"]),b["queries"]).cpu().numpy()}
                for name,model in models.items():
                    if name == "query_only":
                        y = model(b["queries"])
                    elif name in ("direct","no_action"):
                        action = b["actions"] if name == "direct" else torch.zeros_like(b["actions"])
                        y = model(b["history"],action,b["queries"])
                    else:
                        y = readers[name](model(b["history"],b["actions"]),b["queries"])
                    outputs[name] = y.cpu().numpy()
                shuffled = b["actions"].roll(1,0)
                outputs["query_summary_shuffled"] = reader(
                    models["query_summary"](b["history"],shuffled),b["queries"]).cpu().numpy()
                for j,i in enumerate(indices):
                    row = store.rows[i]
                    predictions.append({"id":row["id"],"prefix_id":row["prefix_id"],
                        "horizon":horizon,"candidate_id":row["candidate_id"],
                        "target":row["answers"].tolist(),
                        "predictions":{name:y[j].tolist() for name,y in outputs.items()}})

    # Train-fit diagnostic only, computed AFTER all optimization; never selects a
    # checkpoint or alters training. Helps separate underfit from generalization.
    train_fit = {}
    with torch.inference_mode():
        for horizon in store.train_horizons:
            indices = [i for i in store.train if store.rows[i]["horizon"] == horizon]
            squared = {}
            for begin in range(0, len(indices), 16):
                b = store.batch(indices[begin:begin+16], "cuda")
                outputs = {
                    "observed_query_codec": reader(codec(b["future"]), b["queries"]),
                    "observed_generic_codec": generic_reader(generic(b["future"]), b["queries"]),
                    "constant": torch.as_tensor(means, device="cuda").expand(len(b["actions"]), -1),
                }
                for name, model in models.items():
                    if name == "query_only":
                        outputs[name] = model(b["queries"])
                    elif name in ("direct", "no_action"):
                        actions = b["actions"] if name == "direct" else torch.zeros_like(b["actions"])
                        outputs[name] = model(b["history"], actions, b["queries"])
                    else:
                        outputs[name] = readers[name](model(b["history"], b["actions"]), b["queries"])
                for name, values in outputs.items():
                    squared.setdefault(name, []).append((values-b["answers"]).square().cpu())
            train_fit[horizon] = {name: torch.cat(errors).mean(0).tolist()
                                  for name, errors in squared.items()}

    metrics, selection, contrasts = {}, {}, {}
    names = list(predictions[0]["predictions"])
    for horizon in (32,48,64):
        rows = [r for r in predictions if r["horizon"] == horizon]
        target = np.asarray([r["target"] for r in rows])
        metrics[horizon] = {}
        selection[horizon] = {}
        prefixes = sorted({r["prefix_id"] for r in rows})
        regrets = {name:[] for name in names}
        oracle_values,default_values = [],[]
        for prefix in prefixes:
            group = [r for r in rows if r["prefix_id"] == prefix]
            truth = np.asarray([r["target"][6] for r in group])
            oracle_values.append(float(truth.max()))
            default_values.append(float(next(r["target"][6] for r in group if r["candidate_id"]==1)))
            for name in names:
                scores = [r["predictions"][name][6] for r in group]
                regrets[name].append(float(truth.max()-truth[np.argmax(scores)]))
        for name in names:
            error = np.asarray([r["predictions"][name] for r in rows])-target
            metrics[horizon][name] = {"mae":np.abs(error).mean(0).tolist(),
                                      "mse":(error**2).mean(0).tolist()}
            selection[horizon][name] = {"mean_order_regret":float(np.mean(regrets[name])),
                "episode_bootstrap_ci":bootstrap(regrets[name]),
                "prefix_regrets":dict(zip(prefixes,regrets[name]))}
        selection[horizon]["bank"] = {"mean_oracle_order_score":float(np.mean(oracle_values)),
            "mean_default_order_score":float(np.mean(default_values)),
            "prefixes":len(prefixes), "candidates":8,
            "informative_prefixes":sum(np.ptp([r["target"][6] for r in rows
                                                if r["prefix_id"]==p]) > .05 for p in prefixes)}
        contrasts[horizon] = {}
        for baseline in ("frame","direct","generic_summary"):
            improvement = np.asarray(regrets[baseline])-np.asarray(regrets["query_summary"])
            contrasts[horizon][baseline] = {"regret_reduction":float(improvement.mean()),
                                           "paired_prefix_ci":bootstrap(improvement)}
    result = {"status":"PROFILE_ONLY" if args.profile else "DEVELOPMENT_ONE_SEED",
        "seed":args.seed,"queries":QUERY_NAMES,"feature_scale_train":scale,"logs":logs,
        "metrics":metrics,"selection":selection,"paired_selection_contrasts":contrasts,
        "train_fit_mse":train_fit,
        "architecture":{"width":w,"summary_tokens":m,"patch_count":p,"patch_dim":d,
                        "query_dim":store.query_dim,"batch_size":batch_size},
        "feature_root":str(args.feature_root),
        "test_read":False,"physical_success_evaluated":False,"control_evaluated":False,
        "train_prefixes":24,"val_prefixes":12,"train_horizons":[32,48],"ood_horizon":64,
        "limitations":["same three layouts, held-out prefixes only",
                       "hand-designed RGB query grammar and proposal; no policy learning",
                       "generic reader and joint query codec have different training schedules",
                       "profile is code validation, not a scientific comparison"]}
    # Convert NumPy scalar metadata without permitting NaN in artifacts.
    def default(x):
        if isinstance(x,np.generic): return x.item()
        raise TypeError(type(x).__name__)
    (args.output/"result.json").write_text(json.dumps(result,indent=2,allow_nan=False,default=default)+"\n")
    with (args.output/"predictions.jsonl").open("w") as f:
        for row in predictions: f.write(json.dumps(row,allow_nan=False)+"\n")
    torch.save({"models":{k:v.state_dict() for k,v in models.items()},
                "readers":{k:v.state_dict() for k,v in readers.items()},
                "codec":codec.state_dict(),"generic_codec":generic.state_dict(),
                "projection":store.projection,"result":result},args.output/"checkpoint.pt")
    print(f"COMPLETED {result['status']}",flush=True)


if __name__ == "__main__":
    main()
