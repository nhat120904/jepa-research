"""Bounded observed-future diagnosis: feature probe and matched codec/loss ablations.

Not a world-model/control result: EVERY arm here can observe the true future.
No simulator state, reward, policy, new data collection, or test-set access.
"""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from .runtime import require_slurm
from .branch_data import BranchStore
from .models import SpatialObservedCodec, QueryReader, positions
from .queries import chroma_features, similarity


class ObservedReader(nn.Module):
    def __init__(self, patches, dim, query_dim, compressed, width=128):
        super().__init__()
        self.codec = SpatialObservedCodec(patches, dim, width, 4)
        self.reader = QueryReader(query_dim, width)
        self.compressed = compressed

    def forward(self, future, queries):
        if self.compressed:
            tokens = self.codec(future)
        else:
            x = self.codec.project(future)
            x = x + positions(x.shape[1], x.shape[2], x.device, x.dtype)
            tokens = self.codec.encoder(x)
        return self.reader(tokens, queries)


def candidate_difference_loss(predicted, target, groups, margin=.05):
    """Within-prefix difference regression, all eight query types, no cross-prefix pairs."""
    p = predicted.reshape(groups, -1, predicted.shape[-1])
    y = target.reshape_as(p)
    dp, dy = p[:,:,None,:]-p[:,None,:,:], y[:,:,None,:]-y[:,None,:,:]
    mask = dy.abs() > margin
    return ((dp-dy).square()*mask).sum()/mask.sum().clamp_min(1)


def aggregates(trace):
    """Differentiable exact native query rules from predicted A/B similarity traces."""
    a, b = trace.unbind(-1)
    ab = torch.minimum(a[:,:-1].cummax(1).values,b[:,1:]).amax(1)
    ba = torch.minimum(b[:,:-1].cummax(1).values,a[:,1:]).amax(1)
    return torch.stack([a.amax(1),b.amax(1),a.mean(1),b.mean(1),a[:,-1],b[:,-1],ab,ba],-1)


class FrameMetric(nn.Module):
    """Learned visual similarity, not a state/contact probe. Dense RGB supervision."""
    def __init__(self, dimension, mean, std):
        super().__init__()
        self.register_buffer("mean",mean)
        self.register_buffer("std",std)
        self.embed = nn.Sequential(nn.Linear(dimension,128),nn.GELU(),nn.Linear(128,64))

    def forward(self, frame, anchor):
        x = self.embed((frame-self.mean)/self.std)
        y = self.embed((anchor-self.mean)/self.std)
        return torch.exp(-(x-y).square().mean(-1))


def groups_for(store, split):
    groups = {}
    for i,r in enumerate(store.rows):
        if r["split"] == split:
            groups.setdefault((r["prefix_id"],r["horizon"]),[]).append(i)
    expected = [1,2,3,17,53,89,100,120]
    for indices in groups.values():
        if [store.rows[i]["candidate_id"] for i in indices] != expected:
            raise AssertionError("Incomplete or reordered candidate group")
    return groups


def train_codec(model, store, groups, steps, pair_weight, seed, tiny=False):
    rng = np.random.default_rng(seed)
    opt = torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=.01)
    losses=[]
    keys=list(groups)
    for step in range(steps):
        h = keys[0][1] if tiny else int(rng.choice([32,48]))
        pool=[k for k in keys if k[1]==h]
        count=1 if tiny else 2
        picked=rng.choice(len(pool),count,replace=False)
        indices=[i for k in picked for i in groups[pool[k]]]
        b=store.batch(indices,"cuda")
        model.train()
        y=model(b["future"],b["queries"])
        loss=nn.functional.mse_loss(y,b["answers"])
        loss=loss+pair_weight*candidate_difference_loss(y,b["answers"],count)
        opt.zero_grad(set_to_none=True)
        if not torch.isfinite(loss): raise FloatingPointError("Nonfinite codec loss")
        loss.backward()
        norm=nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        if step==0 and norm.item()==0: raise AssertionError("No codec gradient")
        opt.step()
        losses.append(float(loss.detach()))
        if (step+1)%max(1,steps//4)==0:
            print(f"codec compressed={model.compressed} pair={pair_weight} tiny={tiny} "
                  f"step={step+1}/{steps} loss={losses[-1]:.6f}",flush=True)
    return {"first_loss":losses[0],"last_loss":losses[-1],"steps":steps}


@torch.inference_mode()
def evaluate(store, groups, predict):
    rows=[]
    for (prefix,h),indices in groups.items():
        b=store.batch(indices,"cuda")
        y=predict(indices,b).cpu().numpy()
        truth=b["answers"].cpu().numpy()
        selected=int(y[:,6].argmax())
        rows.append({"prefix":prefix,"horizon":h,"mse":((y-truth)**2).mean(0).tolist(),
            "selected_candidate":store.rows[indices[selected]]["candidate_id"],
            "selected_score":float(truth[selected,6]),"oracle_score":float(truth[:,6].max()),
            "default_score":float(truth[0,6]),"regret":float(truth[:,6].max()-truth[selected,6]),
            "target_spread":float(np.ptp(truth[:,6])),
            "predicted_spread":float(np.ptp(y[:,6]))})
    summary={}
    for h in sorted({r["horizon"] for r in rows}):
        selected=[r for r in rows if r["horizon"]==h]
        summary[h]={"mse":np.mean([r["mse"] for r in selected],axis=0).tolist(),
                    "regret":float(np.mean([r["regret"] for r in selected])),
                    "prefixes":len(selected)}
    return {"summary":summary,"groups":rows}


def visual_probe(store, feature_root, train_groups, val_groups, steps):
    """Learn frame/anchor metric on TRAIN, then apply fixed temporal aggregation."""
    manifest=json.loads((feature_root/"manifest.json").read_text())
    raw_root=Path(manifest["dataset_root"])
    anchor_dim=store.patches*store.projection.shape[1]
    frames,anchors,traces={},{},{}
    for i,row in enumerate(store.rows):
        frames[i]=(row["future"].float()@store.projection).flatten(1)
        anchors[i]=torch.stack([row["queries"][0,:anchor_dim],row["queries"][1,:anchor_dim]])
        with np.load(raw_root/row["source"]) as data:
            features=chroma_features(data["rgb"][1:])
            af,bf=chroma_features(data["anchors"])
            trace=np.stack([similarity(features,af),similarity(features,bf)],-1)
        traces[i]=torch.from_numpy(trace)
        torch.testing.assert_close(aggregates(traces[i][None])[0],row["answers"],atol=2e-6,rtol=2e-6)
    x=torch.cat([frames[i] for i in store.train])
    mean,std=x.mean(0).cuda(),x.std(0).clamp_min(.01).cuda()
    pairs=[(i,t,a) for i in store.train for t in range(len(frames[i])) for a in range(2)]
    positive=[v for v in pairs if traces[v[0]][v[1],v[2]]>.05]
    negative=[v for v in pairs if traces[v[0]][v[1],v[2]]<=.05]
    if not positive or not negative: raise ValueError("Frame probe needs both label strata")
    torch.manual_seed(20260923)
    model=FrameMetric(x.shape[1],mean,std).cuda()
    opt=torch.optim.AdamW(model.parameters(),lr=3e-4)
    rng=np.random.default_rng(20260923)
    for step in range(steps):
        batch=[pool[int(j)] for pool in (positive,negative) for j in rng.integers(len(pool),size=128)]
        f=torch.stack([frames[i][t] for i,t,a in batch]).cuda()
        a=torch.stack([anchors[i][a] for i,t,a in batch]).cuda()
        truth=torch.stack([traces[i][t,a] for i,t,a in batch]).cuda()
        y=model(f,a)
        loss=nn.functional.mse_loss(y,truth)
        if not torch.isfinite(loss): raise FloatingPointError("Nonfinite frame-metric loss")
        opt.zero_grad(set_to_none=True); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True); opt.step()
        if (step+1)%max(1,steps//4)==0: print(f"frame_metric {step+1}/{steps} {loss.item():.6f}",flush=True)
    model.eval()
    def predict(indices,b):
        f=torch.stack([frames[i] for i in indices]).cuda()
        a=torch.stack([anchors[i] for i in indices]).cuda()
        trace=model(f[:,:,None,:].expand(-1,-1,2,-1),a[:,None,:,:].expand(-1,f.shape[1],-1,-1))
        return aggregates(trace)
    result={"train":evaluate(store,train_groups,predict),"val":evaluate(store,val_groups,predict),
            "dense_supervision":True,"positive_frame_pairs":len(positive),
            "negative_frame_pairs":len(negative),"steps":steps}
    return result,model.cpu().state_dict()


def main():
    require_slurm()
    parser=argparse.ArgumentParser()
    parser.add_argument("--features",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4)
    if not torch.cuda.is_available(): raise RuntimeError("GPU job required")
    start=time.monotonic()
    store=BranchStore(args.features)
    train,val=groups_for(store,"train"),groups_for(store,"val")
    result={"status":"OBSERVED_FUTURE_DIAGNOSTIC_NOT_WM_RESULT","test_read":False,
            "train_prefixes":24,"val_prefixes":12,"tiny_fit":{},"arms":{}}
    weights={}
    probe,weights["frame_metric"]=visual_probe(store,args.features,train,val,1000)
    result["frame_metric"]=probe
    # Choose informative tiny unit using TRAIN labels only; explicitly not an accuracy test.
    def variation(key):
        y=torch.stack([store.rows[i]["answers"] for i in train[key]])[:,6]
        return float(y.var(unbiased=False))
    tiny_key=max([k for k in train if k[1]==32],key=variation)
    variance=variation(tiny_key)
    if variance<1e-6: raise ValueError("No informative train group for tiny fit")
    result["tiny_prefix"]=tiny_key; result["tiny_constant_mse"]=variance
    tiny_pass=False
    for compressed in (False,True):
        torch.manual_seed(20260923)
        model=ObservedReader(store.patches,store.dim,store.query_dim,compressed).cuda()
        log=train_codec(model,store,{tiny_key:train[tiny_key]},600,1.,20260923,tiny=True)
        model.eval()
        fit=evaluate(store,{tiny_key:train[tiny_key]},lambda i,b:model(b["future"],b["queries"]))
        passed=fit["summary"][32]["mse"][6]<.1*variance
        tiny_pass|=passed
        name="summary" if compressed else "sequence"
        result["tiny_fit"][name]={"fit":fit,"log":log,"pass":passed}
        del model
    if tiny_pass:
        for compressed in (False,True):
            for pair_weight in (0.,1.):
                name=("summary" if compressed else "sequence")+("_paired" if pair_weight else "_mse")
                torch.manual_seed(20260923)
                model=ObservedReader(store.patches,store.dim,store.query_dim,compressed).cuda()
                log=train_codec(model,store,train,1200,pair_weight,20260923)
                model.eval()
                predict=lambda i,b:model(b["future"],b["queries"])
                result["arms"][name]={"train":evaluate(store,train,predict),
                    "val":evaluate(store,val,predict),"log":log}
                weights[name]=model.cpu().state_dict()
                del model
    result["full_ablation_skipped"]=not tiny_pass
    result["seconds"]=time.monotonic()-start
    (args.output/"result.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    torch.save(weights,args.output/"diagnostic_checkpoints.pt")
    lines=["# Observed-future codec diagnosis", "",
        "NOT a learned-world-model or planning result. One seed; development prefixes only.","",
        f"Tiny-fit gate: {tiny_pass}. Full ablation skipped: {not tiny_pass}.","",
        "Frame-metric probe has extra dense RGB-derived supervision and exact temporal "
        "aggregation. It diagnoses recoverability, not a matched method improvement.","",
        "| Arm | H | Train ordered MSE | Val ordered MSE | Val order regret |",
        "|---|---:|---:|---:|---:|"]
    for name,arm in {"frame_metric":probe,**result["arms"]}.items():
        for h,metrics in arm["val"]["summary"].items():
            fit=arm["train"]["summary"].get(h)
            fit_text="—" if fit is None else f"{fit['mse'][6]:.6f}"
            lines.append(f"| {name} | {h} | {fit_text} | {metrics['mse'][6]:.6f} | {metrics['regret']:.6f} |")
    report="\n".join(lines)+"\n"
    (args.output/"REPORT.md").write_text(report)
    print(report,flush=True)


if __name__=="__main__": main()
