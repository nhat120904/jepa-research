"""Forecast a validated fixed 16-bin trajectory target, without a learned codec.

Common phase/duration decoder for compact and full-frame arms; matched coarse-loss
control separates output length from supervision changes. All inputs are past+actions.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from .runtime import require_slurm
from .branch_data import BranchStore
from .models import SpatialActionSummary
from .codec_rescue import FrameMetric, aggregates, groups_for, evaluate
from .metric_bridge import prepare, extra_batch, metric_trace, fixed_compression


def bins(x,count=16):
    return F.adaptive_avg_pool1d(x.transpose(1,2),count).transpose(1,2)


def expand_bins(x,horizon):
    return F.interpolate(x.transpose(1,2),size=horizon,mode="linear",align_corners=False).transpose(1,2)


def time_queries(count,horizon,device,dtype):
    # Expected native sample time of an equal-duration bin of observations 1..H.
    phase=(torch.arange(count,device=device,dtype=dtype)+.5)/count+.5/horizon
    duration=torch.full_like(phase,horizon/48.)
    return torch.stack([phase,phase.square(),duration,phase*duration],-1)


class AlignedForecast(nn.Module):
    def __init__(self,patches,dim,compact=False,width=128):
        super().__init__()
        self.compact=compact
        self.context=SpatialActionSummary(patches,dim,width=width)
        self.time=nn.Sequential(nn.Linear(4,width),nn.GELU(),nn.Linear(width,width))
        self.attn=nn.MultiheadAttention(width,4,batch_first=True)
        self.out=nn.Sequential(nn.LayerNorm(width),nn.Linear(width,64))

    def forward(self,history,actions):
        h=actions.shape[1]; count=16 if self.compact else h
        context=self.context.context(history,actions)
        q=self.time(time_queries(count,h,context.device,context.dtype))[None].expand(len(history),-1,-1)
        x,_=self.attn(q,context,context,need_weights=False)
        slots=self.out(q+x)
        full=expand_bins(slots,h) if self.compact else slots
        return full,slots


def main():
    require_slurm()
    parser=argparse.ArgumentParser()
    parser.add_argument("--features",type=Path,required=True)
    parser.add_argument("--metric-checkpoint",type=Path,required=True)
    parser.add_argument("--bridge-result",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4)
    if not torch.cuda.is_available(): raise RuntimeError("GPU job required")
    store=BranchStore(args.features)
    state=torch.load(args.metric_checkpoint,map_location="cpu",weights_only=True)["frame_metric"]
    metric=FrameMetric(len(state["mean"]),state["mean"],state["std"])
    metric.load_state_dict(state); metric=metric.eval().requires_grad_(False).cuda()
    mean,std=prepare(store,args.features,metric)
    train,val=groups_for(store,"train"),groups_for(store,"val")
    result={"status":"FIXED_SUMMARY_FORECAST_DEVELOPMENT","seed":20260925,
            "test_read":False,"control_evaluated":False,"tiny":{},"arms":{}}
    weights={"metric":state,"projection":store.projection,"mean":mean.cpu(),"std":std.cpu()}
    def readout(full,e): return aggregates(metric_trace(full,e["metric_anchors"],mean,std))
    result["reference"]=evaluate(store,val,lambda i,b:readout(
        fixed_compression(extra_batch(store,i)["metric_future"],16),extra_batch(store,i)))
    prior=json.loads(args.bridge_result.read_text())["arms"]["fixed_pool_16"]["val"]["summary"]
    for h,m in result["reference"]["summary"].items():
        if (np.max(np.abs(np.asarray(m["mse"])-prior[str(h)]["mse"]))>1e-4 or
                abs(m["regret"]-prior[str(h)]["regret"])>1e-4):
            raise AssertionError("Fixed-bin reference did not reproduce 53674")
    result["reference_reproduced"]=True

    def objective(model,b,e,coarse,no_action):
        actions=torch.zeros_like(b["actions"]) if no_action else b["actions"]
        full,slots=model(b["history"],actions)
        pred=bins(full) if not model.compact and coarse else slots
        target=bins(e["metric_future"]) if coarse else e["metric_future"]
        trace=metric_trace(full,e["metric_anchors"],mean,std)
        return (F.mse_loss(pred,target)+F.mse_loss(trace,e["trace"])
                +.1*F.mse_loss(aggregates(trace),b["answers"]))

    def train_model(name,model,groups,steps,coarse,no_action=False,tiny=False):
        rng=np.random.default_rng(20260925)
        opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=.01)
        keys=list(groups)
        for step in range(steps):
            horizon=keys[0][1] if tiny else int(rng.choice([32,48]))
            pool=[k for k in keys if k[1]==horizon]
            selected=rng.choice(len(pool),1 if tiny else 2,replace=False)
            ids=[i for j in selected for i in groups[pool[j]]]
            b=store.batch(ids,"cuda"); e=extra_batch(store,ids)
            model.train(); loss=objective(model,b,e,coarse,no_action)
            if not torch.isfinite(loss): raise FloatingPointError(name)
            opt.zero_grad(set_to_none=True); loss.backward()
            norm=nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            if step==0 and norm.item()==0: raise AssertionError("No gradient: "+name)
            opt.step()
            if (step+1)%max(1,steps//4)==0: print(f"{name} {step+1}/{steps} {loss.item():.6f}",flush=True)
        model.eval()

    def assess(model,groups,no_action=False,shuffle=False):
        def predict(ids,b):
            actions=torch.zeros_like(b["actions"]) if no_action else b["actions"]
            if shuffle: actions=actions.roll(1,0)
            return readout(model(b["history"],actions)[0],extra_batch(store,ids))
        assessment=evaluate(store,groups,predict)
        errors={}
        with torch.inference_mode():
            for (_,h),ids in groups.items():
                b=store.batch(ids,"cuda"); e=extra_batch(store,ids)
                actions=torch.zeros_like(b["actions"]) if no_action else b["actions"]
                if shuffle: actions=actions.roll(1,0)
                full,slots=model(b["history"],actions)
                predicted_bins=slots if model.compact else bins(full)
                values=[float(F.mse_loss(predicted_bins,bins(e["metric_future"]))),
                        float(F.mse_loss(full,e["metric_future"]))]
                errors.setdefault(h,[]).append(values)
        assessment["latent_mse"]={h:dict(zip(["bins","full"],np.mean(v,axis=0).tolist()))
                                  for h,v in errors.items()}
        return assessment

    def train_variance(k):
        return float(torch.stack([store.rows[i]["answers"] for i in train[k]])[:,6].var(unbiased=False))
    tiny_key=max([k for k in train if k[1]==32],key=train_variance)
    result["tiny_train_prefix"]=tiny_key
    tiny_pass=False
    for compact in (False,True):
        name="compact16" if compact else "frame_bin_loss"
        torch.manual_seed(20260925)
        model=AlignedForecast(store.patches,store.dim,compact).cuda()
        train_model("tiny_"+name,model,{tiny_key:train[tiny_key]},1000,True,tiny=True)
        fit=assess(model,{tiny_key:train[tiny_key]})
        passed=fit["summary"][32]["mse"][6]<.01 and fit["latent_mse"][32]["bins"]<.05
        tiny_pass|=passed
        result["tiny"][name]={"pass":passed,"fit":fit}
        del model
    result["full_stage_skipped"]=not tiny_pass
    if tiny_pass:
        arms=[("frame_dense_loss",False,False,False),
              ("frame_bin_loss",False,True,False),
              ("compact16",True,True,False),
              ("compact16_no_action",True,True,True)]
        for name,compact,coarse,no_action in arms:
            torch.manual_seed(20260925)
            model=AlignedForecast(store.patches,store.dim,compact).cuda()
            train_model(name,model,train,1800,coarse,no_action)
            result["arms"][name]={"train":assess(model,train,no_action),"val":assess(model,val,no_action),
                                 "parameters":sum(p.numel() for p in model.parameters())}
            if not no_action:
                result["arms"][name+"_shuffled"]={"train":assess(model,train,shuffle=True),
                                                   "val":assess(model,val,shuffle=True)}
            weights[name]=model.cpu().state_dict()
            del model
    result["limitations"]=["one seed on reused 12-prefix validation", "known temporal query algebra",
        "RGB-affinity metric target", "fixed bins are not a learned-summary novelty claim",
        "all new arms share phase/duration decoder; do not attribute gains over 53674 to bins alone"]
    (args.output/"result.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    torch.save(weights,args.output/"checkpoints.pt")
    lines=["# Fixed-summary forecast feasibility","",
        "Development diagnostic. No physical success or control result.","",
        f"Tiny gates: { {k:v['pass'] for k,v in result['tiny'].items()} }. Full stage skipped: {not tiny_pass}.","",
        "| Arm | H | Train order MSE | Val order MSE | Val bin MSE | Val regret |",
        "|---|---:|---:|---:|---:|---:|"]
    for name,arm in result["arms"].items():
        for h,m in arm["val"]["summary"].items():
            fit=arm["train"]["summary"].get(h)
            v="—" if fit is None else f"{fit['mse'][6]:.6f}"
            lines.append(f"| {name} | {h} | {v} | {m['mse'][6]:.6f} | "
                         f"{arm['val']['latent_mse'][h]['bins']:.6f} | {m['regret']:.6f} |")
    report="\n".join(lines)+"\n"
    (args.output/"REPORT.md").write_text(report); print(report,flush=True)


if __name__=="__main__": main()
