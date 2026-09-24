"""Freeze 53698 predictors; isolate event errors and train-only readout adaptation.

Readout holdout prefixes were seen by the OLD predictor, not by the new reader.
Validation prefixes were seen by neither's optimization, but are reused development.
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
from .fixed_summary_forecast import AlignedForecast
from .codec_rescue import FrameMetric, aggregates, groups_for
from .metric_bridge import prepare, extra_batch, metric_trace


class PairReader(nn.Module):
    def __init__(self,dim=64):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(4*dim,128),nn.GELU(),nn.Linear(128,64),
                               nn.GELU(),nn.Linear(64,1),nn.Sigmoid())

    def forward(self,frame,anchor):
        frame,anchor=torch.broadcast_tensors(frame,anchor)
        return self.net(torch.cat([frame,anchor,frame-anchor,frame*anchor],-1)).squeeze(-1)


def selection(scores,truth,eps=1e-8):
    scores=np.asarray(scores,float); truth=np.asarray(truth,float)
    tied=np.abs(scores-scores.max())<=eps
    difference=truth[:,None]-truth[None,:]
    predicted=scores[:,None]-scores[None,:]
    mask=np.triu(np.abs(difference)>.05,1)
    credit=np.where(np.abs(predicted)<=eps,.5,(predicted*difference>0).astype(float))
    return {"regret_first":float(truth.max()-truth[scores.argmax()]),
            "regret_uniform_ties":float(truth.max()-truth[tied].mean()),
            "ties":int(tied.sum()),"score_spread":float(np.ptp(scores)),
            "informative_pairs":int(mask.sum()),
            "pair_accuracy":float(credit[mask].mean()) if mask.any() else None}


def event_diagnostic(predicted,truth):
    predicted=np.asarray(predicted); truth=np.asarray(truth)
    peak=predicted.max(1); actual=truth.max(1)
    positive=actual>=.5; detected=peak>=.5; both=positive&detected
    timing=np.abs(predicted.argmax(1)-truth.argmax(1))
    return {"positive_events":int(positive.sum()),
            "missed_below_0_1":int((positive&(peak<.1)).sum()),
            "detected_above_0_5":int(both.sum()),
            "negative_events":int((actual<=.05).sum()),
            "false_positive_above_0_5":int(((actual<=.05)&detected).sum()),
            "peak_timing_error_sum":float(timing[both].sum()),
            "positive_predicted_peak_sum":float(peak[positive].sum())}


def main():
    require_slurm()
    parser=argparse.ArgumentParser()
    parser.add_argument("--features",type=Path,required=True)
    parser.add_argument("--checkpoint",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4)
    if not torch.cuda.is_available(): raise RuntimeError("GPU job required")
    store=BranchStore(args.features)
    checkpoint=torch.load(args.checkpoint,map_location="cpu",weights_only=True)
    torch.testing.assert_close(store.projection,checkpoint["projection"])
    state=checkpoint["metric"]
    metric=FrameMetric(len(state["mean"]),state["mean"],state["std"])
    metric.load_state_dict(state); metric=metric.eval().requires_grad_(False).cuda()
    mean,std=prepare(store,args.features,metric)
    torch.testing.assert_close(mean.cpu(),checkpoint["mean"])
    torch.testing.assert_close(std.cpu(),checkpoint["std"])
    train,val=groups_for(store,"train"),groups_for(store,"val")
    fit={k:v for k,v in train.items() if int(k[0].split("_")[-1])%4!=3}
    hold={k:v for k,v in train.items() if int(k[0].split("_")[-1])%4==3}
    assert len({k[0] for k in fit})==18 and len({k[0] for k in hold})==6
    caches={"observed":{i:r["metric_future"] for i,r in enumerate(store.rows)}}
    sources=["compact16","frame_bin_loss","compact16_no_action"]
    with torch.no_grad():
        for name in sources:
            model=AlignedForecast(store.patches,store.dim,compact=name!="frame_bin_loss").cuda()
            model.load_state_dict(checkpoint[name]); model.eval().requires_grad_(False)
            cache={}
            for indices in {**train,**val}.values():
                b=store.batch(indices,"cuda")
                actions=torch.zeros_like(b["actions"]) if name.endswith("no_action") else b["actions"]
                predicted=model(b["history"],actions)[0].cpu()
                for j,i in enumerate(indices): cache[i]=predicted[j].clone()
            caches[name]=cache
            del model
    ids=[i for indices in fit.values() for i in indices]
    # Identical dense-pair sampling across readers; strata computed from FIT labels only.
    labels=torch.cat([store.rows[i]["trace"].flatten() for i in ids]).cuda()
    anchor_bank=torch.cat([((store.rows[i]["metric_anchors"].cuda()-mean)/std)[None]
                           .expand(len(caches["observed"][i]),-1,-1).reshape(-1,64)
                           for i in ids])
    positive=torch.where(labels>.05)[0]; negative=torch.where(labels<=.05)[0]
    if not len(positive) or not len(negative): raise AssertionError("Missing training stratum")
    readers={}; losses={}
    for name in ["observed"]+sources:
        torch.manual_seed(20260926)
        reader=PairReader().cuda()
        x=torch.cat([caches[name][i][:,None,:].expand(-1,2,-1).reshape(-1,64) for i in ids]).cuda()
        opt=torch.optim.AdamW(reader.parameters(),lr=3e-4,weight_decay=.01)
        generator=torch.Generator(device="cuda").manual_seed(20260926)
        for step in range(1200):
            sample=torch.cat([pool[torch.randint(len(pool),(128,),device="cuda",generator=generator)]
                              for pool in (positive,negative)])
            predicted=reader(x[sample],anchor_bank[sample])
            loss=F.mse_loss(predicted,labels[sample])
            if not torch.isfinite(loss): raise FloatingPointError(name)
            opt.zero_grad(set_to_none=True); loss.backward()
            norm=nn.utils.clip_grad_norm_(reader.parameters(),1.,error_if_nonfinite=True)
            if step==0 and norm.item()==0: raise AssertionError("Readout has no gradient")
            opt.step()
            if (step+1)%300==0: print(f"reader {name} {step+1}/1200 {loss.item():.6f}",flush=True)
        readers[name]=reader.eval(); losses[name]=float(loss.detach())
        del x
    # Kernel and adapted-reader comparisons share frozen forecasts and the exact same queries.
    arms={"zero_scores":("observed","zero"),"observed_kernel":("observed","kernel"),
          "observed_clean_reader":("observed","observed")}
    for name in sources:
        arms[name+"_kernel"]=(name,"kernel")
        arms[name+"_clean_reader"]=(name,"observed")
        arms[name+"_adapted_reader"]=(name,name)
    result={"status":"FROZEN_PREDICTOR_READOUT_DIAGNOSTIC","predictors_retrained":False,
        "test_read":False,"fit_prefixes":18,"readout_holdout_prefixes":6,"val_prefixes":12,
        "thresholds":{"event":.5,"miss":.1,"negative":.05,"ties":1e-8},
        "final_readout_fit_losses":losses,"arms":{}}
    with torch.inference_mode():
        for arm,(source,reader_name) in arms.items():
            result["arms"][arm]={}
            for split,groups in (("fit",fit),("readout_holdout",hold),("val",val)):
                rows=[]
                for (prefix,h),indices in groups.items():
                    x=torch.stack([caches[source][i] for i in indices]).cuda()
                    e=extra_batch(store,indices)
                    if reader_name=="kernel": trace=metric_trace(x,e["metric_anchors"],mean,std)
                    elif reader_name=="zero": trace=torch.zeros_like(e["trace"])
                    else:
                        anchor=(e["metric_anchors"]-mean)/std
                        trace=readers[reader_name](x[:,:,None,:],anchor[:,None,:,:])
                    predicted=aggregates(trace).cpu().numpy()
                    target=torch.stack([store.rows[i]["answers"] for i in indices]).numpy()
                    diag=event_diagnostic(trace.cpu().numpy(),e["trace"].cpu().numpy())
                    ordered_positive=target[:,6]>=.5
                    diag["positive_ordered_branches"]=int(ordered_positive.sum())
                    diag["missed_ordered_below_0_1"]=int((ordered_positive&(predicted[:,6]<.1)).sum())
                    # Count reversed order only where the true reverse direction is absent.
                    diag["reverse_only_errors"]=int(((target[:,6]>=.5)&(target[:,7]<=.05)&
                        (predicted[:,7]>predicted[:,6]+.05)).sum())
                    rows.append({"prefix":prefix,"horizon":h,"events":diag,
                        "mse":((predicted-target)**2).mean(0).tolist(),
                        **selection(predicted[:,6],target[:,6]),
                        "default_regret":float(target[:,6].max()-target[0,6])})
                summary={}
                for h in sorted({r["horizon"] for r in rows}):
                    sub=[r for r in rows if r["horizon"]==h]
                    events={k:sum(r["events"][k] for r in sub) for k in sub[0]["events"]}
                    acc=[r["pair_accuracy"] for r in sub if r["pair_accuracy"] is not None]
                    summary[h]={"mse":np.mean([r["mse"] for r in sub],axis=0).tolist(),
                        "regret_first":float(np.mean([r["regret_first"] for r in sub])),
                        "regret_uniform_ties":float(np.mean([r["regret_uniform_ties"] for r in sub])),
                        "pair_accuracy_macro":float(np.mean(acc)) if acc else None,
                        "events":events,"prefixes":len(sub)}
                result["arms"][arm][split]={"summary":summary,"groups":rows}
    prior=json.loads((args.checkpoint.parent/"result.json").read_text())
    for name in sources:
        old=prior["arms"][name]["val"]["summary"]
        for h,m in result["arms"][name+"_kernel"]["val"]["summary"].items():
            if (np.max(np.abs(np.asarray(m["mse"])-old[str(h)]["mse"]))>1e-4 or
                    abs(m["regret_first"]-old[str(h)]["regret"])>1e-4):
                raise AssertionError("Frozen forecast/kernel did not reproduce 53698")
    result["reference_reproduced"]=True
    contrasts={}
    for name in ("compact16","frame_bin_loss"):
        contrasts[name]={}
        for h in (32,48,64):
            before=[r for r in result["arms"][name+"_kernel"]["val"]["groups"] if r["horizon"]==h]
            after=[r for r in result["arms"][name+"_adapted_reader"]["val"]["groups"] if r["horizon"]==h]
            assert [r["prefix"] for r in before]==[r["prefix"] for r in after]
            delta=np.asarray([a["regret_uniform_ties"]-b["regret_uniform_ties"] for a,b in zip(before,after)])
            rng=np.random.default_rng(26)
            ci=np.quantile(rng.choice(delta,(2000,len(delta)),replace=True).mean(1),[.025,.975])
            contrasts[name][h]={"uniform_tie_regret_reduction":float(delta.mean()),"prefix_ci":ci.tolist()}
    result["paired_val_contrasts"]=contrasts
    result["limitations"]=["reused development validation", "predictor saw readout-holdout prefixes during original WM training",
        "fit forecasts are in-sample, not out-of-fold", "local MLP failure is not proof that no readout can work",
        "readout trained with dense RGB affinity labels", "diagnostic events are visual affinities, not physical success"]
    (args.output/"result.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    torch.save({k:v.cpu().state_dict() for k,v in readers.items()},args.output/"readers.pt")
    lines=["# Frozen-predictor readout diagnostic","",
        "World models unchanged. Reader fit: 18 old train prefixes; reader holdout: 6 old train prefixes; val: 12 development prefixes.",
        "No physical success/planning claim. Uniform ties are an analytic expectation, not a new rollout.","",
        "| Arm | H | Fit MSE | Reader-holdout MSE | Val MSE | Val regret first / uniform ties | Missed positive events |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for name,arm in result["arms"].items():
        for h,m in arm["val"]["summary"].items():
            fit=arm["fit"]["summary"].get(h); holdout=arm["readout_holdout"]["summary"].get(h)
            f="—" if fit is None else f"{fit['mse'][6]:.6f}"
            o="—" if holdout is None else f"{holdout['mse'][6]:.6f}"
            e=m["events"]
            lines.append(f"| {name} | {h} | {f} | {o} | {m['mse'][6]:.6f} | "
                         f"{m['regret_first']:.6f} / {m['regret_uniform_ties']:.6f} | "
                         f"{e['missed_below_0_1']}/{e['positive_events']} |")
    lines += ["", "Paired uniform-tie regret changes (positive favors adapted reader):", "",json.dumps(contrasts,indent=2)]
    report="\n".join(lines)+"\n"
    (args.output/"REPORT.md").write_text(report); print(report,flush=True)


if __name__=="__main__": main()
