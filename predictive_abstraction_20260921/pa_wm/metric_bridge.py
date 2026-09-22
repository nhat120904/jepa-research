"""Frozen RGB-affinity metric -> temporal codec -> action-conditioned WM pilot.

Shared exact temporal query readout for frame and compressed models. No privileged
labels. Decoding a summary still uses H lightweight latent tokens: no O(1) claim.
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
from .models import (positions, temporal_encoder, TokenPool, SpatialActionSummary,
                     DirectQueryPredictor)
from .codec_rescue import FrameMetric, aggregates, groups_for, evaluate
from .queries import chroma_features, similarity


class LatentDecoder(nn.Module):
    def __init__(self, dim=64, width=128):
        super().__init__()
        self.attn=nn.MultiheadAttention(width,4,batch_first=True)
        self.out=nn.Sequential(nn.LayerNorm(width),nn.Linear(width,dim))

    def forward(self,z,horizon):
        q=positions(horizon,z.shape[-1],z.device,z.dtype)[None].expand(len(z),-1,-1)
        x,_=self.attn(q,z,z,need_weights=False)
        return self.out(q+x)


class MetricCodec(nn.Module):
    def __init__(self,tokens,dim=64,width=128):
        super().__init__()
        self.project=nn.Linear(dim,width)
        self.encoder=temporal_encoder(width)
        self.pool=TokenPool(width,tokens)
        self.decoder=LatentDecoder(dim,width)

    def forward(self,future):
        x=self.project(future)
        x=x+positions(x.shape[1],x.shape[2],x.device,x.dtype)
        return self.pool(self.encoder(x))


class MetricForecast(nn.Module):
    def __init__(self,patches,dim,tokens=None,width=128):
        super().__init__()
        self.context=SpatialActionSummary(patches,dim,width=width,tokens=tokens or 4)
        self.tokens=tokens
        self.decoder=LatentDecoder(64,width) if tokens is None else None

    def forward(self,history,actions):
        if self.tokens is not None:
            return self.context(history,actions)
        context=self.context.context(history,actions)
        return self.decoder(context,actions.shape[1])


def fixed_compression(future,tokens):
    pooled=F.adaptive_avg_pool1d(future.transpose(1,2),tokens)
    return F.interpolate(pooled,size=future.shape[1],mode="linear",align_corners=False).transpose(1,2)


def metric_trace(normalized,anchors,mean,std):
    embedding=normalized*std+mean
    return torch.exp(-(embedding[:,:,None,:]-anchors[:,None,:,:]).square().mean(-1))


def prepare(store,features,metric):
    manifest=json.loads((features/"manifest.json").read_text())
    anchor_dim=store.patches*store.projection.shape[1]
    with torch.no_grad():
        for row in store.rows:
            frames=(row["future"].float()@store.projection).flatten(1).cuda()
            anchors=torch.stack([row["queries"][0,:anchor_dim],row["queries"][1,:anchor_dim]]).cuda()
            row["metric_future"]=metric.embed((frames-metric.mean)/metric.std).cpu()
            row["metric_anchors"]=metric.embed((anchors-metric.mean)/metric.std).cpu()
            with np.load(Path(manifest["dataset_root"])/row["source"]) as raw:
                f=chroma_features(raw["rgb"][1:]); a,b=chroma_features(raw["anchors"])
                row["trace"]=torch.from_numpy(np.stack([similarity(f,a),similarity(f,b)],-1))
            torch.testing.assert_close(aggregates(row["trace"][None])[0],row["answers"],atol=2e-6,rtol=2e-6)
    x=torch.cat([store.rows[i]["metric_future"] for i in store.train])
    mean,std=x.mean(0),x.std(0).clamp_min(.01)
    for row in store.rows:
        row["metric_future"]=(row["metric_future"]-mean)/std
    return mean.cuda(),std.cuda()


def extra_batch(store,indices):
    return {key:torch.stack([store.rows[i][key] for i in indices]).cuda()
            for key in ("metric_future","metric_anchors","trace")}


def query_vectors(anchors):
    a,b=anchors.unbind(1)
    ops=torch.eye(4,device=a.device)
    specs=[(a,a,0),(b,b,0),(a,a,1),(b,b,1),(a,a,2),(b,b,2),(a,b,3),(b,a,3)]
    return torch.stack([torch.cat([x,y,ops[op].expand(len(x),-1)],-1) for x,y,op in specs],1)


def optimize(name,model,store,groups,steps,objective):
    model.train()
    rng=np.random.default_rng(20260924)
    optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=.01)
    keys=list(groups)
    for step in range(steps):
        h=int(rng.choice([32,48])); pool=[k for k in keys if k[1]==h]
        selected=rng.choice(len(pool),2,replace=False)
        indices=[i for j in selected for i in groups[pool[j]]]
        b=store.batch(indices,"cuda"); e=extra_batch(store,indices)
        loss=objective(b,e)
        if not torch.isfinite(loss): raise FloatingPointError(name)
        optimizer.zero_grad(set_to_none=True); loss.backward()
        norm=nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        if step==0 and norm.item()==0: raise AssertionError("No gradient: "+name)
        optimizer.step()
        if (step+1)%max(1,steps//4)==0: print(f"{name} {step+1}/{steps}: {loss.item():.6f}",flush=True)
    model.eval()


def main():
    require_slurm()
    parser=argparse.ArgumentParser()
    parser.add_argument("--features",type=Path,required=True)
    parser.add_argument("--metric-checkpoint",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4)
    if not torch.cuda.is_available(): raise RuntimeError("GPU sbatch required")
    store=BranchStore(args.features)
    state=torch.load(args.metric_checkpoint,map_location="cpu",weights_only=True)["frame_metric"]
    metric=FrameMetric(len(state["mean"]),state["mean"],state["std"])
    metric.load_state_dict(state); metric=metric.eval().requires_grad_(False).cuda()
    mean,std=prepare(store,args.features,metric)
    train,val=groups_for(store,"train"),groups_for(store,"val")
    results={"status":"DEVELOPMENT_METRIC_BRIDGE_ONE_SEED","test_read":False,
             "physical_success_evaluated":False,"arms":{},"codec_gates":{},
             "seed":20260924,"metric_checkpoint":str(args.metric_checkpoint)}
    weights={"metric":state,"projection":store.projection,"mean":mean.cpu(),"std":std.cpu()}
    def loss_reconstruction(pred,b,e):
        trace=metric_trace(pred,e["metric_anchors"],mean,std)
        return (F.mse_loss(pred,e["metric_future"])+F.mse_loss(trace,e["trace"])
                +.1*F.mse_loss(aggregates(trace),b["answers"]))
    def record(name,predict):
        results["arms"][name]={"train":evaluate(store,train,predict),"val":evaluate(store,val,predict)}
    def answers(latent,e):
        return aggregates(metric_trace(latent,e["metric_anchors"],mean,std))
    record("observed_metric",lambda i,b:answers(extra_batch(store,i)["metric_future"],extra_batch(store,i)))
    previous=json.loads((args.metric_checkpoint.parent/"result.json").read_text())
    for h,m in results["arms"]["observed_metric"]["val"]["summary"].items():
        expected=previous["frame_metric"]["val"]["summary"][str(h)]
        if (np.max(np.abs(np.asarray(m["mse"])-expected["mse"]))>1e-4 or
                abs(m["regret"]-expected["regret"])>1e-4):
            raise AssertionError("Frozen frame-metric reference did not reproduce 53659")
    results["reference_reproduced"]=True
    codecs={}
    for tokens in (4,16):
        record(f"fixed_pool_{tokens}",lambda i,b,k=tokens:answers(
            fixed_compression(extra_batch(store,i)["metric_future"],k),extra_batch(store,i)))
        torch.manual_seed(20260924)
        codec=MetricCodec(tokens).cuda()
        optimize(f"codec_{tokens}",codec,store,train,1600,
                 lambda b,e:loss_reconstruction(codec.decoder(codec(e["metric_future"]),b["actions"].shape[1]),b,e))
        name=f"observed_summary_{tokens}"
        record(name,lambda i,b:answers(codec.decoder(codec(extra_batch(store,i)["metric_future"]),
                                                    b["actions"].shape[1]),extra_batch(store,i)))
        # Predeclared DEVELOPMENT gate only; not confirmation or untouched-test selection.
        refs=results["arms"]["observed_metric"]["val"]["summary"]
        checks={h:(m["mse"][6]<=max(.01,2*refs[h]["mse"][6]) and
                   m["regret"]<=refs[h]["regret"]+.01)
                for h,m in results["arms"][name]["val"]["summary"].items()}
        results["codec_gates"][str(tokens)]={"by_horizon":checks,"pass":all(checks.values())}
        codec.eval().requires_grad_(False)
        codecs[tokens]=codec
        weights[name]=codec.cpu().state_dict()
        codec.cuda()
    # Strong reference and separately trained action ablation always run, so inability
    # to compress cannot be confused with inability to predict the metric trajectory.
    modes=[("frame",None,False),("frame_no_action",None,True)]
    modes += [(f"summary_{k}",k,False) for k in (4,16) if results["codec_gates"][str(k)]["pass"]]
    for name,tokens,no_action in modes:
        torch.manual_seed(20260924)
        model=MetricForecast(store.patches,store.dim,tokens).cuda()
        def predict_latent(b):
            actions=torch.zeros_like(b["actions"]) if no_action else b["actions"]
            z=model(b["history"],actions)
            return z if tokens is None else codecs[tokens].decoder(z,b["actions"].shape[1])
        def objective(b,e):
            value=loss_reconstruction(predict_latent(b),b,e)
            if tokens is not None:
                with torch.no_grad(): target=codecs[tokens](e["metric_future"])
                value=value+.1*F.mse_loss(model(b["history"],b["actions"]),target)
            return value
        optimize(name,model,store,train,1800,objective)
        record(name,lambda i,b:answers(predict_latent(b),extra_batch(store,i)))
        if not no_action:
            def shuffle(i,b):
                shuffled={**b,"actions":b["actions"].roll(1,0)}
                return answers(predict_latent(shuffled),extra_batch(store,i))
            record(name+"_shuffled",shuffle)
        weights[name]=model.cpu().state_dict()
        del model
    torch.manual_seed(20260924)
    direct=DirectQueryPredictor(132,store.patches,store.dim,128).cuda()
    optimize("direct",direct,store,train,1800,lambda b,e:F.mse_loss(
        direct(b["history"],b["actions"],query_vectors(e["metric_anchors"])),b["answers"]))
    record("direct",lambda i,b:direct(b["history"],b["actions"],query_vectors(extra_batch(store,i)["metric_anchors"])))
    weights["direct"]=direct.cpu().state_dict()
    results["limitations"]=["fixed temporal query algebra", "summary decoded to H lightweight latent tokens",
        "metric trained with dense RGB-derived affinities", "12 reused development validation prefixes",
        "summary forecasts gated on development codec retention", "one seed; no novelty or control claim"]
    (args.output/"result.json").write_text(json.dumps(results,indent=2,allow_nan=False)+"\n")
    torch.save(weights,args.output/"checkpoints.pt")
    lines=["# Metric-space trajectory pilot","",
        "Observed arms read true future; frame/summary/direct arms forecast from history + actions.",
        "Development only; no physical success or closed-loop control. Same small bank headroom.","",
        f"Codec gates: {results['codec_gates']}","",
        "| Arm | H | Train order MSE | Val order MSE | Val order regret |",
        "|---|---:|---:|---:|---:|"]
    for name,arm in results["arms"].items():
        for h,m in arm["val"]["summary"].items():
            fit=arm["train"]["summary"].get(h)
            s="—" if fit is None else f"{fit['mse'][6]:.6f}"
            lines.append(f"| {name} | {h} | {s} | {m['mse'][6]:.6f} | {m['regret']:.6f} |")
    report="\n".join(lines)+"\n"
    (args.output/"REPORT.md").write_text(report); print(report,flush=True)


if __name__=="__main__": main()
