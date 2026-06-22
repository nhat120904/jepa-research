"""Convert the HF `facebook/dinov3-vitl16-pretrain-lvd1689m` safetensors to the
original `facebookresearch/dinov3` repo state_dict format that jepa-wms's
`DinoEncoder` loads via `torch.hub.load(..., source="local", weights=<.pth>)`.

Why: the original `.pth` lives on dl.fbaipublicfiles.com (firewalled on this
cluster); only the HF safetensors mirror is reachable. The mapping is mechanical
(no guessing): HF splits q/k/v + uses `layer.*`/`embeddings.*` names, the original
fuses qkv + uses `blocks.*`/`patch_embed.*`. Buffers (rope periods, qkv bias mask)
are deterministic from config, so we keep them from a fresh build.

Validation gate: strict-load into the fresh model must leave ONLY buffers missing
and nothing unexpected; a forward pass must be finite. The end-to-end check is the
terver gripper sanity gate on jepa_wm_droid (the WM head was trained on these exact
features, so any mis-fusion shows up as a failed gate / degenerate CRA).
"""
import sys, torch
from safetensors.torch import load_file

HF_SAFETENSORS = "/mnt/data/nhatnc129/jepa/ossckpt/dinov3/model.safetensors"
OUT_PATH = "/mnt/data/nhatnc129/jepa/ossckpt/dinov3/dinov3_vitl16_pretrain_lvd1689m-7c1da9a5.pth"
DINOV3_REPO = "/home/nhatnc129/dinov3"

sys.path.insert(0, DINOV3_REPO)
from dinov3.hub.backbones import dinov3_vitl16  # noqa: E402

hf = load_file(HF_SAFETENSORS)
model = dinov3_vitl16(pretrained=False)
osd = model.state_dict()
depth = sum(1 for k in osd if k.endswith(".attn.qkv.weight"))
print(f"original depth (blocks) = {depth}; orig tensors = {len(osd)}; hf tensors = {len(hf)}")

new = {
    "cls_token": hf["embeddings.cls_token"],
    "mask_token": hf["embeddings.mask_token"].reshape(osd["mask_token"].shape),
    "storage_tokens": hf["embeddings.register_tokens"],
    "patch_embed.proj.weight": hf["embeddings.patch_embeddings.weight"],
    "patch_embed.proj.bias": hf["embeddings.patch_embeddings.bias"],
    "norm.weight": hf["norm.weight"],
    "norm.bias": hf["norm.bias"],
}
for i in range(depth):
    h, o = f"layer.{i}.", f"blocks.{i}."
    new[o + "norm1.weight"] = hf[h + "norm1.weight"]
    new[o + "norm1.bias"] = hf[h + "norm1.bias"]
    new[o + "norm2.weight"] = hf[h + "norm2.weight"]
    new[o + "norm2.bias"] = hf[h + "norm2.bias"]
    # qkv: HF stores q/k/v separately; original fuses [q;k;v]. DINOv3 masks the
    # k bias (q_proj.bias + v_proj.bias present, k has none) -> insert zeros for k.
    qw, kw, vw = (hf[h + f"attention.{p}_proj.weight"] for p in ("q", "k", "v"))
    new[o + "attn.qkv.weight"] = torch.cat([qw, kw, vw], dim=0)
    qb, vb = hf[h + "attention.q_proj.bias"], hf[h + "attention.v_proj.bias"]
    new[o + "attn.qkv.bias"] = torch.cat([qb, torch.zeros_like(qb), vb], dim=0)
    new[o + "attn.proj.weight"] = hf[h + "attention.o_proj.weight"]
    new[o + "attn.proj.bias"] = hf[h + "attention.o_proj.bias"]
    new[o + "ls1.gamma"] = hf[h + "layer_scale1.lambda1"]
    new[o + "ls2.gamma"] = hf[h + "layer_scale2.lambda1"]
    new[o + "mlp.fc1.weight"] = hf[h + "mlp.up_proj.weight"]
    new[o + "mlp.fc1.bias"] = hf[h + "mlp.up_proj.bias"]
    new[o + "mlp.fc2.weight"] = hf[h + "mlp.down_proj.weight"]
    new[o + "mlp.fc2.bias"] = hf[h + "mlp.down_proj.bias"]

missing, unexpected = model.load_state_dict(new, strict=False)
buffers = {n for n, _ in model.named_buffers()}
missing_nonbuf = [k for k in missing if k not in buffers]
print(f"load: {len(missing)} missing ({len(missing_nonbuf)} non-buffer), {len(unexpected)} unexpected")
assert not unexpected, f"UNEXPECTED keys (mapping wrong): {unexpected[:8]}"
assert not missing_nonbuf, f"MISSING non-buffer keys (mapping incomplete): {missing_nonbuf[:8]}"
print(f"  (missing are buffers only: {sorted(set(missing) & buffers)[:3]}... ok)")

model.eval()
with torch.no_grad():
    feats = model.forward_features(torch.randn(1, 3, 256, 256))
pt = feats["x_norm_patchtokens"]
assert torch.isfinite(pt).all(), "non-finite features!"
print(f"forward OK: patchtokens {tuple(pt.shape)} mean={pt.mean():.4f} std={pt.std():.4f} "
      f"cls finite={torch.isfinite(feats['x_norm_clstoken']).all().item()}")

torch.save(model.state_dict(), OUT_PATH)
print(f"WROTE {OUT_PATH} ({len(model.state_dict())} tensors)")
print("CONVERT_OK")
