"""Read-only video instrumentation around the existing unified closed loop.

Runs on a Slurm CPU worker. Replays three explicitly selected development roots,
with the reference checkpoint and planner settings. CPU numerics may differ from
the reference GPU evaluation; this demo is not a new benchmark result.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def main():
    if "SLURM_JOB_ID" not in os.environ:
        raise RuntimeError("Submit through sbatch; no models or physics on login")
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    opt = ap.parse_args()
    opt.out.mkdir(parents=True, exist_ok=True)
    import gymnasium
    import u_closed_loop as loop

    try:
        import imageio.v2 as imageio
        import imageio_ffmpeg
        print("FFMPEG", imageio_ffmpeg.get_ffmpeg_exe(), flush=True)
    except ImportError:
        imageio = None
        print("MP4 encoder unavailable; export animated GIF instead", flush=True)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 15)
    except OSError:
        font = ImageFont.load_default()

    reference = json.loads(opt.reference.read_text())
    args = reference["summary"]["args"]
    cfg = {k: args[k] for k in ("env",)}
    cfg.update({k: Path(args[k]) for k in ("model", "reader", "skill", "events", "cache")})
    cfg.update({k: int(args[k]) for k in ("m", "timeout", "exec_steps", "recover_steps", "max_expansions", "seed")})
    cfg.update(device="cpu", cube_skill=None, cube_discover=None)
    settings = SimpleNamespace(**cfg)
    outputs = []
    active = {}
    original_make, original_plan = gymnasium.make, loop.Model.plan

    def xyz(env):
        u = env.unwrapped
        return [u._data.joint(f"object_joint_{i}").qpos[:3].copy().tolist() for i in range(3)]

    def paint(ob, success=False, terminal=False):
        canvas = Image.new("RGB", (704, 448), (20, 24, 33))
        canvas.paste(Image.fromarray(np.asarray(ob).copy()).resize((320, 320), Image.Resampling.NEAREST), (16, 72))
        canvas.paste(Image.fromarray(active["goal"]).resize((320, 320), Image.Resampling.NEAREST), (368, 72))
        draw = ImageDraw.Draw(canvas)
        draw.text((16, 10), f"UNIFIED CUBE TRIPLE | task {active['task']} episode {active['episode']}", font=font, fill="white")
        draw.text((16, 43), "Robot observation", font=font, fill="#c9d4e5")
        draw.text((368, 43), "Goal image", font=font, fill="#c9d4e5")
        state = "SUCCESS" if success else ("EPISODE ENDED: FAILURE" if terminal else "RUNNING")
        draw.text((16, 399), f"{state} | step {active['step']} | planner calls {len(active['plans'])}", font=font,
                  fill="#74e6ac" if success else "#ffcf89")
        if active["plans"]:
            plan = active["plans"][-1]
            draw.text((16, 425), f"Full plan found: {plan['found']} | CPU replay, same checkpoint; diagnostic only", font=font, fill="#b5c2d5")
        return canvas

    def record(ob, info=None, terminal=False):
        success = bool((info or {}).get("success", False))
        if active["step"] % 4 == 0 or terminal:
            im = paint(ob, success, terminal)
            if active.get("writer") is not None:
                active["writer"].append_data(np.asarray(im))
            else:
                active["gif_frames"].append(im)
            if active["step"] == 0:
                im.save(opt.out / f"task{active['task']}_ep{active['episode']}_initial.png")
            if terminal:
                im.save(opt.out / f"task{active['task']}_ep{active['episode']}_final.png")
        active["trace"].append({"step": active["step"], "xyz_privileged_diagnostic": xyz(active["env"]),
                                "success": success})

    class CaptureEnv:
        def __init__(self, env):
            self.env = env

        def __getattr__(self, name):
            return getattr(self.env, name)

        def reset(self, **kw):
            ob, info = self.env.reset(**kw)
            active["env"] = self.env
            active["goal"] = np.asarray(info["goal"]).copy()
            active["goal_xyz_privileged_diagnostic"] = self.env.unwrapped._data.mocap_pos[self.env.unwrapped._cube_target_mocap_ids].copy().tolist()
            record(ob)
            return ob, info

        def step(self, action):
            result = self.env.step(action)
            ob, _, term, trunc, info = result
            active["step"] += 1
            record(ob, info, bool(term or trunc or info.get("success", False)))
            return result

    def captured_plan(model, s, g, **kw):
        plan, info = original_plan(model, s, g, **kw)
        chosen = plan if plan is not None else info.get("best_plan", [])
        active["plans"].append({"step": active["step"], "found": plan is not None,
                                "expanded": info.get("expanded"), "reader_state": np.asarray(s).tolist(),
                                "goal_reader_state": np.asarray(g).tolist(),
                                "events": [{"e": int(e), "target": np.asarray(x).tolist()} for e, x in chosen],
                                "xyz_privileged_diagnostic": xyz(active["env"])})
        return plan, info

    gymnasium.make = lambda *a, **k: CaptureEnv(original_make(*a, **k))
    loop.Model.plan = captured_plan
    for task, desired_success in ((1, True), (2, False), (5, False)):
        pool = [e for e in reference["episodes"] if e["task"] == task and e["success"] == desired_success]
        if not pool:
            raise RuntimeError(f"Reference has no requested root for task {task}")
        root = pool[0]
        epi = root["episode"]
        active.clear()
        active.update(task=task, episode=epi, step=0, plans=[], trace=[], gif_frames=[], writer=None)
        name = f"task{task}_ep{epi}"
        if imageio is not None:
            active["writer"] = imageio.get_writer(str(opt.out / f"{name}.mp4"), fps=5,
                                                  codec="libx264", macro_block_size=1,
                                                  ffmpeg_params=["-crf", "20", "-pix_fmt", "yuv420p"])
        try:
            episodes = loop.run(settings, [(task, epi)])
        finally:
            if active["writer"] is not None:
                active["writer"].close()
            if "env" in active:
                active["env"].close()
        if active["gif_frames"]:
            fs = active["gif_frames"]
            fs[0].save(opt.out / f"{name}.gif", save_all=True, append_images=fs[1:], duration=200, loop=0)
        data = {"reference_episode": root, "cpu_demo_episode": episodes[0],
                "planner_trace": active["plans"], "simulator_trace_diagnostic_only": active["trace"],
                "goal_xyz_diagnostic_only": active["goal_xyz_privileged_diagnostic"]}
        (opt.out / f"{name}.json").write_text(json.dumps(data, indent=1))
        outputs.append({"task": task, "episode": epi, "reference_success": root["success"],
                        "cpu_demo_success": episodes[0]["success"], "steps": episodes[0]["steps"],
                        "timeouts": episodes[0]["timeouts"], "replans": episodes[0]["replans"],
                        "video": f"{name}.mp4" if imageio is not None else f"{name}.gif"})
        print("DEMO", json.dumps(outputs[-1]), flush=True)
    hashes = {k: hashlib.sha256(Path(cfg[k]).read_bytes()).hexdigest() for k in ("model", "reader", "skill")}
    (opt.out / "manifest.json").write_text(json.dumps({"reference": str(opt.reference), "device": "cpu",
        "job_id": os.environ["SLURM_JOB_ID"], "checkpoint_sha256": hashes, "episodes": outputs,
        "note": "Selected development roots; CPU replay, not a benchmark. Privileged states are recorded only after env.step for diagnosis, never supplied to the policy."}, indent=2))


if __name__ == "__main__":
    main()
