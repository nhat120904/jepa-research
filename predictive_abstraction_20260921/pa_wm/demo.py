"""A professor-facing demonstrator built from actual simulator replay, not a learned WM."""
import argparse
import base64
import io
import json
from pathlib import Path

from .runtime import require_slurm


def main():
    require_slurm()
    import numpy as np
    import torch
    from PIL import Image, ImageDraw, ImageFont
    from .proposals import visual_goal_bank
    from .wall_adapter import WallRGB
    from .preflight import physical_answers

    torch.set_num_threads(1)
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--upstream", required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    root = Path("/mnt/data/nhatnc129/jepa/predictive_abstraction")
    original = [json.loads(x) for x in (root / "preflight_53529/prefixes.jsonl").read_text().splitlines()]
    review = json.loads((root / "review_default_53568/result.json").read_text())["rows"]
    choices = [
        ("Default fails; oracle finds a successful alternative", next(x for x in review if x["rescuable_over_default"])),
        ("Default already succeeds", next(x for x in review if x["default_a_then_b"])),
        ("No candidate in this bank succeeds", next(x for x in review if not x["oracle"])),
    ]
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    font = ImageFont.truetype(font_path, 16) if Path(font_path).exists() else ImageFont.load_default()
    small = ImageFont.truetype(font_path, 13) if Path(font_path).exists() else font
    cases, gif_frames = [], []
    poster = None
    for case_id, (title, entry) in enumerate(choices):
        row = original[entry["prefix_id"]]
        env = WallRGB(args.upstream, *row["layout"], seed=20260921)
        first = env.reset(row["start"])
        snapshot = env.snapshot()
        anchors = [env.goal_image(row["goal_a"]), env.goal_image(row["goal_b"])]
        bank = visual_goal_bank(np.random.default_rng(row["seed"] + 1), first,
                                *anchors, 128, 48, 7, 1.8)
        selected = row["tasks"]["a_then_b"]["image_selected_index"]
        trajectories = [env.rollout(snapshot, bank[index]) for index in (1, selected)]
        assert physical_answers(trajectories[0][1][1:], row["goal_a"], row["goal_b"], 4.5)["a_then_b"] == entry["default_a_then_b"]
        assert physical_answers(trajectories[1][1][1:], row["goal_a"], row["goal_b"], 4.5)["a_then_b"] == row["tasks"]["a_then_b"]["image_selected_success"]
        encoded = []
        phases = [0, 0]
        for step in range(49):
            frame = Image.new("RGB", (960, 500), "#101827")
            draw = ImageDraw.Draw(frame)
            draw.text((24, 14), "Predictive trajectory abstraction | mechanism demonstrator", fill="white", font=font)
            draw.text((24, 42), "SIMULATOR / ORACLE REPLAY - no learned world model result", fill="#fbbf24", font=font)
            draw.text((24, 72), title + f"   [prefix {row['prefix_id']}]", fill="#cbd5e1", font=small)
            for arm, (images, states) in enumerate(trajectories):
                left = 40 + arm * 470
                picture = Image.fromarray(images[step]).resize((325, 325), Image.Resampling.NEAREST)
                frame.paste(picture, (left, 130))
                draw.text((left, 103), "Default A -> B" if arm == 0 else "Oracle-selected candidate", fill="white", font=font)
                for goal, color, label in ((row["goal_a"], "#2563eb", "A"), (row["goal_b"], "#16a34a", "B")):
                    x, y = left + goal[0] * 5, 130 + goal[1] * 5
                    draw.ellipse((x-14, y-14, x+14, y+14), outline=color, width=3)
                    draw.text((x+16, y-8), label, fill=color, font=font)
                if step > 0:
                    near_a = np.linalg.norm(states[step] - row["goal_a"]) < 4.5
                    near_b = np.linalg.norm(states[step] - row["goal_b"]) < 4.5
                    if phases[arm] == 0 and near_a: phases[arm] = 1
                    elif phases[arm] == 1 and near_b: phases[arm] = 2
                draw.text((left, 464), ["A not reached", "A reached; waiting for B", "A then B complete"][phases[arm]],
                          fill="#4ade80" if phases[arm] == 2 else "#cbd5e1", font=small)
            draw.text((805, 72), f"step {step}/48", fill="white", font=small)
            buffer = io.BytesIO()
            frame.save(buffer, format="PNG", optimize=True)
            encoded.append("data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode())
            gif_frames.append(frame)
            if case_id == 0 and step == 48: poster = frame.copy()
        gif_frames.extend([gif_frames[-1]] * 10)
        cases.append({"title": title, "prefix_id": row["prefix_id"], "frames": encoded,
                      "default_success": bool(entry["default_a_then_b"]),
                      "oracle_success": bool(entry["oracle"])})
    gif_frames[0].save(args.output / "oracle_demo.gif", save_all=True, append_images=gif_frames[1:],
                       duration=110, loop=0, optimize=False)
    poster.save(args.output / "preview.png")
    page = '''<!doctype html><html lang="en"><meta charset="utf-8"><title>Trajectory WM: mechanism demo</title>
<style>body{font:17px system-ui;background:#101827;color:#e2e8f0;max-width:1050px;margin:32px auto;padding:0 16px}
h1{font-size:28px} .tag{color:#fbbf24}img{width:100%;border:1px solid #334155;border-radius:10px}
button,select{font:inherit;padding:8px;background:#243047;color:white;border:1px solid #475569;border-radius:5px}
input{width:65%}.card{background:#1e293b;padding:16px;border-radius:10px;margin:16px 0}small{color:#94a3b8}</style>
<h1>Can a compact predicted trajectory answer useful future queries?</h1>
<p class="tag">Research mechanism demo — simulator/oracle trajectories. Learned WM results pending.</p>
<div class="card">Proposed deployment: observations → shared action proposal → <b>WM summary</b> → query reader → selected action.<br>
Here, the simulator supplies the future to show what a predictor would need to distinguish.</div>
<select id="case"></select> <button id="play">Pause</button> <input id="step" type="range" min="0" max="48" value="0">
<p><img id="view" alt="Actual simulator replay"></p>
<div class="card"><b>All 48 development prefixes:</b> default 11/48 (22.9%); oracle bank 19/48 (39.6%).
Remaining bank headroom: 8/48 (+16.7 pp). This is a 48-step chunk comparison, not closed-loop episode success.</div>
<p>Examples deliberately cover a rescued default failure, an already-successful default, and a bank with no success.
They illustrate behavior; the aggregate above includes every prefix. Blue A and green B are display overlays only.</p>
<p><b>Training:</b> learn observation-derived query answers from RGB/actions. The method predicts a query-independent summary,
then answers reach, occupancy, endpoint and ordered-visit queries. No learned advantage has been measured yet.</p>
<small>Offline, self-contained file. Sources: jobs 53529 and 53568; original DINO-WM Wall simulator.</small>
<script>const cases=__DATA__;const selector=document.getElementById('case'),slider=document.getElementById('step');
cases.forEach((c,i)=>selector.add(new Option(c.title,i)));let running=true;
function render(){document.getElementById('view').src=cases[+selector.value].frames[+slider.value]}
selector.onchange=()=>{slider.value=0;render()};slider.oninput=render;
document.getElementById('play').onclick=()=>{running=!running;document.getElementById('play').textContent=running?'Pause':'Play'};
setInterval(()=>{if(running){slider.value=(+slider.value+1)%49;render()}},130);render();</script></html>'''
    (args.output / "index.html").write_text(page.replace("__DATA__", json.dumps(cases)))
    (args.output / "demo_manifest.json").write_text(json.dumps({
        "type": "SIMULATOR_ORACLE_NOT_LEARNED_WM", "prefixes": [c["prefix_id"] for c in cases],
        "selection": "first prefix in each predeclared outcome category",
        "source_jobs": [53529, 53568], "default_successes": 11, "oracle_successes": 19,
        "total_prefixes": 48}, indent=2))
    print(f"DEMO_READY {args.output}", flush=True)


if __name__ == "__main__":
    main()
