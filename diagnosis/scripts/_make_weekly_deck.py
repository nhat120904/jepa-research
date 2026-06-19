"""Build the plain weekly-report deck (white slides, bullets + figures).
Revised 2026-06-19 after the spatial-probe diagnostics overturned the
encoder/residual story."""
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from PIL import Image
from pathlib import Path

FIG = Path("results/figures/week_report")
OUT = Path("../Weekly_Report_2026-06-19.pptx")
DARK = RGBColor(0x20, 0x20, 0x20)
GREY = RGBColor(0x55, 0x55, 0x55)

prs = Presentation()
prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
blank = prs.slide_layouts[6]


def add_title(slide, text, sub=None):
    tb = slide.shapes.add_textbox(Inches(0.5), Inches(0.35), Inches(12.3), Inches(1.0))
    tf = tb.text_frame; tf.word_wrap = True
    p = tf.paragraphs[0]; r = p.add_run(); r.text = text
    r.font.size = Pt(27); r.font.bold = True; r.font.color.rgb = DARK
    if sub:
        p2 = tf.add_paragraph(); r2 = p2.add_run(); r2.text = sub
        r2.font.size = Pt(14); r2.font.italic = True; r2.font.color.rgb = GREY


def add_bullets(slide, bullets, x=0.6, y=1.5, w=6.4, h=5.4, size=16):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame; tf.word_wrap = True
    for i, (lvl, txt, *bold) in enumerate(bullets):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.level = lvl; p.space_after = Pt(7)
        r = p.add_run(); r.text = ("• " if lvl == 0 else "– ") + txt
        r.font.size = Pt(size - 2 * lvl); r.font.color.rgb = DARK
        if bold and bold[0]:
            r.font.bold = True


def add_image_fit(slide, path, x, y, maxw, maxh):
    w, h = Image.open(path).size; ar = w / h
    bw, bh = maxw, maxw / ar
    if bh > maxh:
        bh, bw = maxh, maxh * ar
    slide.shapes.add_picture(str(path), Inches(x + (maxw - bw) / 2),
                             Inches(y + (maxh - bh) / 2), Inches(bw), Inches(bh))


# 1 — title
s = prs.slides.add_slide(blank)
add_title(s, "Boundary-Blind World Models — Weekly Progress",
          "Week of 2026-06-12 → 06-19")
add_bullets(s, [
    (0, "Goal: get JEPA world models to plan contact tasks (push / pick-place), where they fail.", True),
    (0, "This week: pinned the failure to ONE channel — the predictor can't tell actions apart at the object level."),
    (0, "Encoder and factual object tracking are both fine; the apparent 'fixes' were measurement artifacts."),
    (0, "Reach still beats the paper: 94% vs 44.8%."),
], y=2.2, w=12.0, h=4.8, size=18)

# 2 — recap + question
s = prs.slides.add_slide(blank)
add_title(s, "Recap & this week's question")
add_bullets(s, [
    (0, "Boundary Blindness (BB): the predictor smears different actions into one latent near contact."),
    (0, "Closed-loop: Reach works (94%); Push / Pick-place stuck at 0% — the BB gap."),
    (0, "Last week: a scoring fix gave a small placement gain but zero successes."),
    (0, "Question:", True),
    (1, "Is the bottleneck the PLANNER, the ENCODER, or the PREDICTOR itself?"),
])
add_image_fit(s, FIG / "closed_loop.png", 7.2, 1.6, 5.8, 5.2)

# 3 — two legs + the catch
s = prs.slides.add_slide(blank)
add_title(s, "Two fix legs — and why we had to dig deeper")
add_bullets(s, [
    (0, "hexp — grounded EXPLORATION: null, slightly worse. Not the planner's search."),
    (0, "l2c — residual corrective PREDICTOR: looked like the BEST leg (+0.13).", True),
    (0, "But every fix was read through a lossy 'pooled' object readout (6.4 cm).", True),
    (1, "So we built a precise readout and re-measured — the l2c win did not survive (next slides)."),
])
add_image_fit(s, FIG / "fix_ladder.png", 6.9, 1.9, 6.1, 4.6)

# 4 — encoder + factual tracking are fine
s = prs.slides.add_slide(blank)
add_title(s, "Not the encoder, not factual tracking")
add_bullets(s, [
    (0, "A spatially-aware probe recovers the object to 2 cm from the SAME latent (pooled said 6.4 cm)."),
    (0, "92% within the 5 cm push radius → the encoder clearly holds the object.", True),
    (0, "The frozen predictor also tracks the object factually to ~3 cm over a 6-step rollout."),
    (0, "And the residual 'fix' is no better than frozen under the precise probe — it gamed the old readout.", True),
], y=1.4, w=12.2, h=2.4)
add_image_fit(s, FIG / "precision_pooled_vs_spatial.png", 2.5, 3.9, 8.3, 3.4)

# 5 — the real bottleneck
s = prs.slides.add_slide(blank)
add_title(s, "The real bottleneck: the predictor can't tell actions apart")
add_bullets(s, [
    (0, "Planning needs different actions → different predicted object (so CEM can choose)."),
    (0, "From one state, different actions move the object ~2.4 cm apart (ground truth)."),
    (0, "But the predictor predicts ~the same object for every action: 0.74 cm spread, corr ≈ 0.", True),
    (0, "→ True Boundary Blindness, at the object level, confirmed with a 2 cm-precise readout.", True),
])
add_image_fit(s, FIG / "counterfactual_smear.png", 7.4, 1.7, 5.6, 5.0)

# 6 — resolved + next
s = prs.slides.add_slide(blank)
add_title(s, "Resolved picture & next step")
add_bullets(s, [
    (0, "Where the contact-planning failure lives:", True),
    (1, "Encoder (object in latent): FINE — 2 cm."),
    (1, "Predictor, factual object tracking: FINE — ~3 cm / 6 steps."),
    (1, "Predictor, counterfactual object response: DEAD — the locus."),
    (0, "The residual fix was illusory (gamed the old readout) — an honest negative."),
    (0, "Next (launching now):", True),
    (1, "Supervise the action→object channel separately + plan against it with the 2 cm readout —"),
    (1, "the precisely-motivated shot at flipping contact successes beyond the paper."),
    (0, "Reach 94% vs 44.8% stands.", True),
], y=1.5, w=12.2)

prs.save(str(OUT))
print("wrote", OUT.resolve())
