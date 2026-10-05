Use case: scientific-educational / infographic-diagram.
Create ONE accurate, detailed architecture figure for the CURRENT implemented method "Discrete Event World Model for Long-Horizon Robotic Planning" on OGBench visual Lights Out. This is an academic architecture diagram, not a proposed cube architecture and not CTA. Use sharp vector-like raster rendering, white background, readable clean sans-serif labels, high resolution landscape around 3840x2400. Prioritize correct directed wiring, generous whitespace, strong hierarchy, large text. Professional paper figure, flat rounded boxes, small schematic robot-arm and blue/yellow 4x5 button-grid illustrations. No decorative sci-fi graphics, no invented numerical results.

TITLE: "Discrete Event World Model"
SUBTITLE: "Learn events from offline pixel play • Plan in imagination • Execute and replan"

Layout: TWO large horizontal panels, stacked vertically, separated by a clear dashed boundary.
Panel A top ~45% height titled "A. OFFLINE LEARNING". Blue/teal modules.
Panel B lower ~50% height titled "B. ONLINE PLANNING & CONTROL". Purple imagined-planning group, orange physical execution group. Solid arrows = observations or actions; dashed arrows between panels = learned weights reused. Explicitly distinguish model imagination from the real robot.

PANEL A architecture:
Far left input "Offline play trajectories" with small strip of 64x64 camera-frame thumbnails and action vector icons, sublabel "(images, robot actions)".
Main top-row chain with arrows:
1 "Self-supervised encoder" / "Frozen ViT-tiny" / "64 patch tokens × 192"
2 "Slow state discovery" / "PCA → SFA → band-wise ICA" / "Select slow, binary components"
3 "Binary state code b" / "20 / 24 learned bits"
4 "Event discovery" / "Debounce + merge changes" / "Event signature: Δ = b_after XOR b_before" / "Train-only event vocabulary"
At right show transition tuples "(b, e, b′)" leading to "Event world model" / "Neural MLP: f(b,e) → b′" / "Round predicted bits".

Under these, three compact branches visibly correctly wired:
- From the binary state code and detected event boundaries to "Segment-majority pseudo-labels", then "CNN state reader" / "image → stable bits". This reader is trained to reproduce the discovered code, not ground-truth button labels.
- From event transition data and original images + actions to "Event-conditioned skill" / "2-frame CNN + FiLM" / "Event embedding + learned change map" / "8-action chunk; execute first 4" / "Train on approach → press → release".
- From event WM to "Imagined event rollouts", then "Cost-to-go h(b,g)" / "Bellman backups over all events" / "h = 0 at goal; otherwise 1 + min_e h_target(f(b,e),g)" / "Unit cost per event". No additional real-environment data from imaginary rollouts.

Small bottom footnote INSIDE PANEL A:
"Privileged state / GF(2) solver: diagnostics and oracle controls only; not inputs to the learned decision path."
A separate small neutral gray diagnostics box is acceptable, but NO arrow from privileged information into any deployed learned module.
Another small factual note: "Lights Out specialization: binary state and state-independent XOR effects."

PANEL B architecture:
Numbered large modules flowing left to right, with a visible external feedback loop returning from physical environment to the start.
Step 1: "1  OBSERVE" two inputs "Current image I_t" and "Goal image I_g", each a small button-grid thumbnail. Both arrows go to shared "CNN state reader" producing "Current code b_t" and "Goal code g". Goal code can be read once; keep both inputs explicit.
Step 2: a large PURPLE bounded zone labeled "2  PLAN IN IMAGINATION" containing:
- current and goal codes to "Batch weighted A* search"
- a small branching tree of predicted binary board states with hypothetical event edges; label "Enumerate candidate events"
- two distinct supporting boxes "Event WM f(b,e)" and "Heuristic h(b,g)" clearly feeding search, NOT receiving future real observations.
- search details in a small line "λ = 0.6 • Budget: 50k (4×5), 200k (4×6) expansions"
- "Goal test: predicted code = g"
Output "Event sequence [e1, e2, …]" then a distinct small selection box "Take first event e1".
Small fallback note "If search times out: choose the event with lowest predicted h".
Step 3: "3  EXECUTE ONE EVENT" orange "Learned skill π(a | I_t, I_prev, e1)". Inputs are selected event and current + recent image history via one clean bypass connection from observed images. Skill DOES NOT receive privileged state, analytical button IDs, predicted future frames, or the goal directly. Show arrow "Robot action chunk" to real arm pressing actual board, large label "Physical environment".
Step 4: "4  VERIFY & REPLAN" with "New camera image" to "Stable code-change detection" / "WM-consistency filter". Draw the feedback arrow back to observation / current code and planning with label "Read the actual state; replan after each event".
Physical environment also leads to a small final stop diamond "Environment success?" with "Yes → stop" and "No → continue loop". Do not show stopping only because predicted code equals goal; prediction goal test and environment success are different.
Dashed cross-panel reuse arrows or small matching weight badges should clearly connect trained CNN reader to deployed reader, trained event WM + heuristic to planner, trained skill to deployed skill without clutter.

Bottom legend:
"Blue: learned from offline data    Purple: imagined planning    Orange: real robot execution"
Bottom takeaway:
"Plan several events ahead. Execute one event. Observe the real outcome. Replan."

Accuracy constraints:
This figure represents a learned event planner, not a handwritten Lights Out solver. The XOR is how events are discovered offline, while a neural event WM predicts successors online. Search does not reset the real simulator or access future frames. Train and deploy routes must be visually separate. End-to-end learned controller; actual robot steps remain subject to the benchmark step limit. Do not imply zero-shot grid-size transfer. Do not include cube, CTA, HIQL comparison, success percentages, fake SOTA claims, or ground-truth coordinate labels.
Use English labels exactly as above where feasible for clean academic readability; mathematical subscripts may be typeset. Keep visible node text concise and readable. Avoid overloaded crossings; use orthogonal connectors and alignment.
