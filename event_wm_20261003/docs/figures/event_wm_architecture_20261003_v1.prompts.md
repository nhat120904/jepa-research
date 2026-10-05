# Architecture image prompts

Tool: built-in imagegen. Final artifact: event_wm_architecture_20261003_v1.png.

## Initial generation

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

## Label and routing correction

Edit the supplied architecture diagram. Preserve its overall two-panel academic layout, colors, all correct labels and module contents, robot illustrations, and title. Make precise correctness fixes:
1) In the Offline play trajectories box, replace "Action vectors (8-dim)" with "Robot actions (5-dim)". The 8 refers to temporal chunk length in the skill, not action dimensionality; preserve the skill labels "8-action chunk; execute first 4".
2) Correct online decision arrow routing: Batch weighted A* search outputs to "Event sequence [e1,e2,...]", which outputs to "Take first event e1", which outputs horizontally into the e1 input of the learned skill. Remove the long arrow directly from search into the execution area and remove the arrow from Take first event down into verification. No event selection arrow into verify.
3) The current + recent camera images are available to the learned skill, represented already by I_t and I_prev in the skill. Preserve these. Label older frame I_prev, not I_t-1, since the history gap need not be one step.
4) Correct the real-observation feedback: Physical environment gives "New camera image" in Verify & replan; this goes through Stable code-change detection / WM-consistency filter to "Updated code b_t+1". A clearly visible loop from that updated code returns to the CURRENT CODE input of the planner, or equivalently the Current image input of Observe with label "Next observation → read current state → replan". The loop must NEVER enter Goal image I_g. Goal image is a fixed external input read once.
5) Environment success must come from the Physical environment, not from the predicted code or updated code. Keep the success diamond, but remove Updated code → Environment success arrow. Route a separate short arrow from Physical environment into Environment success diamond. Yes → stop, No → continue loop. The main observation loop from Updated code must continue back to current state independently.
6) Make the offline Segment-majority pseudo-labels → CNN state reader arrow explicit and continuous across the gap between boxes.
7) Add a small crisp annotation near offline skill: "Training inputs: offline images + actions + discovered events". This must not suggest the binary bits themselves replace robot action training labels.
8) Keep all learned-weight reuse dashed connectors, other stage labels and equations. Correct any overlapping line near feedback. Do not add new method modules, performance claims, or hand-coded solver connections.

Output a polished high-resolution corrected diagram, with readable typography and clean orthogonal arrows. The most important requirement is faithful data flow: search → event sequence → first event → skill → physical environment → observed current state → replan; goal is fixed.

## Final deployment-panel redraw

Edit this image, preserving the title and the ENTIRE top panel A (Offline Learning), its colors and all modules.
COMPLETELY ERASE AND REDRAW THE WHOLE BOTTOM PANEL B from scratch, including all its old arrows and boxes. Do not retain any bottom-panel connections. The existing lower panel has wrong feedback and disconnected event arrows; a fresh simpler layout is required.

New panel B title: "B. ONLINE PLANNING & CONTROL".
Use exactly FIVE main numbered columns left-to-right, with clean straight arrows BETWEEN NEIGHBORING COLUMNS and one feedback lane below them. Make the lower panel taller if necessary. Never cross arrows over node text. High-resolution clean scientific diagram.

Column 1 "1. READ STATE": Current image I_t and a separate Goal image I_g both feed a shared CNN reader. Outputs b_t and g. The goal has a small "read once" label. Images should show the 4x5 blue-yellow light board.
Column 2 "2. IMAGINE & SEARCH": a single purple outer box, receives b_t and g from column 1. Within it place small modules Event WM f(b,e) and Heuristic h(b,g) above "Batch weighted A*". Include small branching-state tree. Small details: "Goal test: predicted code = g"; "50k / 200k expansion budget"; "Fallback: event with lowest h".
Column 3 "3. SELECT": a small purple box showing "Plan: [e1, e2, ...]" then a downward arrow INSIDE THE BOX to "First event e1".
Column 4 "4. EXECUTE": orange box "Learned skill". Formula "π(a | I_t, I_prev, e1)". Inputs shown as two small recent camera thumbnails and selected event. Label "8-action chunk; execute first 4". A SINGLE CLEAR solid arrow from the First event e1 output of column 3 to this skill. Camera inputs are available current/past observations, do not draw any future-image input.
Column 5 "5. REAL ROBOT": realistic small robot pressing the light board, label "Physical environment". A SINGLE CLEAR solid arrow from column 4 to column 5 labeled "Robot actions". Small text INSIDE THIS COLUMN: "Stop when environment reports success". Do NOT draw a success diamond anywhere. Do NOT draw a goal test outside the search box.

FEEDBACK LANE below the five columns:
A DOWNWARD arrow ONLY FROM PHYSICAL ENVIRONMENT in column 5 to a new smaller orange feedback box "New camera image → Stable code-change detection → WM-consistency filter → Updated code b_t+1".
From this feedback box, a long LEFTWARD arrow runs along the bottom of panel B, then turns UP along the LEFT EDGE OF COLUMN 2 and terminates precisely INTO THE IMAGINE & SEARCH BOX at its current-state input. Label this arrow "Observe the actual outcome and replan". This loop MUST return into column 2 SEARCH, never into the Goal image. Do not connect this loop to the reader, so there is no chance of an arrow accidentally entering the Goal image. The feedback box already includes state reading/detection.
No arrow from search or event selection to the feedback box. No arrow from updated code to a success decision.
Important: the order of solid arrows is unambiguously 1 READ STATE → 2 IMAGINE & SEARCH → 3 SELECT → 4 EXECUTE → 5 REAL ROBOT → feedback box → 2 IMAGINE & SEARCH.

Dashed weight reuse from panel A to corresponding B modules is allowed but keep it sparse: reader weights to column 1, WM + heuristic to column 2, skill to column 4. These dashed arrows terminate correctly and do not stand in for event/action arrows.

Footer remains three-color legend and takeaway "Plan several events ahead. Execute one event. Observe the real outcome. Replan."
Do not change top-panel action dimensionality (5-dim) or skill temporal chunk length (8).
One small top-panel improvement: draw a continuous clear arrow from Segment-majority pseudo-labels to CNN state reader, training relationship.
This is a complete bottom-panel layout replacement, not tiny local line edits. Preserve sharp readable English typography, enough whitespace, academic flat boxes, and strong colors.

