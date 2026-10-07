Create a technically accurate research architecture diagram for the CURRENT UNIFIED EVENT WORLD MODEL pipeline. This is an explanatory figure, not a proposed improved architecture.
Use case: scientific-educational. Clean publication-quality flat vector-like infographic, white background, crisp readable English text, landscape 16:10, generous spacing, navy typography. Blue for perception, purple for event world model / planning, orange for control, gray for raw data and rule-based processing. Rounded rectangles, clear arrowheads, no decorative robots or photographs, no invented modules. Title: "Unified Event World Model — Current Architecture".
Two clearly separated panels with the exact headings "A. OFFLINE TRAINING" and "B. CLOSED-LOOP PLANNING".

Panel A: Show the actual dependencies in two tiers.
Start with "Offline play data" (images + recorded actions).
Images flow to a blue/gray box "Frozen SAM2 + entity / event rules" with short subtext "Masks → identities → pseudo-labels". This box flows to "Train visual reader" with subtext "Entity state, rest, agent mask".
That trained reader then feeds "Self-label play images" then "Extract coarse events" with subtext "(state before, entity e, target x, state after)".
Fork the coarse-events output into TWO distinct branches:
1. Purple branch "Train event WM + cost-to-go" with subtext "Predict transition; learn search heuristic".
2. Gray box "Refine event timing + filter segments", then orange box "Train skill by behavior cloning" with subtext "Two images + e + start state + target x → 8 actions". A separate arrow from recorded play actions to this BC training box labeled "Action supervision".
Important: WM training consumes coarse events directly; only skill segments go through refinement/filtering. SAM2 is frozen. The reader, WM, and skill are trained in separate stages. No end-to-end optimization or RL is depicted.
A short footer within A: "Same pipeline; checkpoints trained separately for each environment".

Panel B: Show observation and goal images flowing through the blue "Trained reader" box to "Current state S + goal state G".
Then into a large purple container labeled "Event-space planner (weighted A*)". Within this container, explicitly show:
"Propose event (e, x)" → "Event WM: f(S, e, x) → predicted scene S_next" → "Score / search with h(S_next, G)".
The predicted whole-scene state stays INSIDE the planner. The planner's OUTPUT is a clearly labeled arrow "Selected event (e, x)" to the orange box "Trained skill".
Additional inputs to the skill: a gray box "Two recent raw images", and a thin arrow labeled "Entity state at event start". Inside skill write "Entity e + target x → action chunk".
Skill flows to "Execute first 4 of 8 actions", then gray box "Environment", then a feedback arrow labeled "New observation; check completion; replan" back to the reader / planner observation path.
Under the planner-to-skill arrow, a legible note: "x is a proposed target; the full WM prediction is not a skill input".
Compact legend at bottom: "Entity state: (u, v, R, G, B, covered)" and "SAM2 is used during training only".
Use e and x as mathematical symbols only where defined. Avoid substituting the symbol S for a learned future trajectory code: this implementation uses S to denote current entity states. Do NOT label it CTA. Do NOT draw a WM-predicted image or an arrow from the full predicted scene to the skill. Do NOT put SAM2 in the online loop. Do NOT depict privileged simulator state as an input. All arrows must match the described data flow, no arbitrary shortcuts. Prioritize accurate readable architecture over visual ornament.
