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
