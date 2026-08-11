"""Detailed per-environment explanations shown in the CFZOO_Rollout UI.

Every entry documents, for one MODELS/<env> folder: the objectives, the raw
observation the PCN model receives (after the wrappers applied by
pcn.env_setup.build_experiment_setup), the discrete actions, the exact reward
mechanics, termination/truncation rules, and how to read the logged Pareto
front. The "deep_dive" field is a long-form HTML section rendered behind the
"Detailed Environment Info" button. All facts were taken from the environment
source code (envs/ and the installed mo_gymnasium package), not from memory.
"""

# Short per-channel names used by the plain-language counterfactual
# explanations. Order matches the env's reward channels. Whether a channel is
# a "gain" or a "cost" is inferred from the command box (upper bound <= 0 =>
# cost), which classifies every channel of every env below correctly.
OBJECTIVE_SHORT = {
    "collect_two": ["landmark A", "landmark B", "landmark C", "landmark D"],
    "branch-path": ["A-items", "B-items"],
    "reward-line": ["left payout", "right payout"],
    "three-tree": ["P-channel reward", "S-channel reward", "E-channel reward"],
    "dst": ["treasure", "time"],
    "minecart": ["ore 1", "ore 2", "fuel"],
    "fruit-tree-v0": ["Protein", "Carbs", "Fats", "Vitamins", "Minerals", "Water"],
    "four-room-v0": ["type-1 shapes", "type-2 shapes", "type-3 shapes"],
    "resource-gathering-v0": ["enemy risk", "gold", "gem"],
    "breakable-bottles-v0": ["time", "bottle deliveries", "breakage penalty"],
    "mo-mountaincar-timespeed-v0": ["time", "speed bonus"],
    "mo-lunar-lander-v3": ["landing outcome", "approach shaping", "main-engine fuel", "side-engine fuel"],
    "mo-reacher-v5": ["closeness to target 1", "closeness to target 2", "closeness to target 3", "closeness to target 4"],
    "walkroom2": ["dim-0 steps", "dim-1 steps"],
    "walkroom3": ["dim-0 steps", "dim-1 steps", "dim-2 steps"],
}

VARIANT_INFO = {
    "R": (
        "Return-conditioned PCN. The policy network receives (observation, desired-return "
        "vector). At every step the desired return is updated with "
        "remaining ← clip(remaining − reward, −∞, max_return) so the command always means "
        "“what is still left to collect from here on”."
    ),
    "RH": (
        "Return+Horizon-conditioned PCN. The policy network receives (observation, desired-return "
        "vector, desired-horizon scalar). The return command updates exactly like the R variant; "
        "the horizon command counts desired steps-to-go and is updated with "
        "horizon ← max(horizon − 1, 1) after every step. The logged front file stores, for every "
        "Pareto point, the horizon that was used when that return was achieved during training."
    ),
}

ENV_INFO = {
    "collect_two": {
        "title": "Collect-Two (c2)",
        "tagline": "7×7 grid, 4 landmarks, collect exactly two — order decides who gets 1.0 vs 0.8.",
        "objectives": ["A (red square, top)", "B (green triangle, left)", "C (orange diamond, right)", "D (purple hexagon, bottom)"],
        "observation": (
            "6 integers fed to the network as raw floats: [row, col, gotA, gotB, gotC, gotD]. "
            "row/col is the agent position on the 7×7 grid (0-indexed, row 0 = top); the four "
            "flags turn 1 once that landmark has been collected."
        ),
        "actions": "4 moves: 0=right, 1=up, 2=left, 3=down. Moving into a wall (dark cell) or off the grid leaves you in place (the step still counts). Wall moves are masked out of the policy.",
        "rewards": (
            "All zero until you enter an uncollected landmark cell. The FIRST landmark you collect "
            "pays 1.0 in its own channel; the SECOND pays 0.8 in its own channel. A landmark pays "
            "only once, so every achievable outcome is an ordered pair, e.g. (0, 0, 0.8, 1) = "
            "“collect D first (1.0), then C (0.8)”."
        ),
        "termination": "Episode terminates the moment two landmarks are collected. Truncated at 50 steps otherwise.",
        "front": "12 logged Pareto points = the 12 ordered pairs of distinct landmarks (4×3). Command bounds: [0,1] per channel.",
        "quirks": "The paper's worked example lives here: R_t=(0,0,0.8,1) at t=0, greedy 'down', foil 'right'.",
        "deep_dive": """
<h4>The map</h4>
<p>The layout is a fixed 7×7 grid with the start <b>S</b> dead-centre at (3,3) and one landmark on each edge midpoint:
<b>A</b> at (0,3) top, <b>B</b> at (3,0) left, <b>C</b> at (3,6) right, <b>D</b> at (6,3) bottom. Four 2×2 wall blocks fill the
quadrants, leaving a plus-shaped corridor system: from S you can walk straight 3 cells to any landmark, or around the
outer ring. Every landmark is exactly 3 steps away, so the minimal episode is 3 + 6 = 9 steps (3 to the first item,
then 6 around/through the centre to a second item).</p>
<h4>Dynamics</h4>
<p>Deterministic. Each action moves one cell (right/up/left/down); if the target cell is a wall or off-grid the agent
stays put but the step counter still advances. The environment additionally exposes an action mask (used by the
policy) that forbids wall/off-grid moves, so in practice the agent never wastes a step bumping into walls.</p>
<h4>Reward function, exactly</h4>
<ul>
<li>Entering an <i>uncollected</i> landmark cell: reward[that landmark] = <b>1.0</b> if it is the first item this episode,
<b>0.8</b> if it is the second. The landmark is then marked collected (rendered pale) and never pays again.</li>
<li>Every other transition pays the zero vector.</li>
</ul>
<h4>Termination</h4>
<p>Terminates the moment the second item is collected (count ≥ 2); truncates at 50 steps. Because the first item pays 1.0
and the second 0.8, the 12 achievable outcomes are exactly the ordered pairs of distinct landmarks — which is the
entire logged Pareto front.</p>
<h4>What the model sees</h4>
<p>The 6 raw numbers [row, col, 4 flags] go straight into the network (no one-hot). Commands are 4-dimensional in [0,1];
scaling factor 1. The command effectively encodes “which item first (channel with 1.0), which second (channel with 0.8)”.</p>
<h4>Counterfactual behaviour</h4>
<p>This is the paper's showcase env. Because outcomes are a discrete menu of 12 ordered pairs, a minimal ZOO command
R_cf usually sits on the decision boundary between two plans — realized naively it can leave the agent oscillating
between two corridors (the filmed right–left dithering). The verified-realization auditions fix this by re-issuing
the nearest front residual that still flips the first action.</p>
""",
    },
    "branch-path": {
        "title": "Branch-Path (bp)",
        "tagline": "Two opposite corridors of A/B items, 13-step budget — you cannot have it all.",
        "objectives": ["A items collected", "B items collected"],
        "observation": (
            "10 integers fed as raw floats: [row, col, then 8 flags, one per collectible in layout order "
            "(A@0,0), (A@0,2), (A@0,4), (B@0,6), (B@4,0), (B@4,2), (B@4,4), (A@4,6)]."
        ),
        "actions": "4 moves: 0=right, 1=up, 2=left, 3=down (wall/off-grid moves keep you in place; masked in the policy).",
        "rewards": "Entering an uncollected item cell pays +1.0 in that item's channel (A→channel 0, B→channel 1). Each item pays once.",
        "termination": (
            "Terminates only if ALL 8 items are collected — impossible within the 13-step limit, so in "
            "practice every episode is truncated at 13 steps."
        ),
        "front": "3 logged Pareto points. Max return is [3,3] but 13 steps only ever buys you a subset.",
        "quirks": "Because termination never fires, 'terminated' rollouts here are actually truncations at 13 steps — expected, not a failure.",
        "deep_dive": """
<h4>The map</h4>
<p>A 5×7 grid shaped like a capital I. Row 0 (top corridor): <b>A . A . A . B</b> — items at columns 0,2,4,6.
Row 4 (bottom corridor): <b>B . B . B . A</b>. Rows 1–3 are solid wall except the single connecting column 3, and the
start <b>S</b> sits in the middle of that vertical corridor at (2,3). So the agent's first decision is literally
“go up into A-land or down into B-land”, and switching sides later costs 4+ steps of backtracking through the corridor.</p>
<h4>Step budget arithmetic</h4>
<p>The 13-step limit is the whole game. From S it takes 2 steps to reach a corridor entrance cell (which itself holds
no item), then items alternate every 2 steps along the corridor. A one-sided sweep collects the corridor's three
same-type items plus the far-end opposite item if you push all the way (2+1+2+2+... pattern). Terminating requires all
8 items — mathematically unreachable in 13 steps — so <b>every</b> episode ends by truncation, and the front consists of
the best 13-step harvests: heavy-A mixes, heavy-B mixes, and the compromise.</p>
<h4>Reward function, exactly</h4>
<ul>
<li>Entering an uncollected item cell: +1.0 in channel 0 if the item is type A, +1.0 in channel 1 if type B. Once each.</li>
<li>Everything else: zero vector. There is no time penalty — time pressure comes purely from truncation.</li>
</ul>
<h4>What the model sees</h4>
<p>[row, col, 8 collected flags] as raw floats. Command bounds [0,3]² (ref point (0,0), max return (3,3)).</p>
<h4>Counterfactual behaviour</h4>
<p>The known failure mode: a foil at a late timestep can waste enough of the 13-step budget that no front point remains
reachable — the audit protocol then reports an honest best-effort miss. Foils at t=0/1 (redirecting the initial
up/down choice) realize cleanly onto the opposite corridor's front point.</p>
""",
    },
    "reward-line": {
        "title": "Reward-Line",
        "tagline": "Walk to the top row; the column you arrive in fixes the (1−u, u) payout.",
        "objectives": ["left objective (1−u)", "right objective (u)"],
        "observation": "2 floats: [row, col] on a 5×11 grid. Start is bottom-centre (row 4, col 5). No walls.",
        "actions": "4 moves: 0=right, 1=up, 2=left, 3=down (off-grid moves keep you in place; masked).",
        "rewards": (
            "Zero everywhere except the terminal step: reaching any cell in the TOP row (row 0) ends the "
            "episode and pays [1−u, u] where u = col/10. col 0 pays (1,0), col 10 pays (0,1), col 5 pays (0.5,0.5)."
        ),
        "termination": "Terminates on entering the top row; truncated at 20 steps.",
        "front": "11 logged Pareto points, one per top-row column: (1−k/10, k/10) for k = 0..10.",
        "quirks": "The command effectively picks a landing column; small command changes move the landing column by one.",
        "deep_dive": """
<h4>The map & dynamics</h4>
<p>An open 5×11 grid, no obstacles. The agent starts at (4,5), bottom-centre. Deterministic single-cell moves; stepping
off the edge keeps you in place (and is masked). The <b>entire top row is terminal</b>: the first time the agent's row
becomes 0 the episode ends immediately.</p>
<h4>Reward function, exactly</h4>
<p>Only the terminating transition pays: reward = [1 − u, u] with u = landing_column / 10. Everything before is the zero
vector — there is no step cost; the 20-step truncation just prevents infinite wandering.</p>
<h4>Geometry of the front</h4>
<p>The 11 achievable returns form a perfect discretised line from (1,0) to (0,1) in steps of 0.1 — hence the name. All
11 are Pareto-optimal (each trades 0.1 of one objective for 0.1 of the other), and all are reachable from the start
within the limit (worst case: 4 up + 5 sideways = 9 ≤ 20 steps). It is the cleanest possible testbed for command
geometry: the desired return IS the landing column.</p>
<h4>What the model sees</h4>
<p>Raw [row, col] floats. Commands live in [0,1]²; the model must map a command like (0.3, 0.7) to “walk to column 7,
then go up”.</p>
<h4>Counterfactual behaviour</h4>
<p>Foils are lateral moves (“why right rather than up?”). A minimal R_cf nudges the command mass between the two
channels just enough to shift the intended landing column by one — realization almost always lands exactly on the
neighbouring front point (99%+ hit rate in the paper suite).</p>
""",
    },
    "three-tree": {
        "title": "Three-Tree",
        "tagline": "4 sequential picks among P/S/E; every (node, action) pays a full 3-vector from a lookup table.",
        "objectives": ["channel 0 (P-flavoured)", "channel 1 (S-flavoured)", "channel 2 (E-flavoured)"],
        "observation": (
            "5 floats in [0,1]: [depth/4, count(P)/4, count(S)/4, count(E)/4, recovery_flag]. Note the encoding is "
            "order-blind — it sees how many of each action you took, not their order — plus a flag that is 1 on the "
            "special nodes PP, PPP, PPE."
        ),
        "actions": "3 branches: 0=P, 1=S, 2=E. Always all available (no masking).",
        "rewards": (
            "Every step pays a hand-crafted 3-vector from a table keyed by (current node, action). Root: P→(1.0,0.1,0.1), "
            "S→(0.1,1.0,0.1), E→(0.1,0.1,1.0). The PP subtree is special: PP+S pays (1.2,1.8,1.2) and PPP/PPE+S pay "
            "(1.2,1.2,1.2) — the route to balanced high-sum returns."
        ),
        "termination": "Exactly 4 actions, then the episode terminates (depth-4 tree). No truncation path.",
        "front": "14 logged Pareto points — cumulative 4-step sums over the 81 possible paths.",
        "quirks": "Lowest continue-rollout hit rate (~61%): some foils force you into a subtree that contains no front-reachable leaf.",
        "deep_dive": """
<h4>Structure</h4>
<p>A depth-4 decision tree over the alphabet {P, S, E}. Your “position” is the string of actions taken so far (the node,
e.g. <code>"PS"</code>); after the 4th action the episode terminates. There are 3⁴ = 81 leaf paths; their cumulative
reward vectors collapse to 14 non-dominated returns (the logged front).</p>
<h4>Reward table (the key entries)</h4>
<ul>
<li><b>Root</b>: each action pays ≈1.0 in its own channel, 0.1 in the others — the first pick declares a “flavour”.</li>
<li><b>Depth-1 nodes</b>: staying loyal (e.g. P then P) pays 1.1 in-channel; switching pays a 0.45/0.8 mix.</li>
<li><b>The PP anomaly</b>: node <code>PP</code> action S pays <b>(1.2, 1.8, 1.2)</b> — by far the richest single step —
and its children <code>PPP</code>/<code>PPE</code> pay (1.2, 1.2, 1.2) on S. This “recovery” corridor is how the
balanced-and-high front points are earned, and those three nodes are exactly the ones flagged by the observation's
recovery bit.</li>
<li><b>Generic depth-2/3 nodes</b>: symmetric mixes around 0.7–1.4 that reward committing to one channel.</li>
</ul>
<h4>The aliased observation — why it matters</h4>
<p>The model does <i>not</i> see the node string. It sees [depth, #P, #S, #E]/4 + recovery flag. Paths that permute the
same actions (PS vs SP) are indistinguishable — and the reward table is deliberately symmetric for those pairs, so the
aliasing is harmless <i>for the trained policy</i>. But it means the counterfactual command is doing all the work of
steering between subtrees that look identical to the state encoder.</p>
<h4>Counterfactual behaviour</h4>
<p>An early foil (say E at the root when the front point needed the PP corridor) permanently exiles the agent from the
richest leaves: whole subtrees simply do not contain any front-valued completion. That structural exile — not search
failure — is why ~39% of realized counterfactuals here cannot land on the front; the audition log shows every
remaining candidate rolled out and missing.</p>
""",
    },
    "dst": {
        "title": "Deep-Sea Treasure (dst)",
        "tagline": "Classic MORL submarine on the CONCAVE map: deeper treasure is worth much more, every step costs −1.",
        "objectives": ["treasure value", "time (−1 per step)"],
        "observation": (
            "A single integer index = row·11 + col of the submarine on the 11×11 map (an index-observation wrapper "
            "flattens the position; the model one-hot-encodes all 121 cells)."
        ),
        "actions": "4 moves in legacy order: 0=up, 1=right, 2=down, 3=left (remapped internally to the MO-Gymnasium order).",
        "rewards": (
            "Every step pays [treasure, −1]: the time channel is −1 always; the treasure channel is 0 until the submarine "
            "lands on a treasure cell, which pays its value and ends the episode. This build uses the CONCAVE map "
            "(Vamplew et al.): treasures 1, 2, 3, 5, 8, 16, 24, 50, 74, 124."
        ),
        "termination": "Terminates on any treasure; truncated at 200 steps.",
        "front": "10 logged Pareto points, one per treasure: (1,−1), (2,−3), (3,−5), (5,−7), (8,−8), (16,−9), (24,−13), (50,−14), (74,−17), (124,−19).",
        "quirks": (
            "The time channel makes the remaining command drift every step (remaining_time is clipped at the max_return "
            "value −1). Commands are 2-D, so counterfactual geometry is easy to visualise."
        ),
        "deep_dive": """
<h4>The map</h4>
<p>An 11×11 sea. The submarine starts at (0,0), top-left, on the surface row. Treasures sit on a descending staircase:
one per column, each deeper column holding a strictly more valuable chest. Cells below the staircase are rock (−10 in
the map array) — <b>impassable</b>, not penalising: a move into rock or off-grid leaves the submarine in place (and is
masked from the policy). This build passes <code>CONCAVE_MAP</code> to the env, so the values are
<b>1, 2, 3, 5, 8, 16, 24, 50, 74, 124</b> (not the 0.7–23.7 convex set).</p>
<h4>Reward function, exactly</h4>
<ul>
<li>Every step: time channel −1 (including the final step).</li>
<li>Landing on a treasure cell: treasure channel += its value, episode terminates.</li>
</ul>
<p>The optimal route to the treasure in column c is: sail right along the surface to column c, then dive straight down —
the front's time coordinates (−1, −3, −5, −7, −8, −9, −13, −14, −17, −19) are exactly those Manhattan path lengths.</p>
<h4>Concavity — why this map is interesting</h4>
<p>The value-vs-time curve is concave: going from 8→16 costs 1 extra step, but 16→24 costs 4. Linear-scalarisation agents
can only ever find the extreme points of a concave front; PCN's command conditioning reaches the interior points too,
which is exactly what the logged 10-point front demonstrates.</p>
<h4>What the model sees</h4>
<p>The flattened cell index, one-hot over 121 states (DiscreteCommandModel, 256 hidden). Command scaling is 0.1 per
channel — commands like (74, −17) are scaled to (7.4, −1.7) before embedding. Note the remaining-time command grows by
+1 every step (rem − (−1)) and is clipped at −1 from above, so “time remaining” stays meaningful along the rollout.</p>
<h4>Counterfactual behaviour</h4>
<p>Foils are “dive now instead of sailing on” (or vice versa). A minimal R_cf typically moves the treasure coordinate
between two adjacent chest values; realization then lands exactly on the neighbouring front point. Because chest values
are far apart (16→24→50), the fronts are well-separated and hits are clean.</p>
""",
    },
    "minecart": {
        "title": "Minecart (deterministic)",
        "tagline": "Drive a cart between 5 mines, fill a 1.5-unit hold with two ores, haul it home — fuel is the third objective.",
        "objectives": ["ore 1 delivered", "ore 2 delivered", "fuel (negative cost)"],
        "observation": (
            "7 floats: cart x, cart y, speed, sin(heading), cos(heading), ore-1 cargo, ore-2 cargo. Positions are in "
            "[0,1] with the home base at the top-left origin."
        ),
        "actions": "6: 0=Mine, 1=Left (rotate), 2=Right (rotate), 3=Accelerate, 4=Brake, 5=None.",
        "rewards": (
            "Fuel channel every step: −0.02 idle, −0.12 when accelerating, −0.22 when mining. Ore channels pay only at the "
            "moment the cart re-enters the home base circle: the whole cargo [ore1, ore2] is sold in one lump and the "
            "episode ends. Cargo capacity 1.5 total across both ores."
        ),
        "termination": "Terminates on returning to the base (after having left it); truncated at 1000 steps.",
        "front": "≈20+ logged points trading ore mix vs fuel. Continue-rollout uses a wider landing tolerance here (5e-3) because points are close together.",
        "quirks": "Longest episodes of the suite; deterministic variant — mine yields are exact constants, so replays are perfectly reproducible.",
        "deep_dive": """
<h4>World layout</h4>
<p>A unit square. The home base is a circle of radius 0.15 at the origin (top-left). Five mines (radius 0.14) line the
far edges: (0.16, 0.84), (0.50, 0.84), (0.84, 0.84), (0.84, 0.50), (0.84, 0.16). The cart starts at the base.</p>
<h4>Physics (per env step = 4 physics frames, incremental)</h4>
<ul>
<li><b>Left/Right</b>: rotate the heading by 10°/frame → 40° per action.</li>
<li><b>Accelerate</b>: +0.0075 speed/frame → +0.03 per action, capped at max speed 1.0.</li>
<li><b>Brake</b>: hard deceleration (speed drops to 0 within the action).</li>
<li><b>None</b>: coast; the cart keeps moving with its current speed and heading every frame.</li>
<li><b>Mine</b>: only effective when the cart is (nearly) stationary inside a mine's circle.</li>
</ul>
<h4>Mining — deterministic yields</h4>
<p>Each Mine action executes 4 mine ticks; in this deterministic variant each tick yields the mine's fixed mean, so one
Mine action yields: mine 1 → (0.8, 0), mine 2 → (0.6, 0.4), mine 3 → (0.8, 0.8), mine 4 → (0.4, 0.6),
mine 5 → (0, 0.8) — scaled down proportionally if it would overflow the 1.5-unit hold. Which mine you park at is
therefore the entire ore-mix decision.</p>
<h4>Fuel accounting, exactly</h4>
<p>Every action pays idle fuel −0.005 × 4 = <b>−0.02</b>. Accelerate adds −0.025 × 4 (total <b>−0.12</b>); Mine adds
−0.05 × 4 (total <b>−0.22</b>). Rotating, braking and coasting cost only the idle rate — so the fuel objective is
mostly “how many acceleration bursts and mine actions did you spend”.</p>
<h4>Termination & masking</h4>
<p>The episode ends the moment the cart re-enters the base circle after having departed; the cargo is paid out as the
ore rewards on that final step. Action masking: Accelerate masked at max speed, Brake masked when stopped, Mine masked
unless stopped inside a mine with free capacity.</p>
<h4>Counterfactual behaviour</h4>
<p>Foils mid-drive (“rotate left instead of coasting”) redirect the cart toward a different mine — realized episodes
land on a front point with a different ore mix and fuel bill. The front is dense (many ore-mix/fuel combinations), so
the app accepts landings within 5e-3 scaled distance (¼ of the minimum front spacing) instead of the 1e-3 default.</p>
""",
    },
    "fruit-tree-v0": {
        "title": "Fruit-Tree (depth 6)",
        "tagline": "Binary tree, 6 left/right choices, one 6-nutrient fruit at the leaf — every leaf is Pareto-optimal.",
        "objectives": ["Protein", "Carbs", "Fats", "Vitamins", "Minerals", "Water"],
        "observation": "2 integers: [depth, index-within-level] of the current node, consumed by the model as a one-hot over all tree nodes.",
        "actions": "2: 0=left child, 1=right child.",
        "rewards": (
            "Zero on every internal step. Reaching one of the 64 leaves pays that leaf's fixed 6-dimensional nutrient "
            "vector (values in [0, 10]); the leaf fruits are constructed so that every single leaf is Pareto-optimal."
        ),
        "termination": "Terminates at the leaf after exactly 6 choices. No truncation.",
        "front": "The logged front is large (up to the 64 leaves reached during training). Each step down the tree halves the reachable set.",
        "quirks": "Like three-tree, an early 'wrong' branch permanently exiles part of the front — auditions find the best leaf still reachable in your subtree.",
        "deep_dive": """
<h4>Structure</h4>
<p>A full binary tree of depth 6 (from Yang et al. 2019). State = (row, position-in-row); the root is (0,0), and action
a at node (d, p) moves to (d+1, 2p+a). After 6 choices you stand on one of the 2⁶ = 64 leaves; the episode terminates
there.</p>
<h4>Reward function, exactly</h4>
<p>Internal transitions pay the zero 6-vector. The final transition into a leaf pays that leaf's fruit: a fixed
6-dimensional nutrient vector (Protein, Carbs, Fats, Vitamins, Minerals, Water), hard-coded per leaf, values roughly
0–9.6. The fruit set is constructed so that <b>no leaf dominates any other</b> — all 64 are Pareto-optimal, and the
theoretical front is the full leaf set.</p>
<h4>The exponential exile property</h4>
<p>Every left/right choice discards exactly half of the remaining leaves. After k steps only 2^(6−k) fruits remain
reachable. This makes fruit-tree the purest test of command-following: the desired return must be steered to, bit by
bit, with no recovery possible.</p>
<h4>What the model sees</h4>
<p>The node one-hot (FruitTreeModel — a DiscreteCommandModel over all 127 nodes). Commands are 6-dimensional with
scaling 0.1; bounds [0, max-per-nutrient] taken from the leaf set.</p>
<h4>Counterfactual behaviour</h4>
<p>A foil at depth k (“go right, not left”) moves the agent into the sibling subtree: 2^(6−k−1) leaves remain, all
Pareto-optimal but possibly far from the original command. Auditions sort those survivors by distance to the current
remaining command and pick the first that flips — misses happen only when the logged front (leaves actually visited in
training) is sparse in that subtree.</p>
""",
    },
    "four-room-v0": {
        "title": "Four-Room",
        "tagline": "Four connected rooms, 12 shapes of 3 types; the far-corner goal pays a bonus in every channel.",
        "objectives": ["type-1 shapes", "type-2 shapes", "type-3 shapes"],
        "observation": (
            "One-hot row (13) + one-hot column (13) + 12 binary collected flags = 38 floats (custom wrapper around the "
            "base env's (row, col, flags) tuple)."
        ),
        "actions": "4 moves: 0=left, 1=up, 2=right, 3=down. Blocked moves (walls/out of bounds) keep you in place and are masked.",
        "rewards": (
            "Entering an uncollected shape cell pays +1 in that shape-type's channel (4 shapes of each of the 3 types on "
            "the map). Entering the goal G pays +1 in ALL three channels and terminates. Everything else is zero."
        ),
        "termination": "Terminates at the goal (top-right corner); truncated at 150 steps (TimeLimit added in env_setup).",
        "front": "Max return is [5,5,5] (4 shapes + goal bonus per channel); the logged front trades how many shapes of each type you sweep before heading to the goal.",
        "quirks": "Rooms connect through single-cell doorways — the doorway choices are where foil actions get interesting.",
        "deep_dive": """
<h4>The map</h4>
<p>A 13×13 gridworld (Barreto et al.'s successor-features world) split into four rooms by walls with single-cell
doorways. The agent starts at the bottom-left corner (12,0); the goal <b>G</b> is the top-right corner (0,12).
Twelve shape cells are scattered across the rooms: four each of types <b>1</b>, <b>2</b>, <b>3</b>
(type 1 at (0,0), (2,6), (6,10), (12,12); type 2 at (0,5), (5,0), (7,7), (10,6); type 3 at (5,5), (6,2), (7,12), (12,7)).</p>
<h4>Reward function, exactly</h4>
<ul>
<li>First entry into a shape cell: +1.0 in the channel of that shape's type. Each shape pays once.</li>
<li>Entering G: reward = <b>(1, 1, 1)</b> — a bonus in every channel — and the episode terminates.</li>
<li>All other steps: zero vector. No time cost; pressure comes from the 150-step truncation.</li>
</ul>
<p>Hence max return [5,5,5] = 4 shapes of the type + the goal bonus. Note that reaching the goal is worth +1 in every
objective, so all front points involve eventually finishing at G; the trade-off is which shapes you detour for on the
way (each detour costs steps against the 150 budget).</p>
<h4>Dynamics</h4>
<p>Deterministic moves; walls (X) and map edges block (you stay in place; masked). The wrapper adds a TimeLimit of
150 steps — the base env has none.</p>
<h4>What the model sees</h4>
<p>38 floats: one-hot row, one-hot column, and the 12 shape flags in map order. The FourRoomModel embeds position and
inventory separately before mixing with the command embedding (commands scaled by 0.25, bounds [0,5]³).</p>
<h4>Counterfactual behaviour</h4>
<p>Foils redirect doorway decisions (“leave through the north door, not the east one”), which reshuffles which shapes
are on-path. The logged front here is small, so realized counterfactuals either snap onto a nearby sweep-mix or revert
(the four-room revert case in the tests is exactly this: front of size 1 → snap-once re-anchors to the original plan).</p>
""",
    },
    "resource-gathering-v0": {
        "title": "Resource-Gathering (rsg)",
        "tagline": "Fetch gold and/or gem and come home — two ambush cells kill with 10% per touch.",
        "objectives": ["enemy (−1 if killed)", "gold delivered", "gem delivered"],
        "observation": "A flattened index of (position, carrying-gold, carrying-gem) on the 5×5 grid: index = (row·5+col)·4 + 2·gold + gem; one-hot over 100 states in the model.",
        "actions": "4 moves: 0=up, 1=down, 2=left, 3=right. Off-grid moves keep you in place (masked).",
        "rewards": (
            "All zero until an episode-ending event: reaching home pays [0, gold?1:0, gem?1:0] for what you carry; "
            "stepping on an enemy cell kills you with 10% probability, paying [−1, 0, 0] and ending the episode "
            "immediately (you do not respawn)."
        ),
        "termination": "Terminates on home delivery or on a kill; truncated at 100 steps.",
        "front": "Small front (gold-only, gem-only, both — with risk-adjusted step counts). In this suite it collapses to 1 logged point, which is why continue-rollout trivially 'reverts' here.",
        "quirks": "The 10% ambush is random BUT seeded: the app resets with a fixed seed, so replays and auditions are exactly reproducible.",
        "deep_dive": """
<h4>The map</h4>
<pre style="line-height:1.3">
.  .  R1 E2 .
.  .  E1 .  R2
.  .  .  .  .
.  .  .  .  .
.  .  H  .  .
</pre>
<p>5×5 grid. <b>H</b>ome at (4,2) is both the start and the delivery point. Gold <b>R1</b> at (0,2), gem <b>R2</b> at
(1,4). Two enemy cells: <b>E1</b> at (1,2) — directly on the straight path to the gold — and <b>E2</b> at (0,3) — on
the short route between gold and gem.</p>
<h4>Dynamics & reward, exactly</h4>
<ul>
<li>Walking onto R1/R2 silently sets the carrying flag (no reward yet; the item stays “in hand”).</li>
<li>Walking onto E1/E2: with probability 0.1 the agent is killed — reward (−1, 0, 0), episode <b>terminates on the
spot</b> (no respawn-at-home in this implementation).</li>
<li>Walking onto H: episode terminates with reward (0, has_gold, has_gem).</li>
<li>All other steps: zero. TimeLimit 100.</li>
</ul>
<h4>The risk trade-off</h4>
<p>Gold via the straight path crosses E1 twice (10% each pass); dodging E1 costs 2 extra steps per pass. The gem run
can avoid enemies entirely (10 steps round trip). Getting both cheaply means threading E2. The classic front therefore
mixes expected-value outcomes: safe-but-slow routes vs risky-but-short ones.</p>
<h4>Determinism note</h4>
<p>The ambush uses the env's seeded RNG. Because the app always resets with the same seed and replays the same action
prefix, the enemy dice rolls are identical on every audition — the counterfactual comparisons are exact, not averaged.</p>
<h4>Counterfactual behaviour</h4>
<p>This suite's logged front contains a single point (the safe gold+gem run), so there is no alternative front target:
after any foil, the only coherent completion is the original plan — the protocol's fallback snaps back to it, which
the results table reports as a revert. That is a property of the front, not a search failure.</p>
""",
    },
    "breakable-bottles-v0": {
        "title": "Breakable-Bottles (bb)",
        "tagline": "Shuttle bottles down a 5-cell corridor; carrying two risks dropping one forever.",
        "objectives": ["time (−1 per step)", "bottles delivered (+25 each)", "potential-based breakage penalty"],
        "observation": "A flattened index over (location 0–4, carrying 0–2, delivered 0–2, per-interior-cell dropped flags) — 360 discrete states, one-hot in the model.",
        "actions": "3: 0=move left, 1=move right, 2=pick up bottle. Masked when impossible (left at cell 0, right at cell 4, pickup away from the source or at capacity).",
        "rewards": (
            "Time: −1 every step. Delivery: +25 per bottle deposited when entering the destination (cell 4). Breakage: a "
            "potential-based penalty of −1 fires on the step a bottle is dropped (and is never recovered in this "
            "breakable variant)."
        ),
        "termination": "Terminates when 2 bottles have been delivered; truncated at 100 steps.",
        "front": "Effectively 1 logged point in this suite — the safe two-trip strategy. With no second front point, continue-rollout has nowhere alternative to land (the known structural bb limitation).",
        "quirks": "Drops are stochastic but seeded — fixed-seed resets make every replay identical.",
        "deep_dive": """
<h4>Setup</h4>
<p>A 1×5 corridor (cells 0–4). The bottle <b>source</b> is cell 0; the <b>destination</b> is cell 4 — and, perhaps
surprisingly, the agent <b>starts at the destination</b> (cell 4), so every episode begins with a walk left to fetch.
This env is the “BreakableBottles” AI-safety problem from Vamplew et al.'s low-impact-agents paper.</p>
<h4>Dynamics, exactly</h4>
<ul>
<li><b>Pick up</b> (at cell 0, carrying &lt; 2): +1 bottle in hand. In this breakable variant, dropped bottles can
<i>never</i> be picked back up (the pickup-from-ground branch is disabled).</li>
<li><b>Move left/right</b>: shifts one cell. If you are carrying <b>2</b> bottles and move while on an interior cell
(1–3), there is a <b>10% chance per move</b> that one bottle drops onto that cell, permanently.</li>
<li><b>Delivery</b>: entering cell 4 with bottles converts them: +25 per bottle (capped at 2 delivered total).
Delivering the 2nd bottle terminates the episode.</li>
</ul>
<h4>Reward channels, exactly</h4>
<ul>
<li>Channel 0 (time): −1 every single step.</li>
<li>Channel 1 (bottles): +25 per delivered bottle (so 50 for a full episode).</li>
<li>Channel 2 (impact): potential-based, Δφ where φ = −1 if any bottle lies dropped on the ground, else 0. It fires −1
on the step a bottle first drops; since drops are irreversible here, the episode total stays −1 once it happens.</li>
</ul>
<h4>The dilemma</h4>
<p>Carrying 2 bottles needs one round trip (~9 moves + 2 pickups → time ≈ −11) but risks the −1 impact penalty
(~27% chance of at least one drop over 3 risky moves — and a dropped bottle also costs 25 delivery reward since it can
never be delivered). Two single-bottle trips are drop-proof but cost ~8 extra steps. The logged front in this suite
contains only the safe strategy.</p>
<h4>Counterfactual behaviour</h4>
<p>With a front of size 1, no alternative front-coherent completion exists — foils here produce honest best-effort
rollouts (the paper's known bb structural ceiling). The interesting output is the audition log itself: you can watch
every candidate fail the landing check.</p>
""",
    },
    "mo-mountaincar-timespeed-v0": {
        "title": "MO Mountain-Car (time / speed)",
        "tagline": "The classic underpowered car, scored on finishing fast AND driving fast.",
        "objectives": ["time (−1 per step)", "speed bonus (15·|velocity| per step)"],
        "observation": "2 floats: [position ∈ [−1.2, 0.6], velocity ∈ [−0.07, 0.07]].",
        "actions": "3: 0=accelerate left, 1=coast, 2=accelerate right.",
        "action_names": ["accel left", "coast", "accel right"],
        "rewards": "Every step pays [−1 (0 on the terminating step), 15·|velocity|]. Reaching the right hilltop flag ends the episode.",
        "termination": "Terminates at the flag (position ≥ 0.5); truncated at 200 steps.",
        "front": "Trades finishing quickly (fewer −1s) against farming kinetic energy on the way (more speed bonus) — rocking back and forth farms speed but costs time.",
        "quirks": "Continuous physics: tiny command changes can flip bang-bang action choices sharply.",
        "deep_dive": """
<h4>Physics (standard Gymnasium MountainCar)</h4>
<p>State (position p, velocity v). Each step: v ← clip(v + (action−1)·0.001 − cos(3p)·0.0025, ±0.07);
p ← clip(p + v, [−1.2, 0.6]); hitting the left wall zeroes the velocity. Start: p ~ Uniform(−0.6, −0.4) (seeded),
v = 0. The engine (0.001) is weaker than gravity on the slopes (0.0025) — the car must pump energy by swinging.</p>
<h4>Reward channels (the “timespeed” variant)</h4>
<ul>
<li>Channel 0 (time): −1 every step, except 0 on the step that reaches the goal (p ≥ 0.5).</li>
<li>Channel 1 (speed): +15·|v| every step — up to 1.05/step at max speed.</li>
</ul>
<p>This variant registers the base MOMountainCar with <code>remove_move_penalty=True, add_speed_objective=True</code>:
the usual reverse/forward action-penalty channels are removed and replaced by the speed bonus.</p>
<h4>The trade-off, concretely</h4>
<p>The time-optimal policy finishes in ≈ 90–110 steps (return₀ ≈ −100) collecting only the speed earned en route. A
speed-farming policy keeps swinging in the valley near max |v| before finally climbing out — every extra swing costs
~40–60 time units and buys ~30–50 speed units. That produces the long, fine-grained logged front (134 points in the R
variant) — the densest of the suite, which is why continue-rollout occasionally lands 0.01 (scaled) away from the front
and is scored an honest near-miss.</p>
<h4>What the model sees</h4>
<p>Raw [p, v]. Command scaling is 0.01 (returns are order-100). PCN was seeded with an energy-pumping demonstration
policy during training (random policies essentially never reach the goal in 200 steps), which is where the spread of
its front comes from.</p>
<h4>Counterfactual behaviour</h4>
<p>Policies here are near-bang-bang (accelerate in the direction of motion). A foil like “coast instead of accelerating”
corresponds to command shifts along the swing-count axis; realized episodes typically differ by one full extra/fewer
swing — a visibly different trajectory in the rendered video.</p>
""",
    },
    "mo-lunar-lander-v3": {
        "title": "MO Lunar-Lander",
        "tagline": "Land the module: outcome, shaping, and two separate fuel meters.",
        "objectives": ["landing outcome (±100)", "shaping (distance/velocity/angle potential)", "main-engine fuel (−1 per firing step)", "side-engine fuel (−1 per firing step)"],
        "observation": "8 floats: x, y, vx, vy, angle, angular velocity, left-leg contact, right-leg contact (normalised to the viewport/helipad).",
        "actions": "4: 0=do nothing, 1=fire left engine, 2=fire main engine, 3=fire right engine.",
        "action_names": ["do nothing", "left engine", "main engine", "right engine"],
        "rewards": (
            "Channel 0 pays only at the end: +100 safe landing (lander at rest), −100 crash or flying out of bounds. "
            "Channel 1 is the classic dense shaping (potential difference on distance/speed/tilt + leg-contact bonuses). "
            "Channels 2/3 pay −1 on every step the main/side engine fires."
        ),
        "termination": "Terminates on rest (+100), crash, or |x| ≥ 1 (−100); truncated at 1000 frames.",
        "front": "Trades landing reliability and shaping smoothness against the two fuel budgets (gentler descents burn more main-engine fuel).",
        "quirks": "Deterministic for a fixed reset seed (wind disabled; the small engine-dispersion noise uses the seeded RNG).",
        "deep_dive": """
<h4>Physics</h4>
<p>Box2D lander, discrete actions. The main engine applies a large downward-facing impulse (with tiny seeded dispersion
noise); the side engines torque/translate the craft. Terrain and helipad are generated from the reset seed; wind is
off by default, so a fixed seed makes the whole episode deterministic — which the app relies on for replays.</p>
<h4>Reward channels, exactly (the vector differs from the scalar env!)</h4>
<ul>
<li><b>Channel 0 — outcome</b>: 0 all episode; −100 if the body crashes or leaves the screen (|x| ≥ 1), +100 if the
lander comes to rest. Terminal either way.</li>
<li><b>Channel 1 — shaping</b>: potential difference of
φ = −100·√(x²+y²) − 100·√(vx²+vy²) − 100·|angle| + 10·leg₁ + 10·leg₂. Summed over the episode it telescopes to
φ(end) − φ(start): smooth, centred, upright approaches score higher.</li>
<li><b>Channel 2 — main fuel</b>: −1 for every step the main engine fires (the scalar env weighs this ×0.3, the vector
env does NOT — full −1 per firing).</li>
<li><b>Channel 3 — side fuel</b>: −1 for every step a side engine fires.</li>
</ul>
<h4>The trade-off</h4>
<p>A soft, controlled landing needs many main-engine burns (channel 2 −60…−100) but maximises channels 0/1. Fuel-stingy
commands descend faster and burn less, risking harder touchdowns. The logged front spans that spectrum, including
low-fuel profiles that still land.</p>
<h4>What the model sees</h4>
<p>The raw 8-float state; commands scaled by 0.01 (returns are order-100). PCN training used Gymnasium's built-in PD
landing heuristic to seed the replay buffer (random policies almost never land), then spread the front by randomly
suppressing main-engine burns during seeding.</p>
<h4>Counterfactual behaviour</h4>
<p>Foils like “fire the left engine instead of coasting” tilt the descent profile; realization typically lands on a
neighbouring front point with a different fuel/shaping mix. Episodes are hundreds of steps, so auditions take a few
seconds each — the app caps and reports honestly.</p>
""",
    },
    "walkroom2": {
        "title": "WalkRoom 2D",
        "tagline": "Deep-Sea-Treasure generalised: pay −1 per step per dimension, choose which border point to walk to.",
        "objectives": ["dim-0 steps (−1 each)", "dim-1 steps (−1 each)"],
        "observation": "Position (x0, x1) on the 20×20 grid, encoded as a multi-one-hot vector (20 + 20 = 40 floats: one-hot of each coordinate).",
        "actions": "4: 0=+dim0, 1=+dim1, 2=−dim0, 3=−dim1 (forward/backward along each axis). Moves that would leave the grid are masked.",
        "action_names": ["+dim0 (right)", "+dim1 (down)", "−dim0 (left)", "−dim1 (up)"],
        "rewards": (
            "Every move pays −1 in the channel of the dimension you moved along (even if the move is clipped at a wall) "
            "and 0 in the other channel. There are no positive rewards anywhere."
        ),
        "termination": "Terminates when x1 reaches the border height room[x0] — a jittered anti-diagonal curve. Truncated at 200 steps.",
        "front": "The non-dominated set of −(x0, border(x0)) over all columns: each border point costs a different (dim0, dim1) mix.",
        "quirks": "Synthetic scaling benchmark from the PCN paper. The app draws its own visual (the env has no pixel renderer): red cells are the terminal border.",
        "deep_dive": """
<h4>Definition</h4>
<p>WalkRoom is the PCN paper's synthetic benchmark for scaling with the number of objectives — essentially Deep-Sea
Treasure generalised to n dimensions. Here n = 2, grid size S = 20. The agent starts at the origin (0,0); actions move
one cell forward or backward along a chosen dimension (|A| = 2n = 4); moving along dimension i pays reward
<b>−e_i</b> — i.e. −1 on objective i, 0 elsewhere. Nothing ever pays positive reward.</p>
<h4>The border, exactly</h4>
<p>A height value is generated for every column x0 (seed 0):
<code>limit(x0) = clip(round(19 − x0 + N(0, 2)), 0, 19)</code> — an anti-diagonal staircase with Gaussian jitter.
The episode <b>terminates as soon as x1 ≥ limit(x0)</b> after a move. So walking right (spending dim-0 budget) lowers
the wall you must walk down to; walking down (spending dim-1 budget) approaches the wall directly.</p>
<h4>The Pareto front</h4>
<p>A direct path to the border cell in column x0 costs exactly (−x0, −limit(x0)). The front is the non-dominated subset
of those pairs: because of the jitter, some columns are strictly worse than neighbours (their border sits deeper in
both senses) and drop out. That is why the logged front is an uneven subset of the 20 columns.</p>
<h4>Details worth knowing</h4>
<ul>
<li>Backward moves also cost −1 (reward = −|movement|), and a move clipped at the grid edge still pays −1 — wasted
motion is pure loss in this env, which is why optimal policies are monotone staircase walks.</li>
<li>Observation: multi-one-hot of the coordinates (40 floats). Command bounds [−20, 0]²; scaling 0.1.</li>
<li>TimeLimit 200 (never binding for sensible policies — the border is at most 38 steps away).</li>
</ul>
<h4>Reading the app's visual</h4>
<p>White cells = free room; <b>red cells = at/beyond the border</b> (stepping onto the dark-red cell of a column ends the
episode); blue dot = agent; light-blue squares = this episode's trail. The command (−a, −b) literally means “end in a
column near a, after roughly b downward steps”.</p>
<h4>Counterfactual behaviour</h4>
<p>Foils swap a dim-0 step for a dim-1 step (or a forward for a backward). Since every front point is “walk to column
x0's border cell”, realized counterfactuals land on the neighbouring border column's cost pair — clean, discrete
geometry, ideal for sanity-checking the whole pipeline.</p>
""",
    },
    "walkroom3": {
        "title": "WalkRoom 3D",
        "tagline": "The same idea with three cost dimensions — the border becomes a jittered surface over (x0, x1).",
        "objectives": ["dim-0 steps (−1 each)", "dim-1 steps (−1 each)", "dim-2 steps (−1 each)"],
        "observation": "Position (x0, x1, x2) on the 20×20×20 grid as a multi-one-hot vector (3 × 20 = 60 floats).",
        "actions": "6: 0=+dim0, 1=+dim1, 2=+dim2, 3=−dim0, 4=−dim1, 5=−dim2. Out-of-grid moves are masked.",
        "action_names": ["+dim0", "+dim1", "+dim2 (deeper)", "−dim0", "−dim1", "−dim2"],
        "rewards": "Every move pays −1 in the moved dimension's channel (even when clipped), 0 elsewhere. No positive rewards.",
        "termination": "Terminates when the depth x2 reaches the border height room[x0, x1] — a jittered anti-diagonal surface. Truncated at 200 steps.",
        "front": "Non-dominated −(x0, x1, border(x0,x1)) over the 400 (x0,x1) columns — a 3-D staircase of cost trade-offs.",
        "quirks": "The app renders a custom top-down view: cell shade = border depth at that column, the side gauge shows your dim-2 depth against the local border.",
        "deep_dive": """
<h4>Definition</h4>
<p>WalkRoom with n = 3 objectives, grid size S = 20. Start at (0,0,0); 6 actions (±1 along each of the three
dimensions); moving along dimension i pays −1 on objective i and 0 on the others. As in 2D, clipped moves still pay,
and nothing is ever positive — episode returns are exactly −(steps spent per dimension).</p>
<h4>The border surface</h4>
<p>For every column (x0, x1) a depth limit is generated (seed 0):
<code>limit(x0,x1) = clip(round(19 − (x0 + x1) + N(0, 2)), 0, 19)</code>.
The episode terminates when the third coordinate reaches it: <b>x2 ≥ limit(x0, x1)</b>. Spending budget in dims 0/1
moves you to columns whose border is shallower in dim 2 — the three cost channels trade against each other through the
anti-diagonal surface.</p>
<h4>The Pareto front</h4>
<p>A direct path to column (x0, x1) costs (−x0, −x1, −limit(x0,x1)). The logged front is the non-dominated subset over
all 400 columns — much larger than in 2D (the surface jitter creates many incomparable corners), which is exactly why
the paper uses WalkRoom to study scaling: front size explodes with n while the dynamics stay trivial.</p>
<h4>Details worth knowing</h4>
<ul>
<li>Observation: 60-float multi-one-hot (one-hot of each coordinate separately) — positions are never aliased.</li>
<li>Command bounds [−20, 0]³, scaling 0.1; TimeLimit 200.</li>
<li>The RH front file adds the horizon actually used per point (≈ the Manhattan distance to the border cell).</li>
</ul>
<h4>Reading the app's visual</h4>
<p>Top-down map over (x0 → right, x1 → down): <b>cell shade encodes the border depth</b> at that column — darker blue =
the episode ends after fewer dim-2 steps there. The orange dot is the agent's column (white arc = fraction of the local
depth already walked); yellow squares are the trail. The side gauge shows dim-2 directly: the orange marker is your
depth, the red line the local border — when they meet, the episode ends.</p>
<h4>Counterfactual behaviour</h4>
<p>Foils redistribute cost between the three channels (“step in dim 1 instead of dim 2”). Because front points are
columns of the border surface, realizations land on nearby columns with a different (x0, x1, depth) cost split — you
can watch the trail bend toward a different part of the surface.</p>
""",
    },
    "mo-reacher-v5": {
        "title": "MO Reacher (v5)",
        "tagline": "2-joint arm, 4 fixed targets at the compass points, one closeness channel per target.",
        "objectives": ["closeness to target 1 (east)", "closeness to target 2 (west)", "closeness to target 3 (north)", "closeness to target 4 (south)"],
        "observation": "6 floats: [cos θ1, cos θ2, sin θ1, sin θ2, 0.1·ω1, 0.1·ω2] — joint angles and (scaled) angular velocities. Target positions are fixed and not observed.",
        "actions": "9 discrete torque pairs: each joint applies −1, 0 or +1 (index = 3·(τ1+1) + (τ2+1)).",
        "action_names": ["τ(−1,−1)", "τ(−1,0)", "τ(−1,+1)", "τ(0,−1)", "τ(0,0)", "τ(0,+1)", "τ(+1,−1)", "τ(+1,0)", "τ(+1,+1)"],
        "rewards": "Every step, channel i pays 1 − 4·‖fingertip − target_i‖ (planar distance): +1 when touching target i, negative when far. You cannot be close to all four at once.",
        "termination": "No terminal state — every episode runs the full 50 steps (truncation).",
        "front": "Very large logged front (thousands of points): every parking/orbiting profile between the four targets is a different 4-channel mix.",
        "quirks": "Rendering needs MuJoCo's offscreen renderer; if unavailable the app shows the state numerically instead of a frame.",
        "deep_dive": """
<h4>Setup</h4>
<p>The MuJoCo 2-joint planar reacher. Four targets are pinned at the compass points of the workspace:
target 1 (0.14, 0), target 2 (−0.14, 0), target 3 (0, 0.14), target 4 (0, −0.14). The arm starts pointing at
θ = [0, π/2] with tiny seeded velocity noise; each env step applies the chosen torque pair for 2 physics frames.</p>
<h4>Reward function, exactly</h4>
<p>Per step and per target i: r_i = 1 − 4·‖fingertip − target_i‖₂ (planar). Touching a target yields ≈ +1 in its
channel; the workspace radius (~0.21) makes far targets score around −0.4…0. Because the four targets are mutually
exclusive positions, holding +1 in one channel pins the other three at their geometric consolation values — the front
is the set of all time-weighted mixtures achievable in 50 steps.</p>
<h4>Actions</h4>
<p>The 9 discrete actions enumerate torque pairs (τ_shoulder, τ_elbow) ∈ {−1, 0, +1}²; index = 3·(τ1+1) + (τ2+1)
(so action 4 = (0,0) is “hold”). No masking — all 9 always legal.</p>
<h4>Episode profile</h4>
<p>No terminal state exists; every episode is exactly 50 steps and ends by truncation — the app counts that as the
normal ending for this env. Returns are sums of 50 per-step closenesses, hence max_return 50 per channel.</p>
<h4>What the model sees & the front</h4>
<p>Only proprioception (angles/velocities) — the targets are baked into the reward, not observed. The logged front is
huge (≈2,500 points in the R variant): every “swing to target 3 at step k, then hover” profile is a distinct return
vector. Continue-rollout auditions here can reject hundreds of candidates in under a second because the first-action
check is a single forward pass per candidate.</p>
<h4>Counterfactual behaviour</h4>
<p>A foil torque at step t deflects the swing; the realization protocol finds the front residual whose hover-mix
matches the deflected trajectory. With such a dense front, landings are typically within 1e-3 scaled distance of some
logged point even though exact target-identity is rare.</p>
""",
    },
}
