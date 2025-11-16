# Concepting

## Concept

**Energy-Matching Battle Game** (in THRML terms):

- There’s a **graphical model** (nodes + edges) with an energy function (E(x)).
- Players **tweak parameters** (biases, couplings, clamped nodes, temperature β).
- THRML **samples** from the resulting distribution.
- The **sampled configuration** determines who “controls” what, and therefore who wins the round / game.

So instead of units fighting directly, they’re fighting by reshaping an **energy landscape**.

## Models

### Core model

- Use an **Ising model**:

  - Each node: spin ∈ {+1, −1}
  - Interpret:

    - +1 = “Player A owns this tile”
    - −1 = “Player B owns this tile”

- Fixed small grid: e.g. 4×4 or 5×5.
- Every tile connected to its neighbors (up/down/left/right) with weights (J\_{ij}).

Use THRML:

- `SpinNode` for each tile.
- `IsingEBM` for the model.
- Two blocks: checkerboard partition for block Gibbs.

### Player actions (simple)

Each **turn**:

1. Both players get a small **budget** of “influence points”.
2. They can:

   - **Strengthen an edge** (J) between two neighboring tiles to favor alignment for themselves.
   - **Apply a local bias** (h) on a tile toward their spin.

3. After both have committed moves:

   - You run THRML sampling for N steps.
   - You take either:

     - The last sample, or
     - The empirical majority of spins across samples.

**Scoring rule**:

- Count tiles with +1 vs −1:

  - More +1 → Player A wins the round.
  - More −1 → Player B wins.

- Play best of 5 rounds.

### What this buys you

- **Easy to explain**:
  “Change the couplings and local preferences, then the physics engine (THRML) re-equilibrates, and whoever dominates the board wins.”
- **Easy to visualize**:
  Tiles colored red/blue, maybe intensity = confidence/probability.
- **Room for cool extras** if you have time:

  - Show a small chart of energy over sampling steps.
  - Show per-tile ownership probability.
  - Add a “heat” slider (β) affecting randomness.

---

## Scope

If you keep it to the scoped version above, **yes**:

### Backend complexity

- You’re mostly:

  - wiring THRML’s Ising model,
  - defining blocks,
  - defining a sampling schedule (e.g. 100 warmup, 50 samples per turn),
  - running Gibbs each turn.

- A 4×4 or 5×5 grid is tiny → sampling is very fast even on CPU.

### Frontend complexity

Minimal but sufficient:

- Display grid in a simple UI (could be:

  - a Python GUI (streamlit, gradio),
  - or a browser frontend with a small backend API).

- Click edges/tiles to allocate influence each turn.
- Show “Run round” button → call backend → update board.

You can avoid:

- networking complexity by doing everything in a local web app,
- fancy animation (simple fade transitions are enough).

So it’s very feasible **if you resist feature creep**.

# Gameplay

# 🎮 **Turn Sequence (Simple, Intuitive, Balanced)**

This design assumes 2 players (A & B), each trying to reshape the energy landscape of a small Ising grid.

---

## **Turn 0 — Setup**

- Board initializes with random +1/–1 spins, or a clean neutral starting state.
- All couplings (J\_{ij}) start at a default value (e.g., 0.5).
- All biases (h_i) start at 0.

---

## **Turn 1 — Player Planning Phase**

Both players receive **a fresh Influence Budget**, e.g.:

- **3 Edge Tokens** → can strengthen or weaken an edge
- **2 Bias Tokens** → can apply a bias to a tile
- **1 Temperature Shard (optional)** → adjust β for the next sampling run

Then:

### **Step 1: Player A Adjusts the Board**

Player A can:

- Click a tile → add positive bias (favor +1)
- Shift-click a tile → add negative bias (favor −1)
- Click an edge between tiles → increase alignment strength
- Shift-click an edge → weaken alignment
- Use slider to tweak temperature β (+ randomness / − randomness)

A small animation highlights their changes.

### **Step 2: Player B Adjusts the Board**

Player B gets the same actions, but:

- Their moves appear in another color (e.g., blue vs red)
- UI shows how many tokens remain

Both players modify the **same shared PGM**.

**Important:**
You can optionally hide Player A’s moves from Player B for a “fog of war” feeling — but for a hackathon demo, **keeping all moves visible is simpler and more understandable**.

---

## **Turn 2 — THRML Sampling Phase (“The World Reacts”)**

When both players press READY:

1. Show a mini animation:
   **“Rebalancing the World…”**
2. Run THRML:

   - 100 warmup steps
   - 50 Gibbs samples
   - Aggregate sign (majority vote) or take final state

3. Update the grid colors in real-time:

   - red = +1 (Player A)
   - blue = –1 (Player B)
   - optional color intensity = sample confidence

You can even show a small bar plot:

- proportion of +1 vs −1 states over the samples

**This sampling moment is the “magic” that sells the game**.

---

## **Turn 3 — Scoring Phase**

Two possible scoring systems (both good):

### **Option A — Tile Majority**

Whoever controls more tiles (spin direction) wins the round.

### **Option B — Territory Stability**

For each tile:

- if spin aligns with the tiles around it → +more points
- if spin is unstable → fewer points

_Option A is simpler, and I recommend it for hackathons._

---

## **Turn 4 — Next Round or Game End**

- Reset biases to zero.
- Keep couplings slightly persistent (fun strategic layer), _or_ reset everything.
- Re-run 3–5 rounds total.

The winner is the player who wins the majority of rounds.

---

# 📐 **Top-Level Layout**

```
 ----------------------------------------------
|                                              |
|                 GAME TITLE                   |
|          “Energy Matching Battle”            |
|                                              |
 ----------------------------------------------

 ---------------------------------------------------------
|   Left Panel (Player A)   |   Main Grid     | Right Panel (Player B) |
|                           |                 |                         |
 ---------------------------------------------------------
```

---

## **Left Panel — Player A Controls**

```
Player A Controls (Red)
-----------------------
Influence Budget:
• 3 Edge Tokens
• 2 Bias Tokens
• β Shard (optional)

Actions:
[ ] Add positive bias (+1)
[ ] Add negative bias (-1)
[ ] Strengthen edge
[ ] Weaken edge

Temperature (β):
[ Slider  - -●---  ]

[ READY ]
```

---

## **Main Panel — The Board**

A simple 5×5 grid of squares:

- Click tile = apply bias
- Click boundary/edge = adjust coupling
- Mouseover shows the current J or h values

Example:

```
+-----+-----+-----+-----+-----+
|  A  |  B  |  A  |  A  |  B  |
+-----+-----+-----+-----+-----+
|  B  |  B  |  A  |  B  |  A  |
+-----+-----+-----+-----+-----+
...
```

Each cell filled color-coded:

- Red = Player A control
- Blue = Player B control
- White/gray = neutral

During sampling, cells animate (fade) to new colors.

---

## **Right Panel — Player B Controls**

Same as Player A panel, but blue.

---

## **Bottom Panel — System Feedback (Optional but Cool)**

```
System Energy Over Sampling:
[ tiny plot       ]

Spin Proportions:
(A red) ████████▌ 68%
(B blue) ███▌     32%

Message Log:
• Player A strengthened edge (1,2)–(1,3)
• Player B biased tile (4,4) toward -1
• System sampled 50 states
```

This **instantly communicates** that THRML is doing real probabilistic inference.
