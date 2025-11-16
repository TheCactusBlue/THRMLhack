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

# Gameplay (REDESIGNED for Fun & Strategy)

## 🎯 Core Design Principles

1. **Fast Setup, Dramatic Resolution** - Reduce planning time, increase payoff tension
2. **Strategic Continuity** - Actions echo across rounds
3. **Skill > Luck** - Reduce randomness, increase player control
4. **Visible Physics** - Players should see and understand the energy landscape

---

# 🎮 **Redesigned Turn Sequence**

This design creates a strategic, engaging game where physics matters but skill dominates.

---

## **Turn 0 — Game Setup**

- Board initializes with random +1/–1 spins
- All couplings (J\_{ij}) start at base value (0.5)
- All biases (h_i) start at 0
- **Beta (inverse temperature) set to 3.0** (more deterministic than original 1.0)
- Each player starts with equal resources

---

## **Turn 1 — Planning Phase (Fast & Strategic)**

### **NEW: Action Card System**

Instead of clicking individual cells, players use **Action Cards**:

#### Each player:
1. **Draws 5 random action cards** from deck
2. **Has token budget:**
   - Base: 3 edge tokens + 2 bias tokens
   - Bonus tokens if controlling territory from previous round
3. **Plays 2-3 cards** (limited by budget)

#### Card Examples:

**Offensive Cards:**
- **⚔️ INFILTRATE** (Cost: 2 bias) - Apply strong bias (+1.5) to 3 adjacent cells
- **💥 DISRUPTION** (Cost: 3 edge) - Weaken 4 edges in a 2x2 region (-0.5 each)

**Defensive Cards:**
- **🛡️ FORTRESS** (Cost: 3 edge) - Strengthen edges in 3x3 region (+0.5 each)
- **⚡ ANCHOR** (Cost: 2 bias) - Strong bias to 1 cell + 4 neighbors

**Special Cards:**
- **🔥 HEAT WAVE** (Cost: 1 special) - Reduce beta locally (add chaos)
- **❄️ FREEZE** (Cost: 1 special) - Increase beta locally (lock state)

#### How to Play:
1. Click card in hand
2. Click region on grid
3. Effect applies instantly (visible preview)
4. Repeat until satisfied
5. Click READY

**Benefits:** Setup time reduced from 3 min → 30 seconds

---

## **Turn 2 — Preview Phase (NEW)**

Before sampling, players can:

1. **Click "PREVIEW" button**
2. **System runs 10 quick samples** (fast, lightweight)
3. **Shows probability heatmap:**
   - Blue intensity = P(cell becomes +1)
   - Red intensity = P(cell becomes -1)
4. **Displays predicted score:** "Player A: 14.2 ± 2.1 cells"
5. **Players can adjust actions** if preview looks bad

**Benefits:** Reduces "I didn't know that would happen" frustration

---

## **Turn 3 — Sampling Phase (Dramatic & Visible)**

When both players press READY:

### **Phase 3.1: Warmup Animation** (3-5 seconds)
- Cells rapidly flicker between red/blue
- Energy graph shows E(x) decreasing
- "System equilibrating..." message
- Gradually stabilizes toward likely states

### **Phase 3.2: Sampling Visualization** (5-7 seconds)
- Show 50 samples as rapid "flashes"
- Cells build up glow intensity based on frequency
- Probability overlay updates: "67% blue, 33% red"
- Players see distribution emerging in real-time

### **Phase 3.3: Final Resolution** (2-3 seconds)
- Cells snap to final majority states
- Territory conquest animation (cells flip like dominoes)
- Energy bars fill up
- Particle effects for dramatic reveal

**Total: 10-15 seconds** (vs. instant in original)

**Benefits:** Creates tension, makes physics visible, satisfying payoff

---

## **Turn 4 — Final Push Phase (NEW)**

After sampling but before scoring:

1. **Each player gets 1 "Claim" token**
2. **Can force 1 cell to their color** (100% guaranteed, no sampling)
3. **Both players act simultaneously**
4. **Then score final board**

**Strategic depth:**
- Use on contested cells to secure close victories
- Can't save you if losing badly
- Creates clutch moments
- Adds skill to critical juncture

**Benefits:** Reduces "I lost to pure RNG" complaints

---

## **Turn 5 — Scoring Phase**

**Simple Tile Majority:**
- Count cells with +1 (Player A) vs -1 (Player B)
- More cells = win the round
- Ties broken by total energy (lower energy wins)

**Additional Victory Conditions:**
- **Total Dominance:** Control all 25 cells → instant win
- **Perfect Harmony:** Achieve >95% probability on all your cells → instant win

---

## **Turn 6 — Round Reset & Resource Management (NEW)**

### **What Persists (Strategic Continuity):**

1. **Biases decay by 50%** (not reset to 0)
   - Example: +2.0 bias → +1.0 next round
   - Rewards building on previous work

2. **Couplings persist fully**
   - All edge modifications carry forward
   - Creates long-term board evolution

3. **Entrenchment bonuses:**
   - Cells controlled for 2+ consecutive rounds get auto-bias (+0.3 per round)
   - Max +0.6 at 3+ rounds
   - Holding territory becomes valuable

4. **Dynamic token income:**
   - Base: 2 edge + 2 bias
   - +1 token per 5 cells controlled
   - +1 bonus if 3+ entrenched cells
   - Winner gets slight resource advantage

### **What Resets:**
- Player ready flags
- Current round increments
- Draw new 5 cards for next round

---

## **Game End**

- **Best of 5 rounds** (unchanged)
- Winner = first to 3 round wins
- If tied at 2-2 after Round 5, play tiebreaker round

---

# 🎯 **Why This Redesign Solves the Problems**

| Problem | Original | Redesigned | Fix |
|---------|----------|------------|-----|
| **Setup too slow** | 2-3 min clicking cells | 30 sec playing cards | ✅ 4x faster |
| **Resolution too fast** | 0.5 sec instant | 10-15 sec animated | ✅ Dramatic tension |
| **No strategic continuity** | Biases reset to 0 | Decay + entrenchment + resources | ✅ Deep strategy |
| **Too much randomness** | Beta=1.0, high variance | Beta=3.0, preview, final push | ✅ Skill matters |
| **Unintuitive UI** | Click + confirm dialogs | Drag & drop, visual feedback | ✅ Smooth interaction |

---

# 📊 **Expected Player Experience**

### Round 1:
- **Setup (30s):** "I'll play Fortress on center + Anchor on corner"
- **Preview:** "Hmm, 62% likely to win... good!"
- **Sampling (12s):** "The cells are stabilizing... YES! 16 cells!"
- **Feeling:** Smart, satisfied, saw physics work

### Round 2:
- **Setup (25s):** "My center is entrenched (+0.3 bonus), I'll invade their side with Infiltrate"
- **Preview:** "Might lose 3 cells... let me adjust with Heat Wave"
- **Sampling (15s):** "High variance in contested zone... 14 cells, but stronger entrenchment"
- **Feeling:** Adapting strategy, making informed decisions

### Round 3:
- **Setup (20s):** "I have more tokens (won Round 2), big push time!"
- **Final Push:** "Close! I'll claim this cell to secure the win!"
- **Result:** "YES! Won by 1 cell! That was clutch!"
- **Feeling:** Skilled play mattered, comeback possible

### Post-Game:
- **"That was intense! Want to try different cards next time"**
- **"I understand coupling now... thick edges mean stable alignment"**
- **Replayability:** ✅ YES

---

# 📐 **Redesigned UI Layout (Card-Based)**

```
┌───────────────────────────────────────────────────────────────┐
│  THRMLHack - Energy Battle       Round 2/5   [Help] [Reset]   │
│  Player A: 2 wins  |  Player B: 1 win                         │
├─────────────┬─────────────────────────────┬───────────────────┤
│             │                             │                   │
│  PLAYER A   │      5×5 GAME GRID          │    PLAYER B       │
│  (Blue)     │                             │    (Red)          │
│             │   [Interactive board with   │                   │
│  ┌────────┐ │    animated cells, edge     │   ┌────────┐     │
│  │ Score  │ │    visualizations, hover    │   │ Score  │     │
│  │ 14/25  │ │    tooltips]                │   │ 11/25  │     │
│  └────────┘ │                             │   └────────┘     │
│             │   Cells show:               │                   │
│  Resources: │   - Color (blue/red/gray)   │   Resources:      │
│  🔗 Edge: 4 │   - Confidence % (overlay)  │   🔗 Edge: 3      │
│  ⚡ Bias: 3 │   - Entrenchment marker     │   ⚡ Bias: 2      │
│             │   - Glow effects            │                   │
│  [PREVIEW]  │                             │   [READY]         │
│  [READY]    │   [Animated during sampling]│                   │
│             │                             │                   │
├─────────────┴─────────────────────────────┴───────────────────┤
│                                                                │
│  YOUR HAND (Current Player):                                  │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌─────┐│
│  │    ⚔️     │ │    🛡️     │ │    ⚡     │ │    🔥     │ │ ❄️  ││
│  │INFILTRATE│ │ FORTRESS │ │  ANCHOR  │ │HEAT WAVE │ │FRZE ││
│  │          │ │          │ │          │ │          │ │     ││
│  │Cost: 2⚡ │ │Cost: 3🔗 │ │Cost: 2⚡ │ │Cost: 1sp │ │Cost:││
│  │Bias 3    │ │Strengthen│ │Bias 1+4  │ │Reduce β  │ │Inc. ││
│  │adjacent  │ │edges 3x3 │ │neighbors │ │in region │ │β   ││
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘ └─────┘│
│                                                                │
│  Click card, then click grid region to play                   │
│                                                                │
├────────────────────────────────────────────────────────────────┤
│  GAME STATUS & FEEDBACK:                                       │
│  ┌────────────────────┬──────────────────┬───────────────────┐│
│  │ Energy: -24.3      │ Magnetization:   │ System Message:   ││
│  │ (lower = stable)   │ +0.12 (A ahead)  │ "Waiting for both ││
│  │                    │                  │  players to ready"││
│  │ [Energy graph____] │ A: ████████ 56%  │                   ││
│  │                    │ B: ████▌    44%  │                   ││
│  └────────────────────┴──────────────────┴───────────────────┘│
└────────────────────────────────────────────────────────────────┘
```

---

## **Main Grid Features (Center Panel)**

### **Cell Visualization:**
- **Size:** 55px × 55px responsive cells
- **Color coding:**
  - Blue = Player A control (+1 spin)
  - Red = Player B control (-1 spin)
  - Gray = neutral/uncertain
  - **Intensity:** Based on confidence % (darker = more certain)
- **Overlays:**
  - White text: Confidence percentage (e.g., "87%")
  - Small badge: Entrenchment level (⭐⭐ = 2 rounds held)
  - Glow effect: Recently modified cells
- **Edges between cells:**
  - Line thickness = coupling strength
  - Thicker = stronger alignment force
  - Animated pulse for modified edges

### **During Sampling Animation:**
- **Phase 1 (Warmup):** Rapid flickering between states
- **Phase 2 (Sampling):** Ghost overlays showing sample distribution
- **Phase 3 (Resolution):** Domino-effect cell flipping, particle bursts

### **Hover Tooltips:**
```
┌─────────────────────────────┐
│  Cell (2, 3)                │
│  ─────────────────────────  │
│  Spin: +1 (Blue - Player A) │
│  Bias: +0.8                 │
│  Entrenchment: 2 rounds     │
│  Neighbors: 4 edges         │
│  ─────────────────────────  │
│  Predicted hold prob: 88%   │
└─────────────────────────────┘
```

---

## **Card Hand (Bottom Panel)**

### **Card Design:**
- **Visual:** Icon + name + description + cost
- **Interaction:**
  - Click to select (card highlights)
  - Click grid region to play
  - Invalid regions shake/highlight red
  - Valid regions glow green on hover
- **Feedback:**
  - Card disappears from hand when played
  - Resources update immediately
  - Grid shows effect preview (transparent overlay)
- **Undo:** Button to undo last played card (before READY)

### **Card Types Available:**

**Offensive (Red border):**
- ⚔️ Infiltrate, 💥 Disruption, 🗡️ Assault, 💣 Sabotage

**Defensive (Blue border):**
- 🛡️ Fortress, ⚡ Anchor, 🏰 Stronghold, 🔒 Monument

**Utility (Purple border):**
- 🔥 Heat Wave, ❄️ Freeze, 🌀 Chaos, 📊 Analysis

**Special (Gold border):**
- 🎯 Claim (Final Push phase only)
- 🔄 Momentum (resource bonus)
- ⚙️ Reconfigure (move biases around)

---

## **Player Panels (Left & Right)**

### **Information Displayed:**
- Player name & color
- Current score (cells controlled)
- Round wins counter
- "YOUR TURN" indicator (pulsing when active)
- Available resources:
  - 🔗 Edge tokens: X/Y
  - ⚡ Bias tokens: X/Y
  - ⭐ Special tokens: X/Y
- **PREVIEW button:** Run quick sampling preview
- **READY button:** Confirm actions and proceed

### **Visual Enhancements:**
- Player avatar/character sprite
- Animated glow when active player
- Resource tokens shown as draggable chips (alternative interaction)
- Win streak indicator (flames if winning 2+ in a row)

---

## **Bottom Status Bar**

### **System Metrics (Always Visible):**
- **Energy graph:** Line chart showing E(x) over recent samples
- **Magnetization:** Average spin (-1 to +1), bar chart
- **Territory distribution:** A vs B percentage bars
- **Phase indicator:** "Planning / Sampling / Scoring"

### **Message Log:**
- Recent actions from both players
- System events ("Sampling complete", "Player A won round")
- Combo notifications ("COMBO: Fortress + Anchor = Stronghold!")
- Color-coded by player/system

---

## **Interaction Improvements Over Original**

| Original | Redesigned |
|----------|------------|
| Click cell → browser confirm dialog | Click card → click region (smooth) |
| No preview of effects | Preview button shows probabilities |
| Instant sampling (no tension) | 10-15 sec animated sampling |
| Can't undo mistakes | Undo button before READY |
| No visual coupling strength | Edge thickness shows strength |
| No entrenchment visible | Star badges show held rounds |
| No combo feedback | Special effects for combos |
| Static grid | Animated, glowing, responsive |

---

## **Responsive Design Notes**

- **Desktop (1200px+):** Full 3-column layout
- **Tablet (768-1199px):** Stack panels vertically, grid stays centered
- **Mobile (< 768px):** Single column, card hand scrolls horizontally
- **Grid scales:** 55px cells on desktop → 40px on mobile

---

## **Accessibility Features**

- **Color blind mode:** Pattern overlays (stripes/dots) in addition to colors
- **Screen reader support:** All cells and cards have aria-labels
- **Keyboard navigation:** Tab through cards, arrow keys for grid
- **High contrast mode:** Toggle for increased visibility

---

## **Visual Polish**

- **Animations:** Smooth CSS transitions (0.3s ease)
- **Particle effects:** Confetti on round wins, sparks on sampling
- **Sound effects (optional):** Card play, sampling whoosh, victory fanfare
- **Background:** Subtle animated energy field pattern
- **Typography:** Bold headers, readable body text (16px minimum)
