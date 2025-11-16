# 🧪 Quick Test Guide - Phase 1 Improvements

## Start the Game

### Terminal 1: Backend
```bash
cd /Users/hayley/Projects/thrmlhack
python -m src.main
```
**Expected:** Server running at `http://localhost:8000`

### Terminal 2: Frontend
```bash
cd /Users/hayley/Projects/thrmlhack/web
npm run dev
```
**Expected:** App running at `http://localhost:5173`

---

## Test Checklist

### ✅ Test 1: No More Confirm Dialogs (UI Improvement)
**Before:** Annoying confirm popups on every click
**After:** Smooth, instant feedback

**Steps:**
1. Open game in browser
2. Make sure "Player A" is selected
3. Click any cell on the grid (bias mode)
4. **Expected:** Cell immediately turns blue (no popup!)
5. Click another cell
6. **Expected:** Instant feedback, no waiting

**Result:** ⬜ PASS / ⬜ FAIL

---

### ✅ Test 2: Preview Mode (See the Future!)
**Before:** No way to predict outcomes
**After:** 🔮 Preview button shows probabilities

**Steps:**
1. Player A: Click 3-5 cells on the left side
2. Click the purple "🔮 Preview Outcome" button
3. **Expected:**
   - Grid cells show probability gradient (red → purple → blue)
   - Numbers show % probability
   - Summary appears: "Player A: 14.2 cells, Player B: 10.8 cells (Confidence: 87%)"
4. Click Preview again with different setup
5. **Expected:** Different predictions based on your changes

**Result:** ⬜ PASS / ⬜ FAIL

---

### ✅ Test 3: Animated Sampling (Drama!)
**Before:** Instant, boring resolution
**After:** 2-second wiggle animation

**Steps:**
1. Both players click "Ready"
2. Click "⚡ Run Sampling"
3. **Watch carefully!**
4. **Expected:**
   - Cells wiggle/rotate for ~2 seconds
   - Grid pulses
   - Then snap to final state with confidence percentages
5. **Feel:** Satisfying, dramatic, you can "see" the physics working

**Result:** ⬜ PASS / ⬜ FAIL

---

### ✅ Test 4: Higher Beta = More Predictable (Skill > Luck)
**Before:** Beta=1.0, outcomes felt random
**After:** Beta=3.0, outcomes match your strategy

**Steps:**
1. **Round 1:**
   - Player A: Click ALL 5 cells in the top row
   - Player A: Click ALL 5 cells in the 2nd row
   - Player B: Don't click anything
   - Both ready → Run Sampling
2. **Expected:** Player A should win with ~20+ cells (very high confidence)
3. **Repeat test 3 times**
4. **Expected:** Consistent results (not random)

**Result:** ⬜ PASS / ⬜ FAIL

---

### ✅ Test 5: Bias Decay (Strategic Continuity)
**Before:** Round 1 actions → worthless in Round 2
**After:** 50% of bias carries forward

**Steps:**
1. **Round 1:**
   - Player A: Click cells (2,2), (2,3), (3,2), (3,3) multiple times (build strong bias)
   - Run Sampling → Next Round
2. **Round 2 - DON'T TOUCH ANYTHING:**
   - Click "🔮 Preview Outcome" immediately
   - **Expected:** The cells you biased in Round 1 still show blue probability!
   - Check the actual bias values (they should be ~50% of Round 1)
3. **Before:** Cells would be gray (bias = 0)
4. **After:** Cells still favor Player A (bias = 0.5x previous)

**Result:** ⬜ PASS / ⬜ FAIL

---

### ✅ Test 6: Confidence Visualization
**Before:** All cells looked the same
**After:** Confident cells are brighter, uncertain cells are dim

**Steps:**
1. Create mixed scenario:
   - Player A: Click center cell (2,2) 2 times (strong bias)
   - Player A: Click corner cell (0,0) 1 time (weak bias)
   - Player B: Click opposite corner (4,4) 2 times
2. Both ready → Run Sampling
3. **Expected:**
   - Strongly biased cells: Bright, solid color + high % (90-100%)
   - Weakly biased cells: Dim color + low % (50-70%)
   - Contested cells: Medium color + medium % (60-80%)

**Result:** ⬜ PASS / ⬜ FAIL

---

### ✅ Test 7: Preview Accuracy
**Before:** N/A (no preview existed)
**After:** Preview should closely match actual outcome

**Steps:**
1. Player A: Click 5 cells randomly
2. Player B: Click 5 different cells randomly
3. Click "🔮 Preview Outcome"
4. **Note the prediction:** e.g., "A: 14.2, B: 10.8"
5. Both ready → Run Sampling
6. **Check actual result:** e.g., "A: 15, B: 10"
7. **Expected:** Within ±2 cells of prediction (with beta=3.0, should be close!)

**Result:** ⬜ PASS / ⬜ FAIL

---

## 🎯 Overall Experience Test

Play 3 full rounds and answer:

1. **Setup feels faster?** ⬜ YES / ⬜ NO
2. **No annoying popups?** ⬜ YES / ⬜ NO
3. **Sampling feels dramatic?** ⬜ YES / ⬜ NO
4. **Outcomes feel less random?** ⬜ YES / ⬜ NO
5. **Previous rounds still matter?** ⬜ YES / ⬜ NO
6. **You understand the physics better?** ⬜ YES / ⬜ NO
7. **You want to play again?** ⬜ YES / ⬜ NO

**If 5+ YES:** Phase 1 is a success! 🎉
**If 3-4 YES:** Good progress, needs tweaking
**If <3 YES:** Something's wrong, check implementation

---

## 🐛 Troubleshooting

### Backend won't start
```bash
# Check if already running
lsof -ti:8000 | xargs kill -9

# Reinstall dependencies
pip install -r requirements.txt
```

### Frontend won't start
```bash
cd web
rm -rf node_modules package-lock.json
npm install
npm run dev
```

### Preview button does nothing
- Open browser console (F12)
- Check for errors
- Verify backend is running (`http://localhost:8000`)

### Animation doesn't show
- Clear browser cache (Cmd+Shift+R)
- Check CSS loaded correctly
- Verify `isAnimating` state is being set

---

## 📸 What Good Output Looks Like

### Console Output (Backend):
```
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### Console Output (Frontend):
```
VITE v5.x.x  ready in XXX ms

➜  Local:   http://localhost:5173/
➜  Network: use --host to expose
```

### Browser Behavior:
- ✅ Cells change color instantly (no popups)
- ✅ Preview shows purple gradient
- ✅ Sampling triggers wiggle animation
- ✅ Confidence percentages show in cells
- ✅ Messages appear at top: "Preview generated!", "Sampling completed!"

---

## 🎮 Try This Fun Test

**"Can you predict the outcome perfectly?"**

1. Create a simple pattern:
   - Player A: Click all 9 cells in the top-left 3x3 grid (2-3 clicks each)
   - Player B: Click all 9 cells in the bottom-right 3x3 grid (2-3 clicks each)
2. Click Preview
3. **Expected:**
   - Top-left: 100% blue
   - Bottom-right: 100% red
   - Middle row: Mixed probabilities
4. Run Sampling
5. **Expected:** Should match preview almost exactly (beta=3.0 is very deterministic!)

If this works → You've successfully made the game skill-based! 🏆

---

## ✨ Phase 1 Complete!

All tests passing? **Congratulations!** You've successfully implemented:
- ✅ More deterministic gameplay (beta=3.0)
- ✅ Strategic continuity (bias decay)
- ✅ Informed decisions (preview mode)
- ✅ Dramatic feedback (animations)
- ✅ Smooth interaction (no dialogs)
- ✅ Visual physics (confidence colors)

**The game should now feel 10x better!** 🚀

Next: Consider Phase 2 features (entrenchment, dynamic resources, card system) if you want even more strategic depth.
