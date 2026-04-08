# Adaptive Multi-Agent Task Allocation via Reinforcement Learning
### KnowledgeQuarry 2026 · Convoke 8.0 · CIC University of Delhi · ML Engineering Track

---

## 🔗 Links

| Resource | Link |
|----------|------|
| 🌐 **Live Dashboard** | [Open on Streamlit Cloud](https://DUMMY_STREAMLIT_LINK) |
| 📄 **Detailed Report (PDF)** | [View on Google Drive](https://DUMMY_GDRIVE_PDF_LINK) |
| 📊 **Presentation (PPT)** | [View on Google Drive](https://DUMMY_GDRIVE_PPT_LINK) |
| 💻 **GitHub Repository** | [RAK2315/knowledgequarry-2026](https://github.com/RAK2315/knowledgequarry-2026) |

---

## 👨‍⚖️ For Judges — What to Look At

### Quickest way (2 minutes):
1. Open the **Live Dashboard** link above — no installation needed
2. Go to **⚔️ Trained vs Random** — drag the slider and watch both grids simultaneously
3. Go to **📈 Proof of Learning** — see the +20% reward improvement numbers at the top
4. Go to **🎬 Episode Replay** — select `final` policy, hit ▶ Play

### If you want to run it locally:
```bash
pip install -r requirements.txt
python evaluate.py        # generates replay data (2 mins)
streamlit run dashboard.py
```
> Note: `python train.py` takes ~90 mins (GPU) and is not required — evaluation data is already in `logs/`.

---

## What This Is

A Reinforcement Learning system where 5 agents **learn** to complete 10 tasks on a grid as efficiently as possible — purely through trial and error across 95,000+ episodes. No rules. No hardcoding. The policy discovers everything from reward signals alone.

**The scenario:** Think of a warehouse with 5 workers and 10 jobs scattered across a floor. Workers have different strengths:
- **■ Type A agents (2)** — Strong but slow. High capacity = cheaper energy cost per task. Best for hard tasks.
- **● Type B agents (3)** — Fast but weak. Low capacity = higher energy cost per task. Best for nearby easy tasks.

The model learns *which agent should go to which task, in what order* — and the correct assignment strategy emerges entirely from experience.

---

## The Learning Story

| Policy | Avg Reward | vs Random |
|--------|-----------|-----------|
| Random (no learning) | 21.92 | baseline |
| Early — 30k steps | 26.26 | +19.8% |
| Mid — 100k steps | 26.37 | +20.3% |
| Final — 300k steps | 26.40 | +20.4% |

All policies complete 100% of tasks. The learning signal is **reward efficiency** — the trained policy completes tasks in **3 steps** consistently, wastes less energy, and never overlaps agent assignments. The random policy takes up to 80 steps, doubles up on tasks, and drains energy on unnecessary moves.

---

## Dashboard

### 🎬 Episode Replay — Watch Agents Learn
Animated grid with play/pause. Watch agents fan out and complete tasks in 3 steps. Step-by-step mode shows a colour-coded action log explaining exactly what each agent did and why.

![Episode Replay Animated](images/1.png)
*Animated view — agents moving to tasks. Green diamonds = pending, dark = completed.*

![Step-by-Step Narration](images/2.png)
*Step-by-step mode — narration box updates every step showing each agent's decision and how energy/capacity affected it.*

---

### ⚔️ Trained vs Random — Same Grid, Different Brain
Both policies run on **identical starting positions and identical task layouts** (same random seed). Only the decision-making differs. One slider controls both grids simultaneously.

![Trained vs Random](images/4.png)
*Left = trained policy (parallel, no overlap). Right = random policy (chaotic, redundant moves).*

![Cumulative Reward Chart](images/5.png)
*Green line (trained) above red dotted line (random) every step — each gap = a smarter decision.*

---

### 📈 Proof of Learning
Bar charts, per-episode reward lines, and training curve across 95,000+ episodes with checkpoint markers.

![Proof of Learning Charts](images/6.png)
*Left: avg reward per policy with error bars. Right: reward consistency across 20 eval episodes.*

![Training Curve](images/7.png)
*Rising trend + narrowing confidence band = policy converging from random exploration to stable strategy.*

---

### 🎛️ What-If Explorer — Stress Test the Policy
Change agent count, energy limits, and task count live. Run the trained model on configurations it was never trained on. Tests generalisation.

![What-If Explorer](images/8.png)
*3 agents, 15 tasks — harder than training. Policy still allocates tasks efficiently.*

---

## How Energy and Capacity Work

| Agent | Type | Speed | Capacity | Energy | Best For |
|-------|------|-------|----------|--------|----------|
| Agent 0 | A | 1 | 3 | 100 | Hard tasks (cost ÷ 3) |
| Agent 1 | A | 1 | 3 | 100 | Hard tasks (cost ÷ 3) |
| Agent 2 | B | 2 | 1 | 60 | Nearby easy tasks |
| Agent 3 | B | 2 | 1 | 60 | Nearby easy tasks |
| Agent 4 | B | 2 | 1 | 60 | Nearby easy tasks |

**Move cost** = `distance / speed`
**Task cost** = `difficulty / capacity`

Type A on difficulty-3 task: `3/3 = 1 energy`
Type B on difficulty-3 task: `3/1 = 3 energy` — 3× more expensive

The policy learned to exploit this asymmetry without being told about it.

---

## Why MaskablePPO?

Standard PPO with `MultiDiscrete` action space (5 agents × 21 options) ≈ 4M combinations. Initial experiments with standard PPO completely failed — `explained_variance` stayed near 0 over 500,000 steps.

**Two fixes that worked:**

**1. Action masking** — completed tasks and zero-energy agents are removed from the valid action set each step. The policy only ever sees actions that make sense. Cleaner gradient, dramatically faster convergence.

**2. Dense reward shaping** — proximity bonus (+0.3/step for closing distance), time penalty (-0.02/step), early completion bonus (+5.0). Signal every step instead of only on completion.

Result: convergence within 30,000 steps instead of failure at 500,000.

---

## Project Structure

```
knowledgequarry/
├── environment.py      # Custom Gymnasium env — grid, agents, tasks, rewards, masking
├── train.py            # MaskablePPO training — checkpoints at 30k/100k/200k/300k steps
├── evaluate.py         # Generates replay_data.json + comparison_data.json for dashboard
├── dashboard.py        # Streamlit dashboard — 4 pages
├── models/             # PPO checkpoints (gitignored — run train.py to regenerate)
├── logs/               # Training CSV + replay JSONs (committed for dashboard)
├── images/             # Dashboard screenshots
├── requirements.txt
└── README.md
```

---

## Setup and Run

```bash
# Install dependencies
pip install -r requirements.txt

# Optional: retrain from scratch (~90 mins on GPU)
python train.py

# Generate evaluation data (required if retraining, already present otherwise)
python evaluate.py

# Launch dashboard
streamlit run dashboard.py
```

---

## Key Numbers

| Metric | Value |
|--------|-------|
| Timesteps trained | 300,000 |
| Episodes completed | 95,030 |
| Reward improvement vs random | +20.4% |
| Avg steps to complete all tasks | 3 |
| Task completion rate | 100% |
| Explained variance (final) | 1.0 |
| Checkpoints saved | 4 |

---

## Stack

```
gymnasium==0.29.1         custom RL environment
stable-baselines3==2.3.0  PPO backbone
sb3-contrib==2.3.0        MaskablePPO + ActionMasker
torch>=2.0.0              neural network backend (CUDA)
streamlit>=1.32.0         dashboard
plotly>=5.18.0            animated visualisations
pandas>=2.0.0             metrics logging
numpy>=1.24.0             state handling
```

---

## Submitted by

**Rehaan Ahmad Khan**
B.Tech CSE (AI & ML) · JSS University, Noida
Team: Sigmoid