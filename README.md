# Adaptive Multi-Agent Task Allocation via Reinforcement Learning
### KnowledgeQuarry 2026 · Convoke 8.0 · CIC University of Delhi · ML Engineering Track

---

## What This Is

A Reinforcement Learning system where 5 agents **learn** to complete 10 tasks on a grid as efficiently as possible — purely through trial and error across 95,000+ episodes. No rules. No hardcoding. The policy discovers everything from reward signals alone.

**The scenario:** Think of a warehouse with 5 workers and 10 jobs scattered across a floor. Workers have different strengths:
- **■ Type A agents (2)** — Strong but slow. High capacity means cheaper energy cost per task. Best for hard tasks.
- **● Type B agents (3)** — Fast but weak. Low capacity means higher energy cost per task. Best for nearby easy tasks.

The model learns *which agent should go to which task, in what order* — and the correct assignment strategy emerges entirely from experience.

---

## The Learning Story

| Policy | Avg Reward | vs Random |
|--------|-----------|-----------|
| Random (no learning) | 21.92 | baseline |
| Early — 30k steps | 26.26 | +19.8% |
| Mid — 100k steps | 26.37 | +20.3% |
| Final — 300k steps | 26.40 | +20.4% |

All policies complete 100% of tasks. The learning signal is **reward efficiency** — the trained policy completes tasks in 3 steps consistently, wastes less energy, and never overlaps agent assignments. The random policy takes up to 80 steps, doubles up on tasks, and drains energy on long unnecessary moves.

---

## Dashboard

### 🎬 Episode Replay — Watch Agents Learn
Animated grid with play/pause. Watch agents fan out and complete tasks in 3 steps. Step-by-step mode shows a colour-coded action log explaining exactly what each agent did and why.

![Episode Replay](screenshots/1.png)

*Animated view — agents moving to tasks, green = pending, dark = completed*

![Step-by-Step with Narration](screenshots/2.png)

*Step-by-step mode — narration box updates every step showing agent decisions*

---

### ⚔️ Trained vs Random — Same Grid, Different Brain
Both policies run on **identical starting positions and identical tasks** (same random seed). The only difference is the decision-making. Slider controls both simultaneously.

![Trained vs Random](screenshots/4.png)

*Left = trained policy (organised, parallel). Right = random policy (chaotic, overlapping).*

![Cumulative Reward Chart](screenshots/5.png)

*Green line (trained) consistently above red dotted line (random) — every step above = a smarter decision.*

---

### 📈 Proof of Learning
Bar charts, evaluation episode lines, and training curve across 95,000+ episodes.

![Proof of Learning](screenshots/6.png)

*Left: avg reward per policy with error bars. Right: reward across 20 eval episodes — trained is higher and more consistent.*

![Training Curve](screenshots/7.png)

*Rising reward trend + narrowing confidence band = policy converging from random exploration to stable strategy.*

---

### 🎛️ What-If Explorer — Stress Test the Policy
Change agent count, energy limits, and task count. Run the trained model on configurations it was never trained on. Tests generalisation.

![What-If Explorer](screenshots/8.png)

*3 agents, 15 tasks — harder than training. Policy still allocates efficiently.*

---

## How Energy and Capacity Work

| Agent | Type | Speed | Capacity | Energy | Best For |
|-------|------|-------|----------|--------|----------|
| Agent 0 | A | 1 | 3 | 100 | Hard tasks (energy cost ÷3) |
| Agent 1 | A | 1 | 3 | 100 | Hard tasks (energy cost ÷3) |
| Agent 2 | B | 2 | 1 | 60 | Nearby easy tasks |
| Agent 3 | B | 2 | 1 | 60 | Nearby easy tasks |
| Agent 4 | B | 2 | 1 | 60 | Nearby easy tasks |

**Move cost** = `distance / speed` — Type B moves cheaper per unit distance  
**Task cost** = `difficulty / capacity` — Type A completes cheaper per difficulty point  
**Type A on difficulty-3 task:** `3/3 = 1` energy  
**Type B on difficulty-3 task:** `3/1 = 3` energy — 3× more expensive

The policy learned to exploit this asymmetry without being told about it.

---

## Why MaskablePPO?

Standard PPO with `MultiDiscrete` action space (5 agents × 21 options) = ~4M combinations. During initial experiments, the model completely failed to converge — `explained_variance` stayed near 0 over 500,000 steps.

**Two fixes applied:**

1. **Action masking** — at each step, completed tasks and zero-energy agents are masked from the valid action set. The policy only ever sees actions that make sense. Cleaner gradient, dramatically faster convergence.

2. **Dense reward shaping** — proximity bonus (+0.3 per step for closing distance to nearest task), time penalty (-0.02/step), early completion bonus (+5.0). Gives signal every step instead of only on completion.

Result: convergence within 30,000 steps instead of failure at 500,000.

---

## Stack

```
gymnasium==0.29.1       # Custom RL environment
stable-baselines3==2.3.0  # PPO backbone
sb3-contrib==2.3.0      # MaskablePPO + ActionMasker
torch>=2.0.0            # Neural network backend (CUDA)
streamlit>=1.32.0       # Dashboard
plotly>=5.18.0          # Animated visualisations
pandas>=2.0.0           # Metrics logging
numpy>=1.24.0           # State handling
```

---

## Project Structure

```
knowledgequarry/
├── environment.py      # Custom Gymnasium env — grid, agents, tasks, rewards, action masking
├── train.py            # MaskablePPO training loop — checkpoints every 30k/100k/200k/300k steps
├── evaluate.py         # Evaluation — generates replay_data.json + comparison_data.json
├── dashboard.py        # Streamlit dashboard — 4 pages
├── models/             # PPO checkpoints (auto-created on train)
├── logs/               # Training CSV + replay JSON (auto-created on train)
├── screenshots/        # Dashboard screenshots for this README
├── requirements.txt
└── README.md
```

---

## Setup and Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train (~90 mins on GPU, ~3hrs CPU)
python train.py

# 3. Generate evaluation data and replay files
python evaluate.py

# 4. Launch dashboard
streamlit run dashboard.py
```

Models and logs are gitignored (large binary files). Run steps 2-3 to regenerate them.

---

## Key Results Summary

- **300,000 timesteps** trained across **95,030 episodes**
- **+20.4% reward improvement** over random baseline
- **3 steps** average to complete all 10 tasks (vs 80 step limit)
- **100% task completion rate** across all evaluation episodes
- **Explained variance = 1.0** — value network fully converged
- **4 policy checkpoints** saved for before/after comparison

---

## Submitted by

B.Tech CSE (AI & ML) · JSS University, Noida  
Team Sigmoid