# Adaptive Multi-Agent Task Allocation
### KnowledgeQuarry 2026 · Convoke 8.0 · CIC University of Delhi · ML Engineering Track

---

## What This Is

A Reinforcement Learning system where 5 agents **learn** to complete 10 tasks on a grid as efficiently as possible — purely through trial and error, no hardcoded rules.

Think of it as a warehouse with 5 workers and 10 jobs. Workers have different strengths:
- **Type A (2 agents)** — Strong but slow. High capacity means cheaper energy cost per task. Best for hard tasks.
- **Type B (3 agents)** — Fast but weak. Low capacity means higher energy cost per task. Best for nearby easy tasks.

The RL model (MaskablePPO) learns *which agent should go to which task, in what order* — trained over **300,000 timesteps / 95,000+ episodes**.

---

## Problem Statement

**Problem 02 — Adaptive Systems: Learning Under Constraints**

- Environment: 10 tasks scattered on a 10×10 grid, each with a difficulty rating (1–3)
- Agents: 5 agents with different speed, capacity, and energy limits
- Objective: Maximize tasks completed while minimizing energy waste
- Constraint: Agents run out of energy — they can't do everything

**What the model learns:**
> Random policy completes tasks but wastes energy and overlaps assignments.
> Trained policy splits tasks intelligently — Type A handles hard tasks, Type B covers nearby easy ones.
> Result: **+20% reward improvement** over random baseline across 20 evaluation episodes.

---

## How Energy & Capacity Work

| Agent | Type | Speed | Capacity | Energy |
|-------|------|-------|----------|--------|
| Agent 0 | A | 1 | 3 | 100 |
| Agent 1 | A | 1 | 3 | 100 |
| Agent 2 | B | 2 | 1 | 60 |
| Agent 3 | B | 2 | 1 | 60 |
| Agent 4 | B | 2 | 1 | 60 |

- **Move cost** = `distance / speed`
- **Task cost** = `difficulty / capacity`
- Type A at difficulty-3 task: `3/3 = 1` energy — very efficient
- Type B at difficulty-3 task: `3/1 = 3` energy — 3× more expensive

This asymmetry is what the RL model learns to exploit.

---

## Stack

- `gymnasium` — custom RL environment
- `stable-baselines3` + `sb3-contrib` — MaskablePPO (action masking prevents agents from targeting completed tasks)
- `streamlit` — interactive dashboard
- `plotly` — animated simulation + training curves
- `pandas / numpy` — metrics logging

---

## Results

| Policy | Avg Reward | Avg Completion |
|--------|-----------|----------------|
| Random | 21.9 | 100% |
| Early (30k steps) | 26.3 | 100% |
| Mid (100k steps) | 26.4 | 100% |
| Final (300k steps) | 26.4 | 100% |

All policies complete 100% of tasks — the learning signal is **reward efficiency** (less energy wasted, fewer redundant moves). The trained policy achieves **+20% reward** over random.

---

## Project Structure

```
knowledgequarry/
├── environment.py      # Custom Gymnasium env — grid, agents, tasks, rewards
├── train.py            # MaskablePPO training loop with checkpointing
├── evaluate.py         # Evaluation + replay data generation for dashboard
├── dashboard.py        # Streamlit dashboard — simulation, curves, comparison, what-if
├── models/             # Saved PPO checkpoints (auto-created on train)
├── logs/               # Training metrics CSV + replay JSON (auto-created on train)
├── requirements.txt
└── README.md
```

---

## Setup & Run

```bash
# Install dependencies
pip install -r requirements.txt

# Train the model (~90 mins on GPU, ~3hrs CPU)
python train.py

# Evaluate and generate dashboard data
python evaluate.py

# Launch dashboard
streamlit run dashboard.py
```

---

## Dashboard Tabs

| Tab | What it shows |
|-----|--------------|
| 🎬 Episode Replay | Animated grid — watch agents move and complete tasks. Step-by-step action log explains every decision. |
| 📈 Training Curves | Reward per episode, reward per step, energy used, policy stability over 95k+ episodes |
| 🏆 Model Comparison | Random vs early vs mid vs final policy — reward progression across 20 eval episodes |
| 🎛️ What-If Explorer | Change agent count, energy limits, task count — run trained model live on new scenarios |

---

## Key Technical Decisions

**Why MaskablePPO over vanilla PPO?**
Standard PPO with `MultiDiscrete` action space of 5 agents × 21 options = 4M+ combinations. The model was wasting gradient on invalid actions (targeting already-completed tasks). Action masking eliminates invalid actions at inference time — massive improvement in convergence.

**Why dense reward shaping?**
Sparse reward (only +1 on task completion) means agents wander for many steps with zero signal. Added proximity bonus (+0.3 per step for moving closer to nearest task), time penalty (-0.02/step), and early completion bonus (+5.0) to give the policy constant gradient signal.

**Why 10 tasks instead of 20?**
5 agents solving 20 tasks is too large for PPO to crack in 300k steps without curriculum learning. 10 tasks allows the policy to converge fully and demonstrate clear learned behaviour within the competition timeframe.

---

## Team

**Rehaan Ahmad Khan** — B.Tech CS (AI & ML), JSS University Noida
- GitHub: [RAK2315](https://github.com/RAK2315)
- Email: rehaanahmadkhan178@gmail.com
- LinkedIn: [linkedin.com/in/rehaanak](https://linkedin.com/in/rehaanak)
