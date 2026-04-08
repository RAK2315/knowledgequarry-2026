import os
import json
import numpy as np
import pandas as pd
from sb3_contrib import MaskablePPO
from environment import TaskAllocationEnv, AGENT_CONFIGS, NUM_TASKS

MODEL_DIR = "models"
LOG_DIR = "logs"
NUM_EVAL_EPISODES = 20
COMPARISON_SEED = 42
os.makedirs(LOG_DIR, exist_ok=True)


def describe_action(agent_idx, action, env, prev_energy, new_energy, task_completed):
    cfg = AGENT_CONFIGS[agent_idx] if agent_idx < len(AGENT_CONFIGS) else {"type": "?", "speed": 1, "capacity": 1}
    agent_type = cfg["type"]
    speed = cfg["speed"]
    capacity = cfg["capacity"]

    if action == env.num_tasks:
        return f"Agent {agent_idx} (Type {agent_type}) — stayed idle (no valid task)"

    task_pos = env.task_positions[action]
    energy_spent = round(prev_energy - new_energy, 1)

    if task_completed:
        diff = int(env.task_difficulty[action])
        speed_note = "moved quickly" if speed == 2 else "moved steadily"
        cap_note = f"high capacity (x{capacity}) reduced energy cost" if capacity > 1 else "standard capacity"
        return (
            f"Agent {agent_idx} (Type {agent_type}) → Task {action} at ({int(task_pos[0])},{int(task_pos[1])}) "
            f"[difficulty {diff}] ✅ Completed — {speed_note}, {cap_note}, used {energy_spent} energy"
        )
    else:
        return (
            f"Agent {agent_idx} (Type {agent_type}) → Task {action} at ({int(task_pos[0])},{int(task_pos[1])}) "
            f"— moved but insufficient energy to complete (used {energy_spent} energy)"
        )


def run_episode(model, env, seed=None, deterministic=True):
    obs, _ = env.reset(seed=seed)
    frames = []
    total_reward = 0.0
    done = False
    prev_energy = env.agent_energy.copy()
    prev_task_done = env.task_done.copy()
    cumulative_reward = 0.0

    while not done:
        masks = env.action_masks()
        action, _ = model.predict(obs, deterministic=deterministic, action_masks=masks)
        action = np.clip(action, 0, env.num_tasks)

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        cumulative_reward += reward
        done = terminated or truncated

        new_energy = env.agent_energy.copy()
        new_task_done = env.task_done.copy()
        newly_completed = np.where((new_task_done - prev_task_done) > 0)[0]

        action_logs = []
        for i, a in enumerate(action):
            task_completed = a in newly_completed
            log = describe_action(i, a, env, prev_energy[i], new_energy[i], task_completed)
            action_logs.append(log)

        step_summary = f"Step {env._step}: {int(new_task_done.sum())}/10 tasks done"
        if len(newly_completed) > 0:
            step_summary += f" — {len(newly_completed)} task(s) completed this step ✅"
        else:
            step_summary += " — agents repositioning"

        frame = {
            "step": env._step,
            "agent_positions": env.agent_positions.tolist(),
            "agent_energy": new_energy.tolist(),
            "task_positions": env.task_positions.tolist(),
            "task_done": new_task_done.tolist(),
            "task_difficulty": env.task_difficulty.tolist(),
            "tasks_completed": int(new_task_done.sum()),
            "reward": float(reward),
            "cumulative_reward": float(cumulative_reward),
            "action_logs": action_logs,
            "step_summary": step_summary,
            "actions": action.tolist(),
        }
        frames.append(frame)
        prev_energy = new_energy.copy()
        prev_task_done = new_task_done.copy()

    summary = env.get_episode_summary()
    summary["total_reward"] = total_reward
    return frames, summary


def run_random_episode(env, seed=None):
    obs, _ = env.reset(seed=seed)
    frames = []
    total_reward = 0.0
    done = False
    prev_energy = env.agent_energy.copy()
    prev_task_done = env.task_done.copy()
    cumulative_reward = 0.0

    while not done:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        cumulative_reward += reward
        done = terminated or truncated

        new_energy = env.agent_energy.copy()
        new_task_done = env.task_done.copy()
        newly_completed = np.where((new_task_done - prev_task_done) > 0)[0]

        action_logs = []
        for i, a in enumerate(action):
            task_completed = a in newly_completed
            log = describe_action(i, a, env, prev_energy[i], new_energy[i], task_completed)
            action_logs.append(log)

        step_summary = f"Step {env._step}: {int(new_task_done.sum())}/10 tasks done"
        if len(newly_completed) > 0:
            step_summary += f" — {len(newly_completed)} task(s) completed this step ✅"
        else:
            step_summary += " — agents moving randomly"

        frame = {
            "step": env._step,
            "agent_positions": env.agent_positions.tolist(),
            "agent_energy": new_energy.tolist(),
            "task_positions": env.task_positions.tolist(),
            "task_done": new_task_done.tolist(),
            "task_difficulty": env.task_difficulty.tolist(),
            "tasks_completed": int(new_task_done.sum()),
            "reward": float(reward),
            "cumulative_reward": float(cumulative_reward),
            "action_logs": action_logs,
            "step_summary": step_summary,
            "actions": action.tolist(),
        }
        frames.append(frame)
        prev_energy = new_energy.copy()
        prev_task_done = new_task_done.copy()

    summary = env.get_episode_summary()
    summary["total_reward"] = total_reward
    return frames, summary


def evaluate_model(model_path, label, num_episodes=NUM_EVAL_EPISODES):
    if not os.path.exists(model_path + ".zip"):
        print(f"[SKIP] {model_path} not found")
        return None, None

    model = MaskablePPO.load(model_path)
    env = TaskAllocationEnv()
    all_summaries = []
    best_frames = None
    best_reward = -999999

    for ep in range(num_episodes):
        frames, summary = run_episode(model, env)
        summary["episode"] = ep + 1
        summary["model"] = label
        all_summaries.append(summary)
        if summary["total_reward"] > best_reward:
            best_reward = summary["total_reward"]
            best_frames = frames

    print(f"[{label}] Avg tasks: {np.mean([s['tasks_completed'] for s in all_summaries]):.2f} / {NUM_TASKS}")
    print(f"[{label}] Avg reward: {np.mean([s['total_reward'] for s in all_summaries]):.2f}")
    print(f"[{label}] Avg completion: {np.mean([s['completion_rate'] for s in all_summaries]):.2%}")
    return all_summaries, best_frames


def run_random_baseline(num_episodes=NUM_EVAL_EPISODES):
    env = TaskAllocationEnv()
    all_summaries = []
    for ep in range(num_episodes):
        frames, summary = run_random_episode(env)
        summary["episode"] = ep + 1
        summary["model"] = "random"
        all_summaries.append(summary)
    print(f"[RANDOM] Avg tasks: {np.mean([s['tasks_completed'] for s in all_summaries]):.2f} / {NUM_TASKS}")
    return all_summaries


def generate_comparison_episode(model):
    env = TaskAllocationEnv()
    print(f"\n=== Generating same-seed comparison (seed={COMPARISON_SEED}) ===")

    trained_frames, trained_summary = run_episode(model, env, seed=COMPARISON_SEED)
    print(f"[TRAINED] reward: {trained_summary['total_reward']:.2f} | steps: {trained_summary['total_steps']}")

    random_frames, random_summary = run_random_episode(env, seed=COMPARISON_SEED)
    print(f"[RANDOM]  reward: {random_summary['total_reward']:.2f} | steps: {random_summary['total_steps']}")

    return {
        "trained": trained_frames,
        "random": random_frames,
        "trained_summary": trained_summary,
        "random_summary": random_summary,
    }


def main():
    checkpoints = {
        "early_30k": os.path.join(MODEL_DIR, "ppo_checkpoint_30000"),
        "mid_100k": os.path.join(MODEL_DIR, "ppo_checkpoint_100000"),
        "late_200k": os.path.join(MODEL_DIR, "ppo_checkpoint_200000"),
        "final": os.path.join(MODEL_DIR, "ppo_final"),
    }

    all_summaries = []
    replay_data = {}

    print("\n=== Random Baseline ===")
    random_summaries = run_random_baseline()
    all_summaries.extend(random_summaries)

    final_model = None
    for label, path in checkpoints.items():
        print(f"\n=== Evaluating: {label} ===")
        summaries, best_frames = evaluate_model(path, label)
        if summaries:
            all_summaries.extend(summaries)
            replay_data[label] = best_frames
            if label == "final":
                final_model = MaskablePPO.load(path)

    df = pd.DataFrame(all_summaries)
    df.to_csv(os.path.join(LOG_DIR, "eval_metrics.csv"), index=False)
    print(f"\n[SAVED] eval_metrics.csv — {len(df)} rows")

    with open(os.path.join(LOG_DIR, "replay_data.json"), "w") as f:
        json.dump(replay_data, f)
    print(f"[SAVED] replay_data.json — {len(replay_data)} model replays")

    if final_model is not None:
        comparison = generate_comparison_episode(final_model)
        with open(os.path.join(LOG_DIR, "comparison_data.json"), "w") as f:
            json.dump(comparison, f)
        print(f"[SAVED] comparison_data.json — same-seed trained vs random")

    print("\n=== Learning Summary ===")
    for label in ["random", "early_30k", "mid_100k", "late_200k", "final"]:
        subset = df[df["model"] == label]
        if not subset.empty:
            print(f"{label:12s} | tasks: {subset['tasks_completed'].mean():.1f} | "
                  f"reward: {subset['total_reward'].mean():.2f} | "
                  f"completion: {subset['completion_rate'].mean():.1%}")


if __name__ == "__main__":
    main()