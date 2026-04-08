import os
import torch
import numpy as np
import pandas as pd
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import BaseCallback
from environment import TaskAllocationEnv

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TOTAL_TIMESTEPS = 300_000
SAVE_STEPS = [30_000, 100_000, 200_000, 300_000]
MODEL_DIR = "models"
LOG_DIR = "logs"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)


def mask_fn(env):
    return env.action_masks()


class MetricsCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_tasks = []
        self.episode_efficiency = []
        self.episode_energy = []
        self.episode_completion_rate = []
        self.episode_count = 0
        self._current_reward = 0.0
        self._save_at = set(SAVE_STEPS)

    def _on_step(self):
        self._current_reward += self.locals["rewards"][0]

        if self.locals["dones"][0]:
            wrapped = self.training_env.envs[0]
            env = wrapped.env if hasattr(wrapped, 'env') else wrapped
            summary = env.get_episode_summary()

            self.episode_rewards.append(self._current_reward)
            self.episode_tasks.append(summary.get("tasks_completed", 0))
            self.episode_efficiency.append(summary.get("efficiency", 0))
            self.episode_energy.append(summary.get("total_energy_used", 0))
            self.episode_completion_rate.append(summary.get("completion_rate", 0))
            self.episode_count += 1
            self._current_reward = 0.0

        if self.num_timesteps in self._save_at:
            checkpoint_path = os.path.join(MODEL_DIR, f"ppo_checkpoint_{self.num_timesteps}")
            self.model.save(checkpoint_path)
            avg_tasks = np.mean(self.episode_tasks[-100:]) if self.episode_tasks else 0
            avg_reward = np.mean(self.episode_rewards[-100:]) if self.episode_rewards else 0
            print(f"[CHECKPOINT] Saved at {self.num_timesteps} | avg_tasks(100ep): {avg_tasks:.2f} | avg_reward: {avg_reward:.1f}")
            self._save_logs()

        return True

    def _save_logs(self):
        df = pd.DataFrame({
            "episode": list(range(1, len(self.episode_rewards) + 1)),
            "reward": self.episode_rewards,
            "tasks_completed": self.episode_tasks,
            "efficiency": self.episode_efficiency,
            "energy_used": self.episode_energy,
            "completion_rate": self.episode_completion_rate,
        })
        df.to_csv(os.path.join(LOG_DIR, "training_metrics.csv"), index=False)

    def on_training_end(self):
        self._save_logs()
        avg_tasks = np.mean(self.episode_tasks[-200:]) if self.episode_tasks else 0
        print(f"[DONE] {self.episode_count} episodes | Final avg tasks (last 200ep): {avg_tasks:.2f}/10")


def make_env():
    def _init():
        env = TaskAllocationEnv()
        env = ActionMasker(env, mask_fn)
        return env
    return _init


def train():
    print(f"[INFO] Device: {DEVICE}")
    print(f"[INFO] Training MaskablePPO for {TOTAL_TIMESTEPS} timesteps")

    env = DummyVecEnv([make_env()])
    env = VecMonitor(env)

    model = MaskablePPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.005,
        verbose=1,
        device=DEVICE,
        policy_kwargs=dict(net_arch=[256, 256]),
    )

    callback = MetricsCallback()
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback)

    final_path = os.path.join(MODEL_DIR, "ppo_final")
    model.save(final_path)
    print(f"[INFO] Final model saved to {final_path}")
    callback._save_logs()


if __name__ == "__main__":
    train()