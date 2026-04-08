import gymnasium as gym
from gymnasium import spaces
import numpy as np

GRID_SIZE = 10
NUM_TASKS = 10
NUM_AGENTS = 5

AGENT_CONFIGS = [
    {"type": "A", "capacity": 3, "speed": 1, "energy": 100},
    {"type": "A", "capacity": 3, "speed": 1, "energy": 100},
    {"type": "B", "capacity": 1, "speed": 2, "energy": 60},
    {"type": "B", "capacity": 1, "speed": 2, "energy": 60},
    {"type": "B", "capacity": 1, "speed": 2, "energy": 60},
]


class TaskAllocationEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, num_tasks=NUM_TASKS, render_mode=None):
        super().__init__()
        self.num_tasks = num_tasks
        self.num_agents = NUM_AGENTS
        self.grid_size = GRID_SIZE
        self.render_mode = render_mode
        self.agent_configs = AGENT_CONFIGS

        obs_size = self.num_agents * 3 + self.num_tasks * 3 + self.num_tasks
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_size,), dtype=np.float32
        )
        self.action_space = spaces.MultiDiscrete(
            [self.num_tasks + 1] * self.num_agents
        )

        self.max_steps = 80
        self._step = 0
        self.history = []
        self._prev_min_dist = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        rng = self.np_random

        self.agent_positions = rng.integers(0, self.grid_size, size=(self.num_agents, 2)).astype(np.float32)
        self.agent_energy = np.array([cfg["energy"] for cfg in self.agent_configs], dtype=np.float32)
        self.agent_speed = np.array([cfg["speed"] for cfg in self.agent_configs], dtype=np.float32)
        self.agent_capacity = np.array([cfg["capacity"] for cfg in self.agent_configs], dtype=np.float32)

        self.task_positions = rng.integers(0, self.grid_size, size=(self.num_tasks, 2)).astype(np.float32)
        self.task_difficulty = rng.integers(1, 4, size=(self.num_tasks,)).astype(np.float32)
        self.task_done = np.zeros(self.num_tasks, dtype=np.float32)

        self._step = 0
        self.history = []
        self._prev_min_dist = self._compute_min_distances()

        return self._get_obs(), {}

    def _compute_min_distances(self):
        incomplete = np.where(self.task_done == 0)[0]
        if len(incomplete) == 0:
            return np.zeros(self.num_agents)
        dists = np.zeros(self.num_agents)
        for i in range(self.num_agents):
            d = np.linalg.norm(self.task_positions[incomplete] - self.agent_positions[i], axis=1)
            dists[i] = d.min()
        return dists

    def _get_obs(self):
        agent_part = np.zeros(self.num_agents * 3, dtype=np.float32)
        for i in range(self.num_agents):
            agent_part[i * 3] = self.agent_positions[i, 0] / self.grid_size
            agent_part[i * 3 + 1] = self.agent_positions[i, 1] / self.grid_size
            agent_part[i * 3 + 2] = self.agent_energy[i] / 100.0

        task_part = np.zeros(self.num_tasks * 3, dtype=np.float32)
        for j in range(self.num_tasks):
            task_part[j * 3] = self.task_positions[j, 0] / self.grid_size
            task_part[j * 3 + 1] = self.task_positions[j, 1] / self.grid_size
            task_part[j * 3 + 2] = self.task_difficulty[j] / 3.0

        done_part = self.task_done.copy()
        return np.concatenate([agent_part, task_part, done_part]).astype(np.float32)

    def action_masks(self):
        masks = np.ones((self.num_agents, self.num_tasks + 1), dtype=bool)
        for i in range(self.num_agents):
            if self.agent_energy[i] <= 0:
                masks[i, :self.num_tasks] = False
            else:
                for t in range(self.num_tasks):
                    if self.task_done[t]:
                        masks[i, t] = False
        return masks.flatten()

    def step(self, actions):
        reward = 0.0
        tasks_completed_this_step = 0

        cur_min_dist = self._compute_min_distances()
        for i in range(self.num_agents):
            if self.agent_energy[i] > 0:
                improvement = self._prev_min_dist[i] - cur_min_dist[i]
                reward += improvement * 0.3

        for i, action in enumerate(actions):
            if self.agent_energy[i] <= 0:
                continue
            if action == self.num_tasks:
                continue
            if self.task_done[action]:
                reward -= 0.2
                continue

            dist = np.linalg.norm(self.agent_positions[i] - self.task_positions[action])
            move_cost = dist / self.agent_speed[i]

            if self.agent_energy[i] < move_cost:
                reward -= 0.1
                continue

            self.agent_positions[i] = self.task_positions[action].copy()
            self.agent_energy[i] -= move_cost

            task_cost = self.task_difficulty[action] / self.agent_capacity[i]
            if self.agent_energy[i] >= task_cost:
                self.agent_energy[i] -= task_cost
                self.task_done[action] = 1.0
                tasks_completed_this_step += 1
                energy_bonus = (self.agent_energy[i] / 100.0) * 0.5
                reward += 1.0 + energy_bonus
            else:
                reward -= 0.1

        reward -= 0.02

        tasks_done = int(self.task_done.sum())
        self._step += 1
        terminated = tasks_done == self.num_tasks
        truncated = self._step >= self.max_steps

        if terminated:
            steps_remaining = self.max_steps - self._step
            reward += 5.0 + steps_remaining * 0.1

        self._prev_min_dist = self._compute_min_distances()

        self.history.append({
            "step": self._step,
            "tasks_completed": tasks_done,
            "reward": reward,
            "agent_energy": self.agent_energy.copy(),
            "agent_positions": self.agent_positions.copy(),
            "task_done": self.task_done.copy(),
        })

        info = {
            "tasks_completed": tasks_done,
            "tasks_this_step": tasks_completed_this_step,
            "efficiency": tasks_done / max(self._step, 1),
            "total_energy_remaining": float(self.agent_energy.sum()),
            "action_masks": self.action_masks(),
        }

        return self._get_obs(), reward, terminated, truncated, info

    def get_episode_summary(self):
        if not self.history:
            return {}
        final = self.history[-1]
        total_energy_used = sum(
            self.agent_configs[i]["energy"] - final["agent_energy"][i]
            for i in range(self.num_agents)
        )
        return {
            "tasks_completed": int(final["tasks_completed"]),
            "total_steps": self._step,
            "total_energy_used": float(total_energy_used),
            "efficiency": final["tasks_completed"] / max(self._step, 1),
            "completion_rate": final["tasks_completed"] / self.num_tasks,
        }

    def render(self):
        pass