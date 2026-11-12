"""表格型 SARSA [R01, §6.4, Eq. 6.7]；NumPy 数组工具 [R06]。

R01: Sutton & Barto (2018), Reinforcement Learning: An Introduction.
R06: https://doi.org/10.1038/s41586-020-2649-2
环境与奖励规则见 模型假设/最终模型假设.md；本模块不使用最短路径指导训练。
"""

from dataclasses import dataclass
import math

import numpy as np

from gridworld import GridMap, GridWorld


@dataclass(frozen=True)
class SarsaConfig:
    episodes: int = 5000  # 训练回合数：控制总训练轮数。
    alpha: float = 0.1  # 学习率：控制每次Q值更新的修正幅度。
    epsilon_start: float = 1.0  # 初始探索率：训练开始时进入随机动作分支的概率。
    epsilon_end: float = 0.05  # 最低探索率：衰减结束后保留的随机探索概率。
    epsilon_decay_episodes: int = 4000  # 探索率衰减回合数：控制从初值降至下限的速度。
    seed: int = 42  # 训练随机种子：复现探索和并列动作选择的随机序列。
    eval_episodes: int = 100  # 评估回合数：冻结Q表后重复测试策略的次数。

    def __post_init__(self):
        for name in ("episodes", "epsilon_decay_episodes", "eval_episodes"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.epsilon_decay_episodes < 2:
            raise ValueError("epsilon_decay_episodes must be at least 2")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if not math.isfinite(self.alpha) or not 0 < self.alpha <= 1:
            raise ValueError("alpha must be in (0, 1]")
        if not 0 <= self.epsilon_end <= self.epsilon_start <= 1:
            raise ValueError("Require 0 <= epsilon_end <= epsilon_start <= 1")

    def epsilon(self, episode: int) -> float:
        """回合编号从0开始；第4000回合到达默认下限，此后保持。"""
        fraction = min(1.0, max(0.0, episode / (self.epsilon_decay_episodes - 1)))
        return self.epsilon_start + fraction * (self.epsilon_end - self.epsilon_start)


def select_action(q, state, epsilon, rng) -> int:
    """epsilon为探索率，控制随机动作分支概率；rng为随机数生成器。

    四个动作都参与探索；贪心分支在严格并列最大值间均匀抽样。
    """
    if rng.random() < epsilon:
        return int(rng.integers(4))
    values = q[state]
    best = np.flatnonzero(values == values.max())
    return int(rng.choice(best))


def sarsa_update(q, state, action, reward, next_state, next_action,
                 terminated, alpha, gamma=1.0) -> float:
    """SARSA [R01, 式6.7]。只有真正终止时清零后续价值。

    alpha为学习率，控制更新幅度；gamma为折扣因子，控制后续价值权重。
    本项目gamma固定为1，不折扣未来奖励。
    next_action 必须来自当前行为策略；不以 max Q 替代。
    外部截断仍需传入非空 next_action，随后采样回合才停止。
    """
    if not terminated and next_action is None:
        raise ValueError("Nonterminal transition needs a sampled next action")
    target = float(reward)
    if not terminated:
        target += gamma * float(q[next_state][next_action])
    td_error = target - float(q[state][action])
    q[state][action] += alpha * td_error
    return td_error


def train_episode(env, q, rng, config, episode):
    epsilon = config.epsilon(episode)
    state = env.reset()
    if env.terminated:
        return dict(episode=episode + 1, epsilon=epsilon, steps=0, episode_return=0,
                    terminated=True, truncated=False, arrival_time=0,
                    blocked_steps=0, mean_abs_td_error=0.0)
    action = select_action(q, state, epsilon, rng)
    blocked = 0
    td_sum = 0.0
    while True:
        next_state, reward, terminated, truncated, info = env.step(action)
        # 在更新之前选 A'；普通转移会在下一步执行同一个 A'。
        next_action = None if terminated else select_action(q, next_state, epsilon, rng)
        td_sum += abs(sarsa_update(q, state, action, reward, next_state, next_action,
                                  terminated, config.alpha, env.gamma))
        blocked += int(info["blocked"])
        if terminated or truncated:
            return dict(episode=episode + 1, epsilon=epsilon,
                        steps=info["elapsed_time"], episode_return=info["total_reward"],
                        terminated=terminated, truncated=truncated,
                        arrival_time=info["arrival_time"], blocked_steps=blocked,
                        mean_abs_td_error=td_sum / info["elapsed_time"])
        state, action = next_state, next_action


def train_sarsa(grid: GridMap, config: SarsaConfig):
    """返回Q表和每回合日志；终点及障碍的存储槽不参与更新。"""
    q = np.zeros((grid.rows, grid.cols, 4), dtype=np.float64)  # Q初值为0：不注入路径先验；float64为存储精度。
    rng = np.random.default_rng(config.seed)
    env = GridWorld(grid)
    history = []
    total_steps = 0
    for episode in range(config.episodes):
        record = train_episode(env, q, rng, config, episode)
        total_steps += record["steps"]
        record["cumulative_steps"] = total_steps
        history.append(record)
    return q, history


def evaluate_sarsa(grid: GridMap, q, episodes=100, seed=1000042):
    """episodes为评估回合数；seed为评估种子，用于复现并列动作选择。

    冻结Q、epsilon=0（关闭随机探索分支）；并列仍随机选择，使用独立随机源。
    """
    if type(episodes) is not int or episodes < 1:
        raise ValueError("episodes must be a positive integer")
    if q.shape != (grid.rows, grid.cols, 4) or not np.isfinite(q).all():
        raise ValueError("Q table shape or values invalid")
    rng = np.random.default_rng(seed)
    env = GridWorld(grid)
    records = []
    first_path = None
    for episode in range(episodes):
        state = env.reset()
        path = [state]
        while not (env.terminated or env.truncated):
            action = select_action(q, state, 0.0, rng)  # 评估探索率为0：仅按贪心策略执行。
            state, _, _, _, _ = env.step(action)
            path.append(state)
        records.append(dict(episode=episode + 1, success=env.terminated,
                            truncated=env.truncated, elapsed_time=env.elapsed,
                            arrival_time=env.elapsed if env.terminated else None,
                            episode_return=env.total_reward))
        if first_path is None:
            first_path = path  # 如首回合失败，也原样保存，不挑选最好路径。
    arrivals = [r["arrival_time"] for r in records if r["success"]]
    summary = dict(episodes=episodes, seed=seed, epsilon=0.0,
                   successes=len(arrivals), timeouts=episodes - len(arrivals),
                   success_rate=len(arrivals) / episodes,
                   mean_arrival_time=float(np.mean(arrivals)) if arrivals else None,
                   min_arrival_time=min(arrivals) if arrivals else None,
                   max_arrival_time=max(arrivals) if arrivals else None,
                   first_episode_path=first_path,
                   first_episode_success=records[0]["success"])
    return summary, records
