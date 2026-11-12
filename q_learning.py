"""表格型Q-learning：最大下一状态动作价值更新[R02, §2, pp.280–281]。

R02: Watkins & Dayan (1992), https://doi.org/10.1007/BF00992698
NumPy工具[R06]。共享SARSA的参数校验、探索调度、动作选择与冻结评估，
但训练循环和更新函数独立，不调用SARSA更新。环境规则不变。
"""

from dataclasses import dataclass

import numpy as np

from gridworld import GridMap, GridWorld
from sarsa import SarsaConfig, select_action, evaluate_sarsa


@dataclass(frozen=True)
class QLearningConfig(SarsaConfig):
    """继承相同的参数校验与探索率调度，便于同配置实验。"""
    episodes: int = 5000  # 训练回合数：控制总训练轮数。
    alpha: float = 0.1  # 学习率：控制每次Q值更新幅度。
    epsilon_start: float = 1.0  # 初始探索率：训练初期随机动作分支概率。
    epsilon_end: float = 0.05  # 最低探索率：后期保留的探索概率。
    epsilon_decay_episodes: int = 4000  # 衰减回合数：控制探索率下降速度。
    seed: int = 42  # 训练种子：复现探索与并列动作选择。
    eval_episodes: int = 100  # 评估回合数：冻结Q表后重复测试次数。


def q_learning_update(q, state, action, reward, next_state, terminated,
                      alpha, gamma=1.0):
    """Q-learning更新[R02]；alpha为学习率，gamma为后续奖励折扣因子。

    gamma默认1是已确认环境设定；只有真正终止才取消自举，截断仍保留。
    先完整计算目标再修改Q，保证碰撞原地转移时使用更新前的值。
    """
    target = float(reward)
    if not terminated:
        target += gamma * float(np.max(q[next_state]))
    error = target - float(q[state][action])
    q[state][action] += alpha * error
    return error


def train_episode(env, q, rng, config, episode):
    state = env.reset()
    epsilon = config.epsilon(episode)
    if env.terminated:
        return dict(episode=episode + 1, epsilon=epsilon, steps=0, episode_return=0,
                    terminated=True, truncated=False, arrival_time=0, blocked_steps=0)
    blocked = 0
    while True:
        action = select_action(q, state, epsilon, rng)
        nxt, reward, terminated, truncated, info = env.step(action)
        q_learning_update(q, state, action, reward, nxt, terminated, config.alpha, env.gamma)
        blocked += int(info["blocked"])
        if terminated or truncated:
            return dict(episode=episode + 1, epsilon=epsilon, steps=info["elapsed_time"],
                        episode_return=info["total_reward"], terminated=terminated,
                        truncated=truncated, arrival_time=info["arrival_time"],
                        blocked_steps=blocked)
        state = nxt  # 下一轮在更新后的Q上选择行为动作，不用其取代本次max目标。


def train_q_learning(grid: GridMap, config: QLearningConfig):
    q = np.zeros((grid.rows, grid.cols, 4), dtype=np.float64)  # Q初值0：不注入最短路径先验。
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


def evaluate_q_learning(grid, q, episodes=100, seed=1000042):
    """episodes为评估回合数，seed为独立评估种子；ε=0，冻结Q。

    两算法的贪心评估完全相同，复用现有纯评估函数，不执行SARSA学习。
    """
    return evaluate_sarsa(grid, q, episodes=episodes, seed=seed)
