import unittest
from unittest.mock import patch

import numpy as np

from gridworld import GridMap, GridWorld
from q_learning import (QLearningConfig, q_learning_update, train_episode,
                        train_q_learning, evaluate_q_learning)


class QLearningTests(unittest.TestCase):
    def test_update_uses_maximum_not_sampled_or_mean(self):
        q = np.zeros((1, 2, 4))
        q[0, 0, 0] = 2
        q[0, 1] = [10, 4, 0, 0]
        error = q_learning_update(q, (0, 0), 0, -1, (0, 1), False, alpha=0.5)
        self.assertEqual(error, 7)
        self.assertEqual(q[0, 0, 0], 5.5)

    def test_terminal_does_not_bootstrap(self):
        q = np.full((1, 2, 4), 10.0)
        q_learning_update(q, (0, 0), 3, -1, (0, 1), True, alpha=1.0)
        self.assertEqual(q[0, 0, 3], -1.0)

    def test_self_transition_uses_old_max(self):
        q = np.array([[[5., 2., 1., 0.]]])
        q_learning_update(q, (0, 0), 0, -1, (0, 0), False, alpha=0.5)
        self.assertEqual(q[0, 0, 0], 4.5)

    def test_truncation_keeps_max_and_no_extra_action(self):
        class TruncatedEnv:
            gamma = 1.0
            terminated = False
            def reset(self):
                return (0, 0)
            def step(self, action):
                return (0, 1), -1, False, True, {
                    "blocked": False, "elapsed_time": 1,
                    "total_reward": -1, "arrival_time": None}
        q = np.zeros((1, 2, 4))
        q[0, 1] = [1, 7, 2, 0]
        with patch("q_learning.select_action", return_value=3) as chooser:
            row = train_episode(TruncatedEnv(), q, None, QLearningConfig(alpha=1.0), 0)
        self.assertEqual(q[0, 0, 3], 6)
        self.assertEqual(chooser.call_count, 1)
        self.assertTrue(row["truncated"])
        self.assertIsNone(row["arrival_time"])

    def test_behavior_is_selected_after_update(self):
        grid = GridMap(1, 2, (0, 0), (0, 1), frozenset())
        q = np.zeros((1, 2, 4))
        calls = []
        def choose(table, state, epsilon, rng):
            calls.append(table[0, 0, 0])
            return 0 if len(calls) == 1 else 3
        with patch("q_learning.select_action", side_effect=choose):
            row = train_episode(GridWorld(grid), q, None, QLearningConfig(alpha=1.0), 0)
        self.assertEqual(calls, [0, -1])
        self.assertEqual(row["blocked_steps"], 1)
        self.assertEqual(row["steps"], 2)

    def test_config_validation_and_schedule(self):
        config = QLearningConfig()
        self.assertEqual(config.epsilon(0), 1)
        self.assertAlmostEqual(config.epsilon(3999), 0.05)
        self.assertAlmostEqual(config.epsilon(4999), 0.05)
        for kwargs in ({"episodes": 0}, {"alpha": float("nan")}, {"seed": -1},
                       {"epsilon_end": 2}, {"epsilon_decay_episodes": 1}):
            with self.assertRaises(ValueError):
                QLearningConfig(**kwargs)

    def test_training_reproducible_and_accounted(self):
        grid = GridMap.load()
        config = QLearningConfig(episodes=15)  # 测试回合数：缩短回归测试，不替代正式训练预算。
        q1, log1 = train_q_learning(grid, config)
        q2, log2 = train_q_learning(grid, config)
        np.testing.assert_array_equal(q1, q2)
        self.assertEqual(log1, log2)
        self.assertTrue(all(r["episode_return"] == -r["steps"] for r in log1))
        self.assertEqual(log1[-1]["cumulative_steps"], sum(r["steps"] for r in log1))
        for cell in [grid.goal, *grid.obstacles]:
            np.testing.assert_array_equal(q1[cell], np.zeros(4))

    def test_evaluation_frozen_and_failure_explicit(self):
        grid = GridMap(1, 2, (0, 0), (0, 1), frozenset())
        q = np.zeros((1, 2, 4))
        q[0, 0, 0] = 2
        before = q.copy()
        summary, records = evaluate_q_learning(grid, q, episodes=2, seed=1)
        np.testing.assert_array_equal(q, before)
        self.assertEqual(summary["successes"], 0)
        self.assertIsNone(summary["mean_arrival_time"])
        self.assertTrue(all(r["truncated"] and r["arrival_time"] is None for r in records))

    def test_start_at_goal_zero_steps(self):
        grid = GridMap(1, 1, (0, 0), (0, 0), frozenset())
        _, history = train_q_learning(grid, QLearningConfig(episodes=1))
        self.assertEqual(history[0]["steps"], 0)
        self.assertTrue(history[0]["terminated"])


if __name__ == "__main__":
    unittest.main()
