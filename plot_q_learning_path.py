"""Q-learning最终贪心策略路径入口；共享冻结评估与绘图[R04, R06, R07]。

--seed默认沿用历史评估种子：复现并列动作选择；--dpi默认180：PNG清晰度。
参数定义及注释在plot_sarsa_path.py中，--run-dir可选择其他训练结果。
"""
from gridworld import ROOT
from plot_sarsa_path import main

if __name__ == "__main__":
    main(default_run=ROOT / "outputs/q_learning/design_validation_seed42")
