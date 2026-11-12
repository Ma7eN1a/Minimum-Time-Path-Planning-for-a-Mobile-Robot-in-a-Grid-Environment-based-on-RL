"""Q-learning奖励训练曲线入口；共享绘图实现[R06, R07]，不绘制损失图。

--window默认100：尾随回合平滑窗口；--dpi默认180：PNG清晰度。
参数定义及注释在plot_sarsa_training.py中，--run-dir可选择其他训练结果。
"""
from gridworld import ROOT
from plot_sarsa_training import main

if __name__ == "__main__":
    main(default_run=ROOT / "outputs/q_learning/design_validation_seed42")
