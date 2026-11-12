# 网格环境中小车的最短时间路径规划

用 SARSA 和 Q-learning 求解 10×10 网格中的最短路径，并比较两种算法在原地图、新地图和直接迁移时的表现。两张地图的最短路径分别为 16 步和 18 步。

## 运行

在仓库根目录使用 Python 3.13 运行以下命令。

```powershell
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python prepare_paper_figures.py --data outputs/paper_experiments_20261104 --output generated/figures
```

如需复现实验，使用已保存的原地图基线，重新运行新地图训练和迁移评估。

```powershell
python run_paper_experiments.py --baseline outputs/paper_experiments_20261027 --output generated/reproduction
python prepare_paper_figures.py --data generated/reproduction --output generated/reproduction/figures
```

省略 `--baseline` 可从头训练两张地图。每次训练请使用尚不存在的结果目录。

## 文件

| 目录或入口 | 内容 |
|---|---|
| `gridworld.py`、`sarsa.py`、`q_learning.py` | 环境和算法实现 |
| `train_*.py`、`plot_*.py` | 单次训练、回报曲线和策略路径图 |
| `maps/`、`模型假设/` | 两张地图及建模规则 |
| `outputs/` | 实验配置、训练和评估日志、Q 表、图表和汇总 |
| `报告/`、`参考文献/` | 模型、算法设计、实验分析和引用 |

