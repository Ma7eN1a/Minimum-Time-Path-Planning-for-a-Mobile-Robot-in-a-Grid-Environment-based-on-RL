# SARSA最终策略路径

## 生成口径

读取`outputs/sarsa/design_validation_seed42/q_table.npy`，使用同目录`config.json`中的训练地图快照。冻结Q表，沿用已有评估代码，以探索率ε=0执行一个回合；最大Q值并列时均匀随机选择，评估种子1000042（沿用历史评估设置）。这些评估参数是本项目设定。

图中展示首个评估回合，不从多条轨迹中挑选最好路径。每个动作都通过原环境执行，碰撞和超时规则不变。没有重新训练，也没有修改原Q表、训练日志或历史评估文件。此图是从S出发的一次轨迹，不是全状态策略箭头图。

NumPy用于加载和校验Q表[R06]，Matplotlib用于绘图[R07]；轨迹来自本项目实际执行。BFS仅在轨迹生成后用于独立最短长度核验[R04]，不参与策略动作选择或路线绘制。

## 实际结果

- 起点S=(8,1)，终点G=(0,9)，坐标按从0开始的(行,列)，左上角为原点。
- 成功到达，未超时；16次动作，耗时16个模型时间单位，累积奖励−16，碰撞0次。
- 每一步均已核验为当前Q表允许的贪心动作转移，Q表在评估前后完全一致。
- 重新生成的路径与历史`evaluation.json`中的首回合路径完全一致。
- BFS基准为16步，本次轨迹达到最短长度；不据此声称最短路径唯一，也不推广到其他地图或训练种子。

完整轨迹：

```text
S(8,1) → (8,2) → (8,3) → (7,3) → (7,4) → (6,4)
→ (6,5) → (6,6) → (5,6) → (4,6) → (3,6) → (2,6)
→ (2,7) → (1,7) → (1,8) → (1,9) → G(0,9)
```

图中箭头表示行进方向，数字1–15为中间状态的到达步序；S对应第0步，G对应第16步。已实际打开PNG检查地图、箭头、编号与图例。

## 运行与输出

在PyCharm中使用项目`.venv`解释器运行`plot_sarsa_path.py`，即可保存并显示图片。不弹窗方式：

```powershell
.venv\Scripts\python.exe plot_sarsa_path.py --no-show
```

输出位于`outputs/sarsa/design_validation_seed42/figures/`：

- `final_policy_path.png`：路径图。
- `final_policy_path.svg`：矢量路径图。
- `final_policy_path.json`：完整坐标序列、评估参数、结果、基准比较及Q表SHA-256摘要。

可以用`--run-dir`指定其他同格式训练结果，用`--seed`指定并列动作选择的随机种子；重复绘制会更新同名绘图产物，不修改源训练文件。

## 实际引用

- [R04] Sedgewick R, Wayne K. BreadthFirstPaths: Algorithms, 4th Edition companion documentation. https://algs4.cs.princeton.edu/code/javadoc/edu/princeton/cs/algs4/BreadthFirstPaths.html
- [R06] Harris C R, et al. Array programming with NumPy. Nature, 2020, 585: 357–362. DOI: 10.1038/s41586-020-2649-2.
- [R07] Hunter J D. Matplotlib: A 2D Graphics Environment. Computing in Science & Engineering, 2007, 9(3): 90–95. DOI: 10.1109/MCSE.2007.55.
