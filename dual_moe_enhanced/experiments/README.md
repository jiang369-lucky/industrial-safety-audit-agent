# 实验目录说明

本目录按照论文实验章节整理。当前策略是：核心消融保留为实验表，小 MoE 保留权重图，其余不适合放论文的复杂图改为表格型实验。

## 目录结构

- `data_preparation/`：低清、粉尘、烟雾和干扰视频生成脚本。
- `data_preparation/resources/`：逐帧标注、视频列表和场景标注资源。
- `ablation/`：Clean 条件下的 A/B/C/D 核心消融实验。
- `moe_analysis/`：小 MoE 权重日志与 Fig.3 可视化。
- `dataset_statistics/`：数据集规模和逐帧标注统计表。
- `main_comparison/`：主对比实验表。
- `qwen_efficiency/`：Qwen 调用率、FPS 与精度权衡表。
- `robustness/`：干扰鲁棒性补充实验表。
- `scene_router/`：宏观场景路由结果表。
- `error_analysis/`：视频级误差与局限性分析表。
- `paper_figures/`：论文图片输出目录，目前建议只使用 `moe/fig3_moe_weights_cn.png`。
- `paper_tables/`：论文表格 Markdown/CSV 输出目录。
- `common/`：公共路径和表格工具。

## 推荐运行顺序

1. 运行核心消融和小 MoE 权重日志：

```cmd
python -B dual_moe_enhanced\experiments\ablation\run_full_ablation.py --conditions Clean Light Medium Heavy --modes A B C D
```

2. 打印 Clean 消融结果：

```cmd
python -B dual_moe_enhanced\experiments\ablation\print_ablation_summary.py --all
```

3. 生成小 MoE 权重图：

```cmd
python -B dual_moe_enhanced\experiments\moe_analysis\plot_fig3_moe_weights.py
```

4. 生成数据集统计表：

```cmd
python -B dual_moe_enhanced\experiments\dataset_statistics\build_dataset_statistics.py
```

5. 生成主对比表：

```cmd
python -B dual_moe_enhanced\experiments\main_comparison\build_main_comparison_table.py
```

6. 生成 Qwen 效率表：

```cmd
python -B dual_moe_enhanced\experiments\qwen_efficiency\build_qwen_efficiency_table.py
```

7. 生成鲁棒性补充表。如果已经有 `robustness_results.csv`，只需要运行第二条：

```cmd
python -B dual_moe_enhanced\experiments\robustness\run_robustness.py
python -B dual_moe_enhanced\experiments\robustness\build_robustness_table.py
```

8. 生成场景路由表。如果已经有 `scene_router_results.csv`，只需要运行第二条：

```cmd
python -B dual_moe_enhanced\experiments\scene_router\eval_scene_router.py
python -B dual_moe_enhanced\experiments\scene_router\build_scene_router_table.py
```

9. 生成视频级误差分析表：

```cmd
python -B dual_moe_enhanced\experiments\error_analysis\build_error_analysis_table.py
```

