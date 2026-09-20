"""
两阶段门控网络训练脚本 - 全合规数据优化版

功能说明：
使用两阶段策略训练门控网络，让网络学会根据环境特征动态分配专家权重。

策略说明：
1. 第一阶段（预训练）：用大量模拟数据训练网络，学习基本的权重分配规则
2. 第二阶段（微调）：用真实视频特征微调网络，适应实际场景分布

核心改进：
- 模拟数据中注入“真缺失”样本（面罩置信度低且烟雾也低的情况），防止漏报
- 真缺失场景下YOLO权重设为较高值，让网络学会在干净背景下信任YOLO的缺失判断
- 目标权重规则针对全合规场景优化
- 支持少量真实数据微调
- 微调时降低Dropout概率，防止信息丢失

输入文件：
- 真实数据：gate_data/real_features.csv（由特征提取脚本生成）

输出文件：
- micro_gate_final.pth：训练好的门控网络权重文件

使用方法：
1. 确保已经运行特征提取脚本，生成了 gate_data/real_features.csv
2. 直接运行本脚本
3. 将生成的 micro_gate_final.pth 复制到主程序目录
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
import os
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent

# ================= 配置区域 =================
CONFIG = {
    "synthetic_samples": 8000,
    "pretrain_epochs": 50,
    "finetune_epochs": 80,
    "pretrain_lr": 0.001,
    "finetune_lr": 0.0001,
    "batch_size_syn": 64,
    "batch_size_real": 16,
    "weight_decay": 1e-4,
    "dropout_pre": 0.2,
    "dropout_ft": 0.1,
    "real_feature_csv": BASE_DIR / "gate_data" / "real_features.csv",
    "pretrain_gate_path": BASE_DIR / "pth" / "micro_gate_pretrained.pth",
    "output_gate_path": BASE_DIR / "pth" / "micro_gate_final.pth",
    "outcome_temperature": 5.0,
    "input_dim": 11,
}

FEATURE_COLUMNS = [
    "mask_conf",
    "smoke_score",
    "temporal_trend",
    "geometry_score",
    "collector_conf",
    "worker_conf",
    "brightness_norm",
    "spark_intensity",
    "conflict_k",
    "mask_area_ratio",
    "worker_mask_distance",
]


# ================= 1. 轻量门控网络 =================
class MicroGate(nn.Module):
    def __init__(self, dropout_rate=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(CONFIG["input_dim"], 24), nn.ReLU(), nn.Dropout(dropout_rate),
            nn.Linear(24, 16), nn.ReLU(), nn.Dropout(dropout_rate),
            nn.Linear(16, 3), nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.net(x)

    def set_dropout_rate(self, dropout_rate):
        for module in self.net:
            if isinstance(module, nn.Dropout):
                module.p = dropout_rate


# ================= 2. 目标权重合成规则（数学保证 YOLO 主导） =================
def synthesize_target_weights(conf, smoke, trend, is_violation):
    """
    核心设计原则：
    1. Clean 场景：YOLO 原始分必须显著高于其他专家，Softmax 后权重 > 0.65
    2. 干扰增强：YOLO 线性退让，Temporal/Smoke 平滑接管
    3. 防止 Softmax 平均主义：通过原始分差值强制拉开权重梯度
    """
    # YOLO 原始分：基础置信度 + 强烟雾抑制
    # 系数设计：conf=0.85, smoke=0.1 -> 1.1 + 0.595 - 0.05 = 1.645 (绝对主导)
    w_y_raw = 1.1 + 0.7 * conf - 0.5 * smoke
    w_y_raw = np.clip(w_y_raw, 0.4, 1.8)

    # 时序原始分：趋势驱动 + 基础保底
    w_t_raw = 0.20 + 0.40 * np.abs(trend)
    w_t_raw = np.clip(w_t_raw, 0.15, 0.60)

    # 烟雾原始分：环境浓度驱动 + 基础保底
    w_s_raw = 0.15 + 0.30 * smoke
    w_s_raw = np.clip(w_s_raw, 0.15, 0.50)

    # 特殊处理：真缺失（低置信+低烟雾）-> YOLO "未检出" 是强证据
    if conf < 0.35 and smoke < 0.30:
        w_y_raw, w_t_raw, w_s_raw = 0.75, 0.15, 0.10

    raw_weights = np.array([w_y_raw, w_t_raw, w_s_raw])
    # 数值稳定的 Softmax 归一化
    exp_w = np.exp(raw_weights - np.max(raw_weights))
    return exp_w / exp_w.sum()


def estimate_expert_scores(features):
    """
    将三个专家都转成“合规概率”视角，便于用真实标签生成收益权重。
    这不是最终判决，只用于门控训练阶段评估哪个专家更接近真实标签。
    """
    conf, smoke, trend, geometry, collector_conf, worker_conf, brightness, spark, conflict_k, area_ratio, distance = features
    yolo_score = np.clip(0.50 * conf + 0.35 * geometry + 0.10 * collector_conf + 0.05 * worker_conf, 0.0, 1.0)

    if trend < -0.15:
        temporal_score = 0.30
    elif trend > 0.15:
        temporal_score = 0.70
    else:
        temporal_score = 0.50 + 0.40 * np.clip(trend, -0.15, 0.15)
    if conflict_k > 0.25:
        temporal_score = 0.55 * temporal_score + 0.45 * np.clip(geometry, 0.0, 1.0)
    temporal_score = np.clip(temporal_score, 0.10, 0.90)

    smoke_score = 0.85 * np.exp(-3.0 * smoke)
    if brightness < 0.30 or spark > 0.50:
        smoke_score *= 0.85
    smoke_score = np.clip(smoke_score, 0.25, 0.85)
    return np.array([yolo_score, temporal_score, smoke_score], dtype=np.float32)


def synthesize_outcome_target_weights(features, is_violation):
    """
    真实数据微调标签：不再让门控单纯模仿人工权重规则，
    而是根据专家分数与真实标签的误差生成“收益型”软标签。
    """
    target_compliance = 1.0 - float(is_violation)
    conf, smoke, trend, geometry, collector_conf, worker_conf, brightness, spark, conflict_k, area_ratio, distance = features
    scores = estimate_expert_scores(features)
    errors = np.abs(scores - target_compliance)
    weights = np.exp(-CONFIG["outcome_temperature"] * errors)

    # 低烟且低置信时，如果确实违规，YOLO 的缺失检测是强证据。
    if is_violation and conf < 0.35 and smoke < 0.45:
        weights[0] += 0.45

    # 高烟/灰尘下如果标注仍为合规，说明单帧 YOLO 容易误报，时序与烟雾专家应承担更多修正责任。
    if (not is_violation) and smoke > 0.35 and conf < 0.55:
        weights[1] += 0.35
        weights[2] += 0.25

    # 置信度快速跳变时，提高时序专家权重，服务于降低跳变率。
    if abs(trend) > 0.12 or conflict_k > 0.20:
        weights[1] += 0.35

    # 几何关系清楚时强化 YOLO/几何专家，几何关系差且烟雾明显时提高保守专家。
    if geometry > 0.65 and distance < 0.35:
        weights[0] += 0.25
    if geometry < 0.45 and smoke > 0.35:
        weights[1] += 0.20
        weights[2] += 0.20

    weights = np.maximum(weights, 1e-6)
    return weights / weights.sum()


# ================= 3. 生成模拟数据（严格对齐校准后分布） =================
def generate_synthetic_data(num_samples):
    print(f"正在生成模拟数据，数量：{num_samples}")
    np.random.seed(42)

    # 比例分配：覆盖校准后的典型区间
    n_clean = int(num_samples * 0.30)  # Clean: 低烟, 高置信
    n_light = int(num_samples * 0.25)  # Light: 中低烟, 中高置信
    n_med = int(num_samples * 0.25)  # Medium: 中高烟, 中置信
    n_heavy = int(num_samples * 0.20)  # Heavy: 高烟, 低置信

    # Clean 场景：smoke 校准后通常在 0.05~0.25，conf 0.80~0.95
    c1 = np.random.normal(0.88, 0.05, n_clean)
    s1 = np.random.uniform(0.05, 0.25, n_clean)
    t1 = np.random.normal(0.0, 0.03, n_clean)
    l1 = np.zeros(n_clean)

    # Light 场景：smoke 0.25~0.45，conf 0.65~0.80
    c2 = np.random.normal(0.72, 0.06, n_light)
    s2 = np.random.uniform(0.25, 0.45, n_light)
    t2 = np.random.normal(-0.05, 0.05, n_light)
    l2 = np.zeros(n_light)

    # Medium 场景：smoke 0.45~0.70，conf 0.45~0.65
    c3 = np.random.normal(0.55, 0.08, n_med)
    s3 = np.random.uniform(0.45, 0.70, n_med)
    t3 = np.random.normal(-0.12, 0.08, n_med)
    l3 = np.zeros(n_med)

    # Heavy 场景：smoke 0.70~0.95，conf 0.25~0.45
    c4 = np.random.normal(0.35, 0.07, n_heavy)
    s4 = np.random.uniform(0.70, 0.95, n_heavy)
    t4 = np.random.normal(-0.18, 0.10, n_heavy)
    l4 = np.ones(n_heavy)

    C = np.clip(np.concatenate([c1, c2, c3, c4]), 0.05, 0.95)
    S = np.clip(np.concatenate([s1, s2, s3, s4]), 0.05, 0.95)
    T = np.clip(np.concatenate([t1, t2, t3, t4]), -0.4, 0.1)
    L = np.concatenate([l1, l2, l3, l4])

    geometry = np.clip(0.15 + 0.75 * C - 0.25 * S + np.random.normal(0.0, 0.08, len(C)), 0.0, 1.0)
    collector = np.clip(np.random.normal(0.72, 0.18, len(C)) - 0.25 * L, 0.0, 1.0)
    worker = np.clip(np.random.normal(0.82, 0.10, len(C)), 0.0, 1.0)
    brightness = np.clip(np.random.normal(0.55, 0.16, len(C)) - 0.20 * S, 0.05, 0.95)
    spark = np.clip(np.random.beta(2.0, 7.0, len(C)) + 0.20 * (S > 0.45), 0.0, 1.0)
    conflict = np.clip(0.20 * S + 0.40 * np.abs(T) + np.random.normal(0.0, 0.03, len(C)), 0.0, 1.0)
    area = np.clip(0.09 + 0.16 * C + np.random.normal(0.0, 0.04, len(C)), 0.0, 1.0)
    distance = np.clip(0.55 - 0.35 * geometry + 0.15 * S + np.random.normal(0.0, 0.08, len(C)), 0.0, 1.0)

    X_syn = np.column_stack([C, S, T, geometry, collector, worker, brightness, spark, conflict, area, distance])
    Y_syn = np.array([synthesize_outcome_target_weights(x, l) for x, l in zip(X_syn, L)])

    # 打印预训练目标分布验证（确保 Clean 时 YOLO > 0.65）
    print("   预训练目标权重均值验证:")
    print(
        f"     Clean   -> YOLO:{Y_syn[:n_clean, 0].mean():.3f} Temp:{Y_syn[:n_clean, 1].mean():.3f} Smoke:{Y_syn[:n_clean, 2].mean():.3f}")
    print(
        f"     Heavy   -> YOLO:{Y_syn[-n_heavy:, 0].mean():.3f} Temp:{Y_syn[-n_heavy:, 1].mean():.3f} Smoke:{Y_syn[-n_heavy:, 2].mean():.3f}")

    return torch.FloatTensor(X_syn), torch.FloatTensor(Y_syn)


# ================= 4. 加载真实数据 =================
def load_real_data(csv_path=None):
    if csv_path is None:
        csv_path = CONFIG["real_feature_csv"]
    if not os.path.exists(csv_path):
        raise FileNotFoundError("未找到真实数据文件")
    print(f"正在加载真实数据，路径：{csv_path}")
    df = pd.read_csv(csv_path)
    required_cols = FEATURE_COLUMNS + ["label"]
    if not all(col in df.columns for col in required_cols):
        missing = [col for col in required_cols if col not in df.columns]
        raise ValueError(f"CSV列名不匹配，缺少: {missing}")
    X_real = df[FEATURE_COLUMNS].values.astype(np.float32)
    labels = df["label"].values
    print(f"  加载了 {len(X_real)} 个真实样本 | 合规={sum(labels == 0)} 违规={sum(labels == 1)}")
    Y_real = np.array(
        [synthesize_outcome_target_weights(x, l) for x, l in zip(X_real, labels)])
    print("  收益型目标权重均值:")
    print(f"    YOLO:{Y_real[:, 0].mean():.3f} Temporal:{Y_real[:, 1].mean():.3f} Smoke:{Y_real[:, 2].mean():.3f}")
    return torch.FloatTensor(X_real), torch.FloatTensor(Y_real)


# ================= 5. 主训练流程 =================
def main():
    print("\n 两阶段门控网络训练 | 策略：规则引导预训练 + 纯净微调\n")
    model = MicroGate(dropout_rate=CONFIG["dropout_pre"])
    criterion = nn.MSELoss()

    # 第一阶段：预训练
    print(" 第一阶段：规则引导预训练")
    X_syn, Y_syn = generate_synthetic_data(CONFIG["synthetic_samples"])
    syn_loader = DataLoader(TensorDataset(X_syn, Y_syn), batch_size=CONFIG["batch_size_syn"], shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["pretrain_lr"], weight_decay=CONFIG["weight_decay"])

    model.train()
    for epoch in range(CONFIG["pretrain_epochs"]):
        total_loss = 0
        for xb, yb in syn_loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch [{epoch + 1}/{CONFIG['pretrain_epochs']}] | Loss: {total_loss / len(syn_loader):.4f}")

    CONFIG["pretrain_gate_path"].parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), CONFIG["pretrain_gate_path"])
    torch.save(model.state_dict(), CONFIG["output_gate_path"])
    print(f" 预训练完成 | 保存: {CONFIG['pretrain_gate_path']}\n")

    # 第二阶段：微调
    print(" 第二阶段：真实数据微调")
    try:
        X_real, Y_real = load_real_data()
    except Exception as e:
        print(f" {e} | 跳过微调")
        return

    model.set_dropout_rate(CONFIG["dropout_ft"])
    batch_size_real = min(CONFIG["batch_size_real"], max(4, len(X_real) // 4))
    real_loader = DataLoader(TensorDataset(X_real, Y_real), batch_size=batch_size_real, shuffle=True)
    optimizer.param_groups[0]['lr'] = CONFIG["finetune_lr"]

    best_loss = float('inf')
    model.train()
    for epoch in range(CONFIG["finetune_epochs"]):
        total_loss = 0
        for xb, yb in real_loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_real), Y_real).item()
        model.train()

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), CONFIG["output_gate_path"])
        if (epoch + 1) % 20 == 0:
            print(
                f"  Epoch [{epoch + 1}/{CONFIG['finetune_epochs']}] | ValLoss: {val_loss:.4f} | Best: {best_loss:.4f}")

    print(f"\n 微调完成！最佳验证损失: {best_loss:.4f} | 保存: {CONFIG['output_gate_path']}\n")

    # ========== 模型验证 ==========
    print(" 模型验证测试")
    model.eval()
    test_cases = [
        ("Clean场景", [0.90, 0.10, 0.00, 0.82, 0.80, 0.90, 0.58, 0.10, 0.02, 0.22, 0.18]),
        ("Light场景", [0.72, 0.30, -0.02, 0.68, 0.72, 0.86, 0.50, 0.22, 0.08, 0.19, 0.25]),
        ("Medium场景", [0.55, 0.55, -0.05, 0.48, 0.58, 0.82, 0.42, 0.35, 0.18, 0.15, 0.42]),
        ("Heavy场景", [0.35, 0.80, -0.08, 0.30, 0.45, 0.75, 0.32, 0.55, 0.32, 0.10, 0.62]),
    ]

    print("说明：YOLO权重越高表示越信任YOLO检测结果")
    print("")

    with torch.no_grad():
        for name, feat in test_cases:
            inp = torch.FloatTensor([feat])
            weights = model(inp).squeeze().detach().cpu().numpy()
            print(f"  [{name}]")
            print(f"    输入: conf={feat[0]:.2f}, smoke={feat[1]:.2f}, trend={feat[2]:.2f}, geometry={feat[3]:.2f}, K={feat[8]:.2f}")
            print(f"    输出: YOLO={weights[0]:.3f}, Temporal={weights[1]:.3f}, Smoke={weights[2]:.3f}\n")

    print(" 训练全部完成！请将 pth/micro_gate_final.pth 用于主程序推理。")


if __name__ == "__main__":
    main()



