"""
==============================================================
[完整可运行版 | 方案B] 敲击(4通道) + GPR 多模态融合缺陷检测系统
--------------------------------------------------------------
融合方案（方案B：GPR沿655方向提特征，最终输出145×64）：

缩减特征提取的数量

Knock:
    Knock 特征(29×64×N_knock) → TransformerAE → prob_knock(29×64×N_sel)
    模态内 Bayes(prob_knock) → post_knock(29×64)
    post_knock 上采样 → post_knock_up(145×64)

GPR:
    GPR 原始(655×145×64) → 沿655提特征 → gpr_feat_stack(145×64×N_gpr)
    gpr_feat_stack → TransformerAE → prob_gpr(145×64×N_gpr)
    模态内 Bayes(prob_gpr) → post_gpr(145×64)

跨模态：
    stack([post_knock_up, post_gpr]) → Bayes → post_final(145×64)

==============================================================
"""

# ===========================================================
# 0. Imports（清理重复/无用）
# ===========================================================
import os
import re
import time
import inspect
import warnings
from contextlib import contextmanager
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import pywt
import scipy.io as sio
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from tqdm import tqdm
from scipy.stats import kurtosis, skew, entropy
from scipy.ndimage import zoom
from scipy.ndimage import label, binary_dilation
from scipy.signal import hilbert, find_peaks
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    accuracy_score, confusion_matrix)



def set_global_seed(seed=10, deterministic=True):
    import os
    import random
    import numpy as np
    import torch

    # -------- Python --------
    random.seed(seed)

    # -------- NumPy --------
    np.random.seed(seed)

    # -------- PyTorch --------
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # -------- cuDNN --------
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True

    # -------- Hash seed (important for dict/set order) --------
    os.environ["PYTHONHASHSEED"] = str(seed)

    print(f"[Seed] Global seed fixed to {seed}, deterministic={deterministic}")

warnings.filterwarnings("ignore")
def tsne_fit_transform_safe(X, seed=43, perplexity=30, init="pca", iters=2000):
    """
    - 自动修正 perplexity（保证 < n_samples）
    - 自动兼容 sklearn: n_iter / max_iter
    - 自动处理 NaN/Inf
    返回: (emb, used_perplexity)
    """
    X = np.asarray(X, dtype=np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    n = int(X.shape[0])
    if n < 3:
        return None, None

    # perplexity 必须 < n_samples
    # 一个稳妥策略：perp <= (n-1)//3，同时不小于 2
    perp = int(min(perplexity, max(2, (n - 1) // 3), n - 1))

    kwargs = dict(
        n_components=2,
        perplexity=perp,
        init=init,
        learning_rate="auto",
        random_state=int(seed),
        metric="euclidean",
        early_exaggeration=12.0,
    )

    # sklearn 版本兼容：max_iter vs n_iter
    sig = inspect.signature(TSNE.__init__)
    if "max_iter" in sig.parameters:
        kwargs["max_iter"] = int(iters)
    else:
        kwargs["n_iter"] = int(iters)

    tsne = TSNE(**kwargs)
    emb = tsne.fit_transform(X)
    return emb, perp
plt.switch_backend("agg")


# ===========================================================
# 1) 用户参数入口（优先只改这里）
# ===========================================================
PROJECT_CFG = {
    "data_name": "sdcq_2",
    "gt_path": r"C:\GUO\jinqiang\Groundtruth_4.xlsx",
    "base_seed": 43,
}

PATH_CFG = {
    "knock_root": r"C:\GUO\jinqiang\Autoencoder",
    "gpr_root": r"C:\GUO\jinqiang\Ultrasonic_GRP",
    "export_root": r"C:\GUO\jinqiang\Paper2\New_plot\result",
}

HEALTHY_REGION_CFG = {
    "mode": "FIRST_COLS",    # "FIRST_COLS" or "COL_LIST"
    "first_cols": 4,
    "col_list": [17],        # 0-based 列索引，仅 mode="COL_LIST" 时生效
    "save_masks": True,
}

TAE_EXPORT_CFG = {
    "export_latent": True,
    "export_topk": 60,
    "knock_fig_features": [],
    "gpr_fig_features": [],
    "tsne_max_samples": 4000,
    "tsne_random_state": 43,
    "tsne_perplexity": 30,
}

RUN_SWITCH = {
    "step1_txt2npy": True,
    "step2_feature": True,
    "step3_matrix": True,
    "step4_stack": True,
    "step0B_gpr_extract_features": True,
    "step5A_AE_knock": True,
    "step5B_AE_gpr": True,
    "step6_bayes_fusion": True,
}

EXPERIMENT_CFG = {
    "use_channels": [1, 2, 3],
    "default_exp_tag": "col_list",
}

HAMMER_CFG = {
    "enabled": True,
    "channel": 0,
    "alpha": 1e-3,
    "fband": (500, 10000),
}

RUNTIME_CFG = {
    "max_workers": min(32, os.cpu_count() or 4),
    "sample_rate": 1_000_000,
    "sensor_map": {"1": 0, "2": 1, "3": 2, "4": 3},
}

AE_CFG = {
    "backbone": "tae",       # "cnn" / "dnn" / "tae"
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "bottleneck_dim": 28,
    "epochs": 80,
    "batch_size": 64,
    "lr": 1e-3,
    "weight_decay": 1e-5,
}

DNN_AE_CFG = {
    "h1": 256,
    "h2": 128,
    "dropout": 0.0,
}

CNN_AE_CFG = {
    "base_ch": 32,
    "num_blocks": 2,
    "dropout": 0.0,
}

GPR_FEATURE_CFG = {
    "window_bins": 12,
    "use_envelope": True,
    "use_fft_feats": True,
    "nfft": 2048,
    "force_rebuild_feats": True,
    "band_edges": [0.0, 0.05, 0.10, 0.20, 0.35, 0.50],
}

KNOCK_AE_FEATURE_SUFFIXES = [
    "Mean",
    "Std",
    "RMS",
    "Kurtosis",
    "ZCR",
    "CrestFactor",
    "ImpulseFactor",
    "SpecEntropy",
    "RMSFreq",
    "SBW",
]

GPR_AE_FEATURES = [
    "EnvMax",
    "EnvEnergy",
    "EnvWidth50",
    "EnvPkNorm",
    "EnvT50Norm",
    "PeakCount",
    "Peak12AmpRatio",
    "SpecEntropy",
    "SpecCentroid",
    "SpecBW",
]

MODEL_REPORT_CFG = {
    "print_model_params": True,
    "print_all_backbones": False,
}

# ===== 兼容旧变量名：下面主流程无需改调用方式 =====
DATA_NAME = PROJECT_CFG["data_name"]
GT_PATH = PROJECT_CFG["gt_path"]
BASE_SEED = PROJECT_CFG["base_seed"]

KNOCK_ROOT = PATH_CFG["knock_root"]
GPR_ROOT = PATH_CFG["gpr_root"]
EXPORT_ROOT = PATH_CFG["export_root"]

HEALTHY_MASK_MODE = HEALTHY_REGION_CFG["mode"]
HEALTHY_FIRST_COLS = HEALTHY_REGION_CFG["first_cols"]
HEALTHY_COL_LIST = list(HEALTHY_REGION_CFG["col_list"])
HEALTHY_SAVE_MASKS = HEALTHY_REGION_CFG["save_masks"]

TAE_EXPORT_LATENT = TAE_EXPORT_CFG["export_latent"]
TAE_EXPORT_TOPK = TAE_EXPORT_CFG["export_topk"]
TAE_KNOCK_FIG_FEATURES = list(TAE_EXPORT_CFG["knock_fig_features"])
TAE_GPR_FIG_FEATURES = list(TAE_EXPORT_CFG["gpr_fig_features"])
TAE_TSNE_MAX_SAMPLES = TAE_EXPORT_CFG["tsne_max_samples"]
TAE_TSNE_RANDOM_STATE = TAE_EXPORT_CFG["tsne_random_state"]
TAE_TSNE_PERPLEXITY = TAE_EXPORT_CFG["tsne_perplexity"]

USE_CHANNELS = list(EXPERIMENT_CFG["use_channels"])
DEFAULT_EXP_TAG = EXPERIMENT_CFG["default_exp_tag"]
EXP_TAG = None

USE_HAMMER_DECONV = HAMMER_CFG["enabled"]
HAMMER_CH = HAMMER_CFG["channel"]
DECONV_ALPHA = HAMMER_CFG["alpha"]
DECONV_FBAND = HAMMER_CFG["fband"]

MAX_WORKERS = RUNTIME_CFG["max_workers"]
SR = RUNTIME_CFG["sample_rate"]
SENSOR_MAP = dict(RUNTIME_CFG["sensor_map"])

AE_BACKBONE = AE_CFG["backbone"]
DEVICE = AE_CFG["device"]
AE_BOTTLENECK_DIM = AE_CFG["bottleneck_dim"]
AE_EPOCHS = AE_CFG["epochs"]
AE_BATCH_SIZE = AE_CFG["batch_size"]
AE_LR = AE_CFG["lr"]
AE_WEIGHT_DECAY = AE_CFG["weight_decay"]

DNN_H1 = DNN_AE_CFG["h1"]
DNN_H2 = DNN_AE_CFG["h2"]
DNN_DROPOUT = DNN_AE_CFG["dropout"]

CNN_BASE_CH = CNN_AE_CFG["base_ch"]
CNN_NUM_BLOCKS = CNN_AE_CFG["num_blocks"]
CNN_DROPOUT = CNN_AE_CFG["dropout"]

GPR_WINDOW_BINS = GPR_FEATURE_CFG["window_bins"]
GPR_USE_ENVELOPE = GPR_FEATURE_CFG["use_envelope"]
GPR_USE_FFT_FEATS = GPR_FEATURE_CFG["use_fft_feats"]
GPR_NFFT = GPR_FEATURE_CFG["nfft"]
GPR_FORCE_REBUILD_FEATS = GPR_FEATURE_CFG["force_rebuild_feats"]
GPR_BAND_EDGES = list(GPR_FEATURE_CFG["band_edges"])

PRINT_MODEL_PARAMS = MODEL_REPORT_CFG["print_model_params"]
PRINT_ALL_BACKBONES = MODEL_REPORT_CFG["print_all_backbones"]

DIRS = {
    # -------- Knock（4通道）--------
    "RAW": os.path.join(KNOCK_ROOT, DATA_NAME),
    "NPY": os.path.join(KNOCK_ROOT, "npy_merged", DATA_NAME),
    "FEAT": os.path.join(KNOCK_ROOT, "feature_extr", DATA_NAME),
    "MAT": os.path.join(KNOCK_ROOT, "feature_extr", DATA_NAME, "feature_matrix"),
    "STACK": os.path.join(KNOCK_ROOT, "stack_results", DATA_NAME),
    # -------- AE / Bayes 输出 --------
    "AE": os.path.join(KNOCK_ROOT, "AE_results", DATA_NAME),
    "BAYES": os.path.join(EXPORT_ROOT, "bayes_results", DATA_NAME),
    # -------- GPR（方案B）--------
    "GPR_MAT": os.path.join(GPR_ROOT, "GPR_resample_mat"),
    "GPR_STACK": os.path.join(GPR_ROOT, "GPR_stack_results", DATA_NAME),
}

for d in DIRS.values():
    os.makedirs(d, exist_ok=True)


# ===========================================================
# 额外导出（统一保存到你指定位置）
# ===========================================================

def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def _exp_dir(*parts):
    if EXP_TAG is None:
        d = os.path.join(EXPORT_ROOT, DATA_NAME, *parts)
    else:
        d = os.path.join(EXPORT_ROOT, DATA_NAME, f"exp_{EXP_TAG}", *parts)
    _ensure_dir(d)
    return d


def _get_ae_out_dir():
    if EXP_TAG is None:
        d = DIRS["AE"]
    else:
        d = os.path.join(DIRS["AE"], f"exp_{EXP_TAG}")
    _ensure_dir(d)
    return d


RUNTIME_STAGE_ORDER = [
    "Data reading",
    "Preprocessing",
    "Feature extraction",
    "TAE training/inference",
    "Fusion",
]

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
RUNTIME_FIG_DIR = os.path.join(SCRIPT_DIR, "result", "aic_figures")
RUNTIME_TEMPLATE_CSV = os.path.join(RUNTIME_FIG_DIR, "runtime_breakdown_template.csv")
RUNTIME_SUMMARY_CSV = os.path.join(RUNTIME_FIG_DIR, "runtime_breakdown_measured.csv")
RUNTIME_DETAILS_CSV = os.path.join(RUNTIME_FIG_DIR, "runtime_breakdown_details.csv")
RUNTIME_META_TXT = os.path.join(RUNTIME_FIG_DIR, "runtime_profile_meta.txt")

RUNTIME_STAGE_SECONDS = {stage: 0.0 for stage in RUNTIME_STAGE_ORDER}
RUNTIME_DETAIL_ROWS = []


def _reset_runtime_profile():
    global RUNTIME_STAGE_SECONDS, RUNTIME_DETAIL_ROWS
    RUNTIME_STAGE_SECONDS = {stage: 0.0 for stage in RUNTIME_STAGE_ORDER}
    RUNTIME_DETAIL_ROWS = []


def _record_runtime(stage: str, label: str, seconds: float):
    sec = float(seconds)
    if stage not in RUNTIME_STAGE_SECONDS:
        raise KeyError(f"Unknown runtime stage: {stage}")

    RUNTIME_STAGE_SECONDS[stage] += sec
    RUNTIME_DETAIL_ROWS.append({
        "Stage": stage,
        "Label": label,
        "Seconds": sec,
    })
    print(f"[Runtime] {stage:<24} {label:<44} {sec:8.2f}s")


@contextmanager
def runtime_block(stage: str, label: str):
    t_start = time.perf_counter()
    try:
        yield
    finally:
        _record_runtime(stage, label, time.perf_counter() - t_start)


def timed_runtime_call(stage: str, label: str, func, *args, **kwargs):
    with runtime_block(stage, label):
        return func(*args, **kwargs)


def apply_run_mode_from_env():
    run_mode = str(os.environ.get("PIPELINE_RUN_MODE", "")).strip().lower()
    if not run_mode:
        return ""

    profiles = {
        "profile_from_intermediate": {
            "step1_txt2npy": False,
            "step2_feature": True,
            "step3_matrix": True,
            "step4_stack": True,
            "step0B_gpr_extract_features": True,
            "step5A_AE_knock": True,
            "step5B_AE_gpr": True,
            "step6_bayes_fusion": True,
        },
        "profile_full_raw": {
            "step1_txt2npy": True,
            "step2_feature": True,
            "step3_matrix": True,
            "step4_stack": True,
            "step0B_gpr_extract_features": True,
            "step5A_AE_knock": True,
            "step5B_AE_gpr": True,
            "step6_bayes_fusion": True,
        },
    }

    if run_mode not in profiles:
        raise ValueError(
            f"Unsupported PIPELINE_RUN_MODE={run_mode}. "
            f"Available: {sorted(profiles.keys())}"
        )

    RUN_SWITCH.update(profiles[run_mode])
    print(f"[Runtime] Applied run mode: {run_mode}")
    print(f"[Runtime] RUN_SWITCH = {RUN_SWITCH}")
    return run_mode


def write_runtime_profile(total_seconds: float, run_mode: str = ""):
    _ensure_dir(RUNTIME_FIG_DIR)

    summary_df = pd.DataFrame({
        "Stage": RUNTIME_STAGE_ORDER,
        "Seconds": [float(RUNTIME_STAGE_SECONDS[s]) for s in RUNTIME_STAGE_ORDER],
    })
    summary_df.to_csv(RUNTIME_TEMPLATE_CSV, index=False)
    summary_df.to_csv(RUNTIME_SUMMARY_CSV, index=False)

    detail_df = pd.DataFrame(RUNTIME_DETAIL_ROWS)
    detail_df.to_csv(RUNTIME_DETAILS_CSV, index=False)

    meta_lines = [
        f"TotalSeconds={float(total_seconds):.4f}",
        f"RunMode={run_mode or 'default'}",
        f"ExperimentTag={EXP_TAG if EXP_TAG is not None else 'None'}",
        f"Device={DEVICE}",
        f"Epochs={AE_EPOCHS}",
        f"BatchSize={AE_BATCH_SIZE}",
        f"DataName={DATA_NAME}",
    ]
    with open(RUNTIME_META_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(meta_lines) + "\n")

    print(f"[Runtime] Summary CSV -> {RUNTIME_SUMMARY_CSV}")
    print(f"[Runtime] Template CSV -> {RUNTIME_TEMPLATE_CSV}")
    print(f"[Runtime] Detail CSV -> {RUNTIME_DETAILS_CSV}")


# ===========================================================
# 2) 工具函数
# ===========================================================
def build_healthy_mask_first_cols(H, W, k):
    k = int(max(1, min(k, W)))
    m = np.zeros((H, W), dtype=bool)
    m[:, :k] = True
    return m


def export_train_tsne_dbscan(
    z_train,                    # (N, D)
    out_dir,
    prefix="Knock",
    rc_train=None,              # (N, 2) 可选：行列坐标
    seed=0,
    pca_dim=10,
    eps_percentile=90,
    min_samples_ratio=0.02,
    small_cluster_ratio=0.01,
    annotate_max=15,
):
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from sklearn.neighbors import NearestNeighbors
    from sklearn.cluster import DBSCAN

    os.makedirs(out_dir, exist_ok=True)

    z = np.asarray(z_train, dtype=np.float32)
    if z.ndim != 2:
        raise ValueError(f"[tSNE] z_train must be 2D, got {z.shape}")

    N, D = z.shape
    if N < 5:
        print(f"[tSNE] N={N} 太少，跳过。")
        return

    # ---------- 1) 标准化 ----------
    z_std = StandardScaler().fit_transform(z)

    # ---------- 2) PCA 预降维（可选但推荐） ----------
    pca_dim_eff = int(min(pca_dim, D, N - 1))
    if pca_dim_eff >= 2:
        z_vis_in = PCA(n_components=pca_dim_eff, svd_solver="randomized", random_state=seed).fit_transform(z_std)
    else:
        z_vis_in = z_std

    # ---------- 3) t-SNE：perplexity 必须 < N ----------
    # 稳妥取值：5~15 且 < N
    perp = int(np.clip(N // 10, 5, 15))
    perp = int(min(perp, max(2, N - 1)))

    tsne = TSNE(
        n_components=2,
        perplexity=perp,
        learning_rate="auto",
        init="pca",
        random_state=seed,
        metric="euclidean",
        early_exaggeration=12.0,
    )

    emb = tsne.fit_transform(z_vis_in)  # ✅ tsne 一定已定义

    # ---------- 4) DBSCAN：eps 用 kNN 距离分位数自动定 ----------
    k = min(10, N - 1)
    nn = NearestNeighbors(n_neighbors=k).fit(emb)
    knn_dist = nn.kneighbors(emb)[0][:, -1]
    eps = float(np.percentile(knn_dist, eps_percentile))
    min_samples = int(max(5, round(N * min_samples_ratio)))

    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(emb)

    # 小簇也当成离群
    uniq, cnt = np.unique(labels[labels >= 0], return_counts=True)
    small_thr = int(max(3, round(N * small_cluster_ratio)))
    small_clusters = set(uniq[cnt < small_thr].tolist())

    outlier_mask = (labels == -1) | np.isin(labels, list(small_clusters))
    inlier_mask = ~outlier_mask

    # ---------- 5) 绘图 ----------
    fig, ax = plt.subplots(figsize=(7.2, 6.2), dpi=220)

    # ✅ 删除六边形底图（就是这一行以前的 hexbin）
    # ax.hexbin(emb[:, 0], emb[:, 1], gridsize=45, bins="log", mincnt=1, alpha=0.22)

    # ✅ 正常点：蓝点
    if np.any(inlier_mask):
        ax.scatter(
            emb[inlier_mask, 0], emb[inlier_mask, 1],
            c="tab:blue",
            s=22, alpha=0.85, linewidths=0,
            label="Inliers"
        )

    # ✅ 离群点：红点（按你要求：红点，不是红叉）
    if np.any(outlier_mask):
        ax.scatter(
            emb[outlier_mask, 0], emb[outlier_mask, 1],
            c="crimson",
            s=30, alpha=0.95, linewidths=0,
            label="Outliers"
        )

        # # 你原来就有的标注（不算乱加东西），保留
        # idxs = np.where(outlier_mask)[0][:annotate_max]
        # for i in idxs:
        #     if rc_train is not None:
        #         txt = f"({int(rc_train[i, 0])},{int(rc_train[i, 1])})"
        #     else:
        #         txt = f"#{i}"
        #     ax.annotate(txt, (emb[i, 0], emb[i, 1]), xytext=(5, 5),
        #                 textcoords="offset points", fontsize=8)

    ax.set_title(f"t-SNE of Bottleneck z (Train-only, {prefix})\nDBSCAN outliers", fontsize=14)
    ax.set_xlabel("t-SNE-1")
    ax.set_ylabel("t-SNE-2")
    ax.legend(loc="lower left", frameon=True)
    ax.grid(alpha=0.18)
    fig.tight_layout()

    out_png = os.path.join(out_dir, f"{prefix}_train_tSNE_DBSCAN.png")
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)

    # ---------- 6) 导出 CSV ----------
    out_csv = os.path.join(out_dir, f"{prefix}_train_tSNE_DBSCAN_labels.csv")
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write("i,tsne1,tsne2,label,is_outlier,row,col\n")
        for i in range(N):
            r = int(rc_train[i,0]) if rc_train is not None else -1
            c = int(rc_train[i,1]) if rc_train is not None else -1
            f.write(f"{i},{emb[i,0]:.6f},{emb[i,1]:.6f},{int(labels[i])},{int(outlier_mask[i])},{r},{c}\n")

    print(f"[OK] Saved: {out_png}")
    print(f"[OK] Saved: {out_csv}")
    print(f"[DBSCAN] N={N}, perplexity={perp}, eps={eps:.4f}, min_samples={min_samples}, outliers={outlier_mask.sum()}")
def export_latent_tsne(
    z_all: np.ndarray,
    score_all: np.ndarray,
    train_mask_1d: np.ndarray,
    out_dir: str,
    tag: str,
):
    """
    z_all:     [N, D] bottleneck vectors
    score_all: [N,]   anomaly score (e.g., mean abs recon error)
    train_mask_1d: [N,] bool, points used for training (healthy_mask)
    """
    os.makedirs(out_dir, exist_ok=True)

    N = z_all.shape[0]
    if N < 10:
        print(f"[t-SNE] Too few samples (N={N}), skip.")
        return

    # ---------- 可选抽样（保证速度） ----------
    rng = np.random.default_rng(BASE_SEED)
    if N > TAE_TSNE_MAX_SAMPLES:
        idx = rng.choice(N, size=TAE_TSNE_MAX_SAMPLES, replace=False)
        z = z_all[idx]
        score = score_all[idx]
        tr = train_mask_1d[idx].astype(bool)
    else:
        idx = np.arange(N)
        z = z_all
        score = score_all
        tr = train_mask_1d.astype(bool)

    # ---------- 保存原始瓶颈特征 ----------
    np.save(os.path.join(out_dir, f"{tag}_z_all.npy"), z_all.astype(np.float32))
    np.save(os.path.join(out_dir, f"{tag}_score_all.npy"), score_all.astype(np.float32))

    # ---------- t-SNE 前可选 PCA（更稳一点） ----------
    z_in = z
    # ---------- robust defaults (avoid NameError) ----------
    seed = int(globals().get("TAE_TSNE_RANDOM_STATE", globals().get("BASE_SEED", 43)))
    perp0 = int(globals().get("TAE_TSNE_PERPLEXITY", 30))

    # ---------- PCA pre-reduction (optional) ----------
    # 注意：PCA 的 n_components 必须 <= (n_samples - 1)
    n_samples = int(z_in.shape[0])
    n_dim = int(z_in.shape[1])

    if n_dim > 10 and n_samples >= 3:
        pca_dim = min(10, n_dim, n_samples - 1)
        if pca_dim >= 2:
            # random_state 只有在 randomized solver 下才真正起作用
            z_in = PCA(n_components=pca_dim, svd_solver="randomized", random_state=seed).fit_transform(z_in)

    # ---------- t-SNE perplexity must be < n_samples ----------
    # 你的写法本质是：perp <= (n-1)//3，同时不小于 5
    # 再加一道硬约束：perp <= n_samples - 1
    perp = int(min(perp0, max(5, (n_samples - 1) // 3)))
    perp = int(min(perp, max(2, n_samples - 1)))  # 防止 perp >= n_samples

    tsne = TSNE(
        n_components=2,
        perplexity=perp,
        init="pca",
        learning_rate="auto",
        random_state=seed,
    )
    emb = tsne.fit_transform(z_in)  # [n,2]

    # ---------- 保存坐标 ----------
    df = pd.DataFrame({
        "idx": idx.astype(int),
        "tsne_x": emb[:, 0],
        "tsne_y": emb[:, 1],
        "score": score.astype(float),
        "is_train": tr.astype(int),
    })
    df.to_csv(os.path.join(out_dir, f"{tag}_tsne_xy.csv"), index=False)

    # ---------- 图1：只用蓝点/红点（训练点 vs 非训练点） ----------
    plt.figure(figsize=(6, 5))

    plt.scatter(emb[tr, 0], emb[tr, 1],
                s=10, c="tab:blue", marker="o", alpha=0.85,
                label="Inliers (train/healthy)")

    plt.scatter(emb[~tr, 0], emb[~tr, 1],
                s=30, c="crimson", marker="o", alpha=0.95,
                edgecolors="k", linewidths=0.3,
                label="Outliers (non-train)")

    plt.legend(frameon=True)
    plt.title(f"t-SNE of Bottleneck z ({tag})")
    plt.xlabel("t-SNE-1")
    plt.ylabel("t-SNE-2")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{tag}_tSNE_blue_red.png"), dpi=200)
    plt.close()

    # ---------- 图2：训练点 vs 其余点（用于说明“健康流形”由训练主导） ----------
    plt.figure(figsize=(6, 5))
    plt.scatter(emb[tr, 0], emb[tr, 1], s=10, label="Train points (healthy_mask)")
    plt.scatter(emb[~tr, 0], emb[~tr, 1], s=10, label="Other points")
    plt.legend()
    plt.title(f"t-SNE of Bottleneck z ({tag})")
    plt.xlabel("t-SNE-1")
    plt.ylabel("t-SNE-2")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{tag}_tSNE_train_vs_others.png"), dpi=200)
    plt.close()

    print(f"[✓] t-SNE exported -> {out_dir} (perplexity={perp}, n={emb.shape[0]})")

def get_unified_healthy_mask(expected_hw, ae_out_dir=None, tag=""):
    H, W = expected_hw

    if HEALTHY_MASK_MODE == "FIRST_COLS":
        m = build_healthy_mask_first_cols(H, W, HEALTHY_FIRST_COLS)

    elif HEALTHY_MASK_MODE == "COL_LIST":
        cols = [int(c) for c in HEALTHY_COL_LIST]
        cols = [c for c in cols if 0 <= c < W]
        m = np.zeros((H, W), dtype=bool)
        if len(cols) > 0:
            m[:, cols] = True
        else:
            # 兜底：如果列表无效，至少保证有1列
            m[:, :1] = True

    else:
        raise ValueError(f"Unknown HEALTHY_MASK_MODE={HEALTHY_MASK_MODE}")

    if (m is not None) and HEALTHY_SAVE_MASKS and (ae_out_dir is not None):
        np.save(
            os.path.join(ae_out_dir, f"healthy_mask_{tag}_{H}x{W}_{HEALTHY_MASK_MODE}.npy"),
            m.astype(np.uint8)
        )
    return m

# ===========================================================
# 2.2) AE 训练曲线导出（两条曲线 + 一个CSV）
# ===========================================================
def export_ae_train_logs(ae_out_dir: str, modality_key: str, loss_hist: list):
    """
    modality_key: "knock" or "gpr"
    输出：
      - train_logs/{modality_key}_train_loss.png
      - train_logs/AE_train_loss_curves.csv        (合并：epoch + knock_loss + gpr_loss)
      - train_logs/AE_train_loss_compare.png       (两条曲线同图)
    """
    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    log_dir = os.path.join(ae_out_dir, "train_logs")
    os.makedirs(log_dir, exist_ok=True)

    col = f"{modality_key}_loss"
    df_new = pd.DataFrame({
        "epoch": np.arange(1, len(loss_hist) + 1, dtype=int),
        col: np.asarray(loss_hist, dtype=float)
    })

    # --- 1) 更新合并 CSV（只保留一个文件）---
    combined_csv = os.path.join(log_dir, "AE_train_loss_curves.csv")
    if os.path.exists(combined_csv):
        df_all = pd.read_csv(combined_csv)
        if "epoch" in df_all.columns:
            df_all["epoch"] = df_all["epoch"].astype(int)
        # 若已有同名列，先删除再合并（避免重复列）
        if col in df_all.columns:
            df_all = df_all.drop(columns=[col])
        df_all = pd.merge(df_all, df_new, on="epoch", how="outer")
    else:
        df_all = df_new.copy()

    df_all = df_all.sort_values("epoch").reset_index(drop=True)
    df_all.to_csv(combined_csv, index=False)

    # --- 2) 单独曲线图 ---
    plt.figure(figsize=(6, 4))
    plt.plot(df_new["epoch"], df_new[col])
    plt.title(f"{modality_key.upper()} AE Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, f"{modality_key}_train_loss.png"), dpi=150)
    plt.close()

    # --- 3) 两条曲线同图（如果另一条还没跑，也不会报错）---
    plt.figure(figsize=(7, 4))
    for c in df_all.columns:
        if c == "epoch":
            continue
        plt.plot(df_all["epoch"], df_all[c], label=c.replace("_loss", "").upper())
    plt.title("AE Training Loss (Knock vs GPR)")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, "AE_train_loss_compare.png"), dpi=150)
    plt.close()

    print(f"[✓] AE train logs updated -> {combined_csv}")


def _percentile_prob(err_2d, p_lo=5, p_hi=95):
    """
    err_2d: [N_pix, N_feat] -> prob in [0,1]
    """
    prob = np.zeros_like(err_2d, dtype=np.float32)
    for k in range(err_2d.shape[1]):
        col = err_2d[:, k]
        lo, hi = np.percentile(col, [p_lo, p_hi])
        denom = hi - lo if hi > lo else 1e-6
        prob[:, k] = np.clip((col - lo) / denom, 0.0, 1.0)
    return prob

def save_bayes_iter_loss(loss_hist, out_dir, tag):
    """
    保存 Bayes 迭代收敛过程：
      - {tag}_bayes_iter_loss.csv
      - {tag}_bayes_iter_loss.png
    """
    os.makedirs(out_dir, exist_ok=True)

    loss_hist = np.array(loss_hist, dtype=float).reshape(-1)
    csv_path = os.path.join(out_dir, f"{tag}_bayes_iter_loss.csv")
    np.savetxt(csv_path, loss_hist, delimiter=",")

    plt.figure(figsize=(6, 3))
    plt.plot(loss_hist)
    plt.title(f"Bayes Iter Convergence - {tag}")
    plt.xlabel("Iteration")
    plt.ylabel("Mean |Post_new - Post|")
    plt.tight_layout()
    png_path = os.path.join(out_dir, f"{tag}_bayes_iter_loss.png")
    plt.savefig(png_path, dpi=150)
    plt.close()

    print(f"[✓] Bayes iter loss saved -> {csv_path} , {png_path}")


def _save_mean_map(mat, out_png, title):
    plt.figure(figsize=(12, 4))

    norm = colors.Normalize(vmin=0.0, vmax=1.0)   # 👈 只是“参考尺”
    im = plt.imshow(mat, cmap="cividis", norm=norm)

    cbar = plt.colorbar(im)
    cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
    cbar.set_label("Anomaly score (reference scale)")

    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()

def save_map_with_01_label(mat, out_png, title):
    """
    mat: 2D ndarray, 不做归一化
    colorbar: 固定显示 0–1（参考尺）
    """
    plt.figure(figsize=(12, 4))
    norm = colors.Normalize(vmin=0.0, vmax=1.0)
    im = plt.imshow(mat, cmap="cividis", norm=norm)
    cbar = plt.colorbar(im)
    cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
    cbar.set_label("Anomaly score (0–1 reference)")
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()

def resize_hw(arr, target_hw, order=1):
    Ht, Wt = target_hw
    if arr.ndim == 2:
        H, W = arr.shape
        return zoom(arr, (Ht / H, Wt / W), order=order)
    if arr.ndim == 3:
        H, W, C = arr.shape
        return zoom(arr, (Ht / H, Wt / W, 1.0), order=order)
    raise ValueError(f"resize_hw only supports 2D/3D, got shape={arr.shape}")

def numeric_sort_key(s):
    m = re.findall(r"\d+", s)
    return int(m[0]) if m else 10**9


def _repeat_columns(mat, factor=5):
    return np.repeat(mat, factor, axis=1)

# ===========================================================
# Model params counter (print + CSV)
# ===========================================================
_PARAM_PRINTED = set()

def count_model_params(model: torch.nn.Module):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return int(total), int(trainable)

def estimate_size_mb(total_params: int, bytes_per_param: int = 4):
    # fp32 约 4 bytes/param；fp16 约 2 bytes/param
    return total_params * bytes_per_param / (1024 ** 2)

def log_model_params_once(ae_out_dir: str, modality: str, backbone: str, n_feat: int, model: torch.nn.Module):
    key = (str(modality), str(backbone), int(n_feat))
    if key in _PARAM_PRINTED:
        return
    _PARAM_PRINTED.add(key)

    total, trainable = count_model_params(model)
    size_mb = estimate_size_mb(total, bytes_per_param=4)

    print(f"[Params] {modality} | {backbone} | n_feat={n_feat}: "
          f"total={total:,} ({total/1e6:.3f}M), "
          f"trainable={trainable:,} ({trainable/1e6:.3f}M), "
          f"size≈{size_mb:.2f} MB (fp32)")

    # 保存到 CSV（每个 ae_out_dir 一份）
    if ae_out_dir is not None:
        os.makedirs(ae_out_dir, exist_ok=True)
        csv_path = os.path.join(ae_out_dir, "model_params.csv")
        row = pd.DataFrame([{
            "Modality": modality,
            "Backbone": backbone,
            "N_feat": int(n_feat),
            "TotalParams": total,
            "TrainableParams": trainable,
            "SizeMB_fp32": float(size_mb),
        }])
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            df = pd.concat([df, row], ignore_index=True)
            df = df.drop_duplicates(subset=["Modality", "Backbone", "N_feat"], keep="last")
        else:
            df = row
        df.to_csv(csv_path, index=False)


# ===========================================================
# 2.1) 额外导出（把 64 列扩到 320 方便看图）
# ===========================================================
def export_map_2d(mat2d: np.ndarray, out_dir: str, base_name: str, title: str):
    """
    输出：
      - base_name.npy
      - base_name.png
    规则：
      - 若 H=29 -> 纵向 repeat 5 -> 145
      - 横向把 64 repeat 5 -> 320（便于视觉展示）
    """
    _ensure_dir(out_dir)
    H, W = mat2d.shape

    if H == 29:
        mat_row = np.repeat(mat2d, 5, axis=0)
    elif H == 145:
        mat_row = mat2d
    else:
        mat_row = resize_hw(mat2d, (145, W), order=1)

    mat_final = np.repeat(mat_row, 5, axis=1)
    if mat_final.shape != (145, 320):
        mat_final = resize_hw(mat_final, (145, 320), order=1)

    np.save(os.path.join(out_dir, f"{base_name}.npy"), mat_final.astype(np.float32))
    _save_mean_map(mat_final, os.path.join(out_dir, f"{base_name}.png"), title)

def export_map_raw_and_vis(mat2d: np.ndarray, out_dir: str, base_name: str, title: str):
    """
    保存两套：
      - raw: H×W -> base_name_raw.(npy/csv/png)     ✅不扩充
      - vis: 145×320 -> base_name_vis.(npy/png)    ✅扩充展示版（复用 export_map_2d）
    """
    _ensure_dir(out_dir)

    # ---------- raw 数值 ----------
    raw_npy = os.path.join(out_dir, f"{base_name}_raw.npy")
    raw_csv = os.path.join(out_dir, f"{base_name}_raw.csv")
    np.save(raw_npy, mat2d.astype(np.float32))
    np.savetxt(raw_csv, mat2d.astype(np.float32), delimiter=",")

    # ---------- raw 图（不扩充，直接 H×W） ----------
    raw_png = os.path.join(out_dir, f"{base_name}_raw.png")
    _save_mean_map(
        mat2d,
        raw_png,
        f"{title} (raw {mat2d.shape[0]}×{mat2d.shape[1]})"
    )

    # ---------- vis 图（扩充到 145×320，便于观看） ----------
    export_map_2d(
        mat2d,
        out_dir,
        f"{base_name}_vis",
        f"{title} (vis expanded)"
    )


def export_gpr_ae_maps(prob_gpr, out_dir, repeat_col=5):
    """
    prob_gpr: (145, 64, N_gpr)
    仅用于展示：
      - 多特征取均值
      - 每列复制 repeat_col 次
      - 不归一化
    """
    _ensure_dir(out_dir)

    gpr_mean = np.mean(prob_gpr, axis=2)
    gpr_vis = _repeat_columns(gpr_mean, repeat_col)

    np.save(
        os.path.join(out_dir, "GPR_AE_Mean.npy"),
        gpr_vis.astype(np.float32)
    )

    save_map_with_01_label(
        gpr_vis,
        os.path.join(out_dir, "GPR_AE_Mean.png"),
        title="GPR AE Anomaly Map (Mean over features)"
    )


def _pick_feature_indices(prob3d: np.ndarray, feat_names, healthy_mask: np.ndarray, topk: int):
    """
    自动挑选 TopK：用“非健康区均值 - 健康区均值”作为对比度分数
    """
    H, W, N = prob3d.shape
    P = prob3d.reshape(-1, N)  # [H*W, N]
    if healthy_mask is None:
        score = P.mean(axis=0)  # 没有健康区就退化成均值（一般不建议）
    else:
        hm = healthy_mask.reshape(-1).astype(bool)
        dm = ~hm
        if dm.sum() < 1 or hm.sum() < 1:
            score = P.mean(axis=0)
        else:
            score = P[dm].mean(axis=0) - P[hm].mean(axis=0)

    idx = np.argsort(score)[::-1]
    idx = idx[: max(1, int(topk))]
    return idx


def export_tae_prob_maps_smallset(
    prob3d: np.ndarray,
    feat_names,
    healthy_mask: np.ndarray,
    out_dir: str,
    prefix: str,
    force_features: list = None,
    topk: int = 6,
):
    """
    只导出少量 TAE 概率矩阵图：
      - Mean over features
      - 若 force_features 非空：导出指定特征
      - 否则：自动按对比度挑 TopK
    """
    _ensure_dir(out_dir)
    feat_names = [str(x) for x in feat_names]

    # 1) 全特征均值图
    mean_map = np.mean(prob3d, axis=2)
    export_map_raw_and_vis(mean_map, out_dir, f"{prefix}_ProbMean", f"{prefix} TAE Prob (Mean over features)")

    # 2) 选特征
    sel_idx = []

    if force_features is not None and len(force_features) > 0:
        # 按名字匹配（尽量鲁棒）
        for q in force_features:
            q = str(q)
            hit = None
            for i, nm in enumerate(feat_names):
                if (nm == q) or nm.endswith(q) or (q in nm):
                    hit = i
                    break
            if hit is not None:
                sel_idx.append(hit)
    else:
        sel_idx = list(_pick_feature_indices(prob3d, feat_names, healthy_mask, topk))

    # 去重
    sel_idx = list(dict.fromkeys(sel_idx))

    # 3) 导出单特征概率图
    for i in sel_idx:
        nm = feat_names[i]
        m2d = prob3d[:, :, i]
        export_map_raw_and_vis(m2d, out_dir, f"{prefix}_{i:03d}_{nm}", f"{prefix} TAE Prob – {nm}")

    # 4) 记录导出清单（方便论文写“展示了哪些特征”）
    rec = pd.DataFrame({"Index": sel_idx, "Feature": [feat_names[i] for i in sel_idx]})
    rec.to_csv(os.path.join(out_dir, f"{prefix}_exported_features.csv"), index=False)


# ===========================================================
# 3) Step1: TXT -> NPY（Knock）
# ===========================================================
def process_one_step1(prefix, sensors):
    for sid in SENSOR_MAP:
        if sid not in sensors:
            return None
    try:
        merged = None
        sorted_sensors = sorted(sensors.items(), key=lambda x: SENSOR_MAP[x[0]])
        for sid, path in sorted_sensors:
            df = pd.read_csv(path, header=None, sep=r"\s+", engine="c")
            vals = df.values.flatten()
            if merged is None:
                merged = np.zeros((len(vals), 4), dtype=np.float32)
            L = min(len(vals), len(merged))
            merged[:L, SENSOR_MAP[sid]] = vals[:L]
        np.save(os.path.join(DIRS["NPY"], prefix + ".npy"), merged)
        return prefix
    except Exception as e:
        return f"ERROR: {prefix} - {e}"

def step1_convert_txt_to_npy():
    print(f"\n=== Step1: TXT -> NPY (Workers: {MAX_WORKERS}) ===")
    files = [f for f in os.listdir(DIRS["RAW"]) if f.endswith(".txt")]
    groups = {}
    for fname in files:
        if "#" not in fname:
            continue
        try:
            prefix, sid_ext = fname.split("#")
            sid = sid_ext.replace(".txt", "")
            if sid in SENSOR_MAP:
                groups.setdefault(prefix, {})[sid] = os.path.join(DIRS["RAW"], fname)
        except ValueError:
            continue

    tasks = list(groups.items())
    print(f"检测到 {len(tasks)} 组测点数据")
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = [ex.submit(process_one_step1, p, s) for p, s in tasks]
        for _ in tqdm(as_completed(futures), total=len(futures), desc="Converting"):
            pass
    print("Step1 完成")


# ===========================================================
# 4) Step2: Knock 特征提取
# ===========================================================
def hammer_deconvolution(f, y, sr, alpha=1e-2, fband=(500, 15000)):
    f = f.astype(np.float64)
    y = y.astype(np.float64)
    N = min(len(f), len(y))
    f = f[:N]
    y = y[:N]

    win = np.hanning(N)
    f = f * win
    y = y * win

    Nfft = 2 ** int(np.ceil(np.log2(N)))
    F = np.fft.rfft(f, Nfft)
    Y = np.fft.rfft(y, Nfft)

    eps = alpha * np.mean(np.abs(F) ** 2)
    H = (Y * np.conj(F)) / (np.abs(F) ** 2 + eps)

    freqs = np.fft.rfftfreq(Nfft, 1 / sr)
    mask = (freqs >= fband[0]) & (freqs <= fband[1])
    H[~mask] = 0.0

    h = np.fft.irfft(H, Nfft)[:N]
    return h.astype(np.float32)

def extract_features_for_file(path: str):
    fname = os.path.basename(path)
    out = {"File": fname}
    try:
        data = np.load(path)
    except Exception:
        out["Error"] = "NPY 加载失败"
        return out
    if data.ndim != 2 or data.shape[1] != 4:
        out["Error"] = f"格式错误 shape={data.shape}"
        return out

    N = data.shape[0]
    freqs = np.fft.rfftfreq(N, 1 / SR)
    mask_f1 = freqs < 2000
    mask_f2 = (freqs >= 2000) & (freqs < 5000)
    mask_f3 = (freqs >= 5000) & (freqs < 10000)
    mask_f4 = freqs >= 10000

    hammer = data[:, HAMMER_CH]

    for ch in range(4):
        if USE_HAMMER_DECONV and ch != HAMMER_CH:
            sig = hammer_deconvolution(hammer, data[:, ch], SR, DECONV_ALPHA, DECONV_FBAND)
        else:
            sig = data[:, ch]

        abs_sig = np.abs(sig)
        mean_val = np.mean(sig)
        std_val = np.std(sig)
        rms_val = np.sqrt(np.mean(sig ** 2))
        max_val = np.max(abs_sig)

        out[f"Ch{ch}_Mean"] = float(mean_val)
        out[f"Ch{ch}_Std"]  = float(std_val)
        out[f"Ch{ch}_RMS"]  = float(rms_val)
        out[f"Ch{ch}_Kurt"] = float(kurtosis(sig, fisher=True, bias=False))
        out[f"Ch{ch}_Skew"] = float(skew(sig, bias=False))
        out[f"Ch{ch}_P2P"]  = float(np.ptp(sig))
        out[f"Ch{ch}_ZCR"]  = float(np.sum(sig[:-1] * sig[1:] < 0) / (N - 1 + 1e-12))

        out[f"Ch{ch}_CrestFactor"]   = float(max_val / (rms_val + 1e-9))
        out[f"Ch{ch}_ImpulseFactor"] = float(max_val / (np.mean(abs_sig) + 1e-9))
        out[f"Ch{ch}_ShapeFactor"]   = float(rms_val / (np.mean(abs_sig) + 1e-9))
        out[f"Ch{ch}_MarginFactor"]  = float(max_val / (np.mean(np.sqrt(abs_sig)) ** 2 + 1e-9))

        fft_val = np.abs(np.fft.rfft(sig))
        sum_fft = np.sum(fft_val) + 1e-9
        psd = fft_val ** 2 / (N + 1e-12)

        sc = np.sum(freqs * fft_val) / sum_fft
        out[f"Ch{ch}_SC"] = float(sc)

        msk_freq = np.sum(freqs ** 2 * fft_val) / sum_fft
        out[f"Ch{ch}_RMSFreq"] = float(np.sqrt(msk_freq))

        sbw = np.sqrt(np.sum((freqs - sc) ** 2 * fft_val) / sum_fft)
        out[f"Ch{ch}_SBW"] = float(sbw)

        geo_mean = np.exp(np.mean(np.log(fft_val + 1e-12)))
        ari_mean = np.mean(fft_val) + 1e-12
        out[f"Ch{ch}_SpecFlatness"] = float(geo_mean / ari_mean)

        psd_norm = psd / (np.sum(psd) + 1e-9)
        out[f"Ch{ch}_SpecEntropy"] = float(entropy(psd_norm + 1e-12))

        pk_idx = int(np.argmax(fft_val))
        out[f"Ch{ch}_PeakFreq"] = float(freqs[pk_idx])
        out[f"Ch{ch}_PeakAmp"]  = float(fft_val[pk_idx])
        out[f"Ch{ch}_Energy"]   = float(np.sum(psd))

        out[f"Ch{ch}_F1"] = float(np.sum(fft_val[mask_f1] ** 2))
        out[f"Ch{ch}_F2"] = float(np.sum(fft_val[mask_f2] ** 2))
        out[f"Ch{ch}_F3"] = float(np.sum(fft_val[mask_f3] ** 2))
        out[f"Ch{ch}_F4"] = float(np.sum(fft_val[mask_f4] ** 2))

        wp = pywt.WaveletPacket(sig, wavelet="db3", maxlevel=3)
        nodes = wp.get_level(3, order="freq")
        eng_nodes = np.array([np.sum(n.data ** 2) for n in nodes], dtype=np.float64)

        for i, e_val in enumerate(eng_nodes):
            out[f"Ch{ch}_WP_E{i}"] = float(e_val)

        eng_norm = eng_nodes / (np.sum(eng_nodes) + 1e-12)
        out[f"Ch{ch}_WP_Entropy"] = float(entropy(eng_norm + 1e-12))
        out[f"Ch{ch}_WP_Std"]     = float(np.std(eng_nodes))

    return out

def step2_extract_features():
    print(f"\n=== Step2: Knock 特征提取 (Workers: {MAX_WORKERS}) ===")
    npy_files = [os.path.join(DIRS["NPY"], f) for f in os.listdir(DIRS["NPY"]) if f.endswith(".npy")]
    print(f"在 {DIRS['NPY']} 中找到 {len(npy_files)} 个 NPY 文件")

    results = []
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = [ex.submit(extract_features_for_file, p) for p in npy_files]
        for f in tqdm(as_completed(futures), total=len(futures), desc="Extracting"):
            results.append(f.result())

    df = pd.DataFrame(results)
    out_csv = os.path.join(DIRS["FEAT"], f"{DATA_NAME}_features.csv")
    if df.empty:
        print("[WARN] 特征提取结果为空。")
        return
    df.to_csv(out_csv, index=False)
    print(f"Step2 完成 -> {out_csv}")


# ===========================================================
# 5) Step3: Knock 矩阵化
# ===========================================================
def step3_matrix_transform():
    print("\n=== Step3: Knock 特征矩阵化 ===")
    feat_csv = os.path.join(DIRS["FEAT"], f"{DATA_NAME}_features.csv")
    if not os.path.exists(feat_csv):
        print("未找到特征表")
        return

    df = pd.read_csv(feat_csv)
    if "Error" in df.columns:
        df = df[df["Error"].isna()]
    if df.empty:
        print("[WARN] 特征 DataFrame 为空，退出 Step3。")
        return

    filenames = df["File"].values
    rows, cols = [], []
    for fname in filenames:
        digits = "".join(c for c in fname if c.isdigit())
        if len(digits) >= 4:
            rows.append(int(digits[-4:-2]))
            cols.append(int(digits[-2:]))
        else:
            rows.append(1)
            cols.append(1)

    rows = np.array(rows)
    cols = np.array(cols)
    nR, nC = int(np.max(rows)), int(np.max(cols))
    print(f"自动识别矩阵尺寸: {nR} x {nC}")

    feature_cols = [c for c in df.columns if c not in ["File", "Error"]]
    flat_indices = (rows - 1) * nC + (cols - 1)

    _ensure_dir(DIRS["MAT"])
    for feat in tqdm(feature_cols, desc="Creating Matrices"):
        mat = np.zeros(nR * nC, dtype=np.float32)
        vals = df[feat].values.astype(np.float32)
        np.put(mat, flat_indices, vals)
        mat = mat.reshape(nR, nC)
        np.savetxt(os.path.join(DIRS["MAT"], f"{feat}.csv"), mat, delimiter=",")


# ===========================================================
# 6) Step4: Knock 堆叠 feat_stack.npy
# ===========================================================
def step4_stack_feature_matrices():
    print("\n=== Step4: Knock 堆叠特征矩阵 ===")
    files = sorted([f for f in os.listdir(DIRS["MAT"]) if f.endswith(".csv")])
    if not files:
        print("无特征矩阵文件！")
        return

    mats, names = [], []
    heatmap_dir = os.path.join(DIRS["STACK"], "heatmaps")
    _ensure_dir(heatmap_dir)

    for i, f in enumerate(tqdm(files, desc="Stacking")):
        path = os.path.join(DIRS["MAT"], f)
        mat = np.loadtxt(path, delimiter=",").astype(np.float32)
        denom = mat.max() - mat.min()
        mat_norm = (mat - mat.min()) / (denom + 1e-12)
        mats.append(mat_norm)
        names.append(os.path.splitext(f)[0])

        if i < 10:
            plt.figure(figsize=(3, 3))
            plt.imshow(mat_norm, cmap="cividis")
            plt.axis("off")
            plt.title(names[-1], fontsize=8)
            plt.savefig(os.path.join(heatmap_dir, f"{names[-1]}.png"), dpi=100)
            plt.close()

    feat_stack = np.stack(mats, axis=2)  # H x W x N
    _ensure_dir(DIRS["STACK"])
    np.save(os.path.join(DIRS["STACK"], "feat_stack.npy"), feat_stack.astype(np.float32))
    np.save(os.path.join(DIRS["STACK"], "feat_names.npy"), np.array(names, dtype=object))
    print(f"[Knock] Stack Shape: {feat_stack.shape}")


# ===========================================================
# 7) Step0B: GPR raw(655×145×64) -> features(145×64×N_gpr)
# ===========================================================
def load_mat_2d(full_path):
    md = sio.loadmat(full_path)
    key = [k for k in md.keys() if not k.startswith("__")][0]
    a = md[key]
    if a.ndim != 2:
        raise ValueError(f"[GPR] mat not 2D: {full_path}, shape={a.shape}")
    a = a.astype(np.float32)
    # 常见：145×655 -> 转为 655×145
    if a.shape[0] < a.shape[1] and a.shape[1] >= 256:
        a = a.T
    return a  # (Nt, H)

def gpr_trace_features(trace_1d: np.ndarray, return_names: bool = False):
    x = trace_1d.astype(np.float32)
    Nt = x.size
    eps = 1e-12

    feats = []
    names = []

    def add(name, val):
        feats.append(float(val))
        names.append(str(name))

    # ---- 基础时域（9）----
    mean = np.mean(x)
    std  = np.std(x)
    rms  = np.sqrt(np.mean(x**2) + eps)
    abs_x = np.abs(x)
    maxabs = np.max(abs_x)
    p2p = np.ptp(x)
    zcr = np.sum(x[:-1] * x[1:] < 0) / (Nt - 1 + eps)
    sk = skew(x, bias=False)
    ku = kurtosis(x, fisher=True, bias=False)
    energy = np.mean(x**2)

    add("Mean", mean); add("Std", std); add("RMS", rms); add("MaxAbs", maxabs)
    add("P2P", p2p); add("ZCR", zcr); add("Skew", sk); add("Kurt", ku); add("Energy", energy)

    # ---- 鲁棒统计（6）----
    q25 = np.percentile(x, 25)
    q50 = np.percentile(x, 50)
    q75 = np.percentile(x, 75)
    iqr = q75 - q25
    mad = np.median(np.abs(x - q50))
    t = np.arange(Nt, dtype=np.float32)
    slope = np.polyfit(t, x, 1)[0]

    add("Q25", q25); add("Q50", q50); add("Q75", q75); add("IQR", iqr); add("MAD", mad); add("Slope", slope)

    # ---- 形状因子（4）----
    mean_abs = np.mean(abs_x) + eps
    crest_factor   = maxabs / (rms + eps)
    impulse_factor = maxabs / mean_abs
    shape_factor   = (rms + eps) / mean_abs
    margin_factor  = maxabs / ((np.mean(np.sqrt(abs_x + eps))**2) + eps)

    add("CrestFactor", crest_factor)
    add("ImpulseFactor", impulse_factor)
    add("ShapeFactor", shape_factor)
    add("MarginFactor", margin_factor)

    # ---- 包络（10）----
    if GPR_USE_ENVELOPE:
        env = np.abs(hilbert(x))
        env_max  = np.max(env)
        env_mean = np.mean(env)
        env_std  = np.std(env)
        env_skew = skew(env, bias=False)
        env_kurt = kurtosis(env, fisher=True, bias=False)
        env_energy = np.mean(env**2)

        env_pk = int(np.argmax(env))
        env_pk_norm = env_pk / max(Nt - 1, 1)

        thr10 = 0.10 * env_max
        thr50 = 0.50 * env_max
        idx10 = int(np.argmax(env >= thr10)) if env_max > 0 else 0
        idx50 = int(np.argmax(env >= thr50)) if env_max > 0 else 0
        idx10_norm = idx10 / max(Nt - 1, 1)
        idx50_norm = idx50 / max(Nt - 1, 1)

        width50 = (np.sum(env >= thr50) / Nt) if env_max > 0 else 0.0

        add("EnvMean", env_mean); add("EnvStd", env_std); add("EnvMax", env_max)
        add("EnvSkew", env_skew); add("EnvKurt", env_kurt); add("EnvEnergy", env_energy)
        add("EnvPkNorm", env_pk_norm); add("EnvT10Norm", idx10_norm); add("EnvT50Norm", idx50_norm)
        add("EnvWidth50", width50)
    else:
        for nm in ["EnvMean","EnvStd","EnvMax","EnvSkew","EnvKurt","EnvEnergy","EnvPkNorm","EnvT10Norm","EnvT50Norm","EnvWidth50"]:
            add(nm, 0.0)

    # ---- 多峰结构（7）----
    sig_peak = np.abs(hilbert(x)) if GPR_USE_ENVELOPE else abs_x
    prom = 0.05 * (np.max(sig_peak) + eps)
    height = 0.20 * (np.max(sig_peak) + eps)
    peaks, props = find_peaks(sig_peak, prominence=prom, height=height, distance=3)

    peak_count = float(len(peaks))
    p1_norm = p2_norm = 0.0
    prom1 = prom2 = 0.0
    peak12_amp_ratio = 0.0
    peak12_sep_norm  = 0.0

    if len(peaks) >= 1:
        order = np.argsort(props["peak_heights"])[::-1]
        p1 = peaks[order[0]]
        a1 = float(props["peak_heights"][order[0]])
        p1_norm = p1 / max(Nt - 1, 1)
        prom1 = float(props["prominences"][order[0]]) if "prominences" in props else 0.0

        if len(peaks) >= 2:
            p2 = peaks[order[1]]
            a2 = float(props["peak_heights"][order[1]])
            p2_norm = p2 / max(Nt - 1, 1)
            prom2 = float(props["prominences"][order[1]]) if "prominences" in props else 0.0
            peak12_amp_ratio = a1 / (a2 + eps)
            peak12_sep_norm  = abs(p1 - p2) / max(Nt - 1, 1)

    add("PeakCount", peak_count)
    add("Peak1PosNorm", p1_norm); add("Peak2PosNorm", p2_norm)
    add("Peak1Prom", prom1); add("Peak2Prom", prom2)
    add("Peak12AmpRatio", peak12_amp_ratio); add("Peak12SepNorm", peak12_sep_norm)

    # ---- 时窗能量（B=8）----
    B = int(GPR_WINDOW_BINS)
    win = Nt // B
    for b in range(B):
        s = b * win
        e = (b + 1) * win if b < B - 1 else Nt
        seg = x[s:e]
        add(f"WinE{b}", np.mean(seg**2))

    # ---- TKEO（3）----
    if Nt >= 3:
        tkeo = x[1:-1]**2 - x[:-2]*x[2:]
        add("TKEO_Mean", np.mean(tkeo))
        add("TKEO_Std", np.std(tkeo))
        add("TKEO_MaxAbs", np.max(np.abs(tkeo)))
    else:
        add("TKEO_Mean", 0.0); add("TKEO_Std", 0.0); add("TKEO_MaxAbs", 0.0)

    # ---- 频域（11 + bandE + bandR）----
    if GPR_USE_FFT_FEATS:
        X = np.abs(np.fft.rfft(x, n=GPR_NFFT)).astype(np.float32)
        P = (X**2).astype(np.float64)
        P_sum = float(np.sum(P) + eps)
        Pn = P / P_sum
        freqs = np.fft.rfftfreq(GPR_NFFT, d=1.0)  # 0~0.5

        sc = np.sum(freqs * X) / (np.sum(X) + eps)
        sbw = np.sqrt(np.sum((freqs - sc)**2 * X) / (np.sum(X) + eps))
        spec_ent = entropy(Pn + eps)
        pk_idx = int(np.argmax(X))
        pk_freq = freqs[pk_idx]
        pk_amp = X[pk_idx]

        geo_mean = np.exp(np.mean(np.log(X + eps)))
        ari_mean = np.mean(X) + eps
        spec_flat = geo_mean / ari_mean

        cdf = np.cumsum(Pn)
        med_freq = freqs[int(np.searchsorted(cdf, 0.50))]
        roll85  = freqs[int(np.searchsorted(cdf, 0.85))]
        roll95  = freqs[int(np.searchsorted(cdf, 0.95))]

        mu = np.sum(freqs * Pn)
        sig = np.sqrt(np.sum((freqs - mu)**2 * Pn) + eps)
        spec_skew = np.sum(((freqs - mu)/sig)**3 * Pn)
        spec_kurt = np.sum(((freqs - mu)/sig)**4 * Pn) - 3.0

        add("SpecCentroid", sc); add("SpecBW", sbw); add("SpecEntropy", spec_ent)
        add("PkFreq", pk_freq); add("PkAmp", pk_amp); add("SpecFlatness", spec_flat)
        add("MedFreq", med_freq); add("Rolloff85", roll85); add("Rolloff95", roll95)
        add("SpecSkew", spec_skew); add("SpecKurt", spec_kurt)

        bandE = []
        for i in range(len(GPR_BAND_EDGES) - 1):
            f1, f2 = GPR_BAND_EDGES[i], GPR_BAND_EDGES[i+1]
            m = (freqs >= f1) & (freqs < f2)
            e = float(np.sum(P[m]))
            bandE.append(e)
            add(f"BandE_{f1:.2f}-{f2:.2f}", e)

        total_e = float(np.sum(P) + eps)
        for i in range(len(GPR_BAND_EDGES) - 1):
            f1, f2 = GPR_BAND_EDGES[i], GPR_BAND_EDGES[i+1]
            add(f"BandR_{f1:.2f}-{f2:.2f}", bandE[i] / total_e)
    else:
        nband = len(GPR_BAND_EDGES) - 1
        for nm in ["SpecCentroid","SpecBW","SpecEntropy","PkFreq","PkAmp","SpecFlatness","MedFreq","Rolloff85","Rolloff95","SpecSkew","SpecKurt"]:
            add(nm, 0.0)
        for i in range(nband):
            f1, f2 = GPR_BAND_EDGES[i], GPR_BAND_EDGES[i+1]
            add(f"BandE_{f1:.2f}-{f2:.2f}", 0.0)
        for i in range(nband):
            f1, f2 = GPR_BAND_EDGES[i], GPR_BAND_EDGES[i+1]
            add(f"BandR_{f1:.2f}-{f2:.2f}", 0.0)

    # ---- 小波能量（5）----
    try:
        coeffs = pywt.wavedec(x.astype(np.float64), "db4", level=3)  # [cA3,cD3,cD2,cD1]
        wE = np.array([np.sum(c**2) for c in coeffs], dtype=np.float64)
        wE_sum = float(np.sum(wE) + eps)
        wEn = wE / wE_sum
        w_entropy = entropy(wEn + eps)

        add("W_E_A3", wE[0]); add("W_E_D3", wE[1]); add("W_E_D2", wE[2]); add("W_E_D1", wE[3])
        add("W_Entropy", w_entropy)
    except Exception:
        for nm in ["W_E_A3","W_E_D3","W_E_D2","W_E_D1","W_Entropy"]:
            add(nm, 0.0)

    feats = np.array(feats, dtype=np.float32)
    if return_names:
        return feats, np.array(names, dtype=object)
    return feats


def step0B_gpr_extract_features_to_stack():
    """
    方案B：
      - MAT(每列一个W，共64个文件) -> raw cube (Nt,H,W)
      - raw cube -> gpr_feat_stack (H,W,N_gpr)
    """
    print("\n=== Step0B: GPR raw(655×145×64) -> features -> (145×64×N_gpr) ===")

    out_stack = os.path.join(DIRS["GPR_STACK"], "gpr_feat_stack.npy")
    out_names = os.path.join(DIRS["GPR_STACK"], "gpr_feat_names.npy")
    out_raw   = os.path.join(DIRS["GPR_STACK"], "gpr_raw_cube.npy")

    gpr_raw = None

    # ---- 1) 若已有 gpr_feat_stack.npy，判断它是 feat 还是 raw ----
    if os.path.exists(out_stack):
        with runtime_block("Data reading", "Step0B inspect existing GPR stack"):
            arr = np.load(out_stack, allow_pickle=True)
        # 情况A：它其实是 raw cube
        if arr.ndim == 3 and arr.shape == (655, 145, 64):
            print(f"[GPR] Found out_stack but it's RAW cube {arr.shape} -> rebuild features.")
            gpr_raw = arr.astype(np.float32)
            if not os.path.exists(out_raw):
                np.save(out_raw, gpr_raw)
                print(f"[GPR] saved raw cube backup -> {out_raw}")
        # 情况B：它是 feature stack
        elif arr.ndim == 3 and arr.shape[0] == 145 and arr.shape[1] == 64 and os.path.exists(out_names):
            if not GPR_FORCE_REBUILD_FEATS:
                print(f"[GPR] Found existing feature stack: {arr.shape} -> {out_stack} (skip rebuild)")
                return
            else:
                print(f"[GPR] Found existing feature stack but FORCE rebuild is ON -> rebuild from MAT/raw.")
        else:
            print(f"[GPR] Found existing out_stack shape={arr.shape} but not recognized -> rebuild from MAT.")

    # ---- 2) 若没有 raw cube，则从 MAT 读取构建 raw cube ----
    if gpr_raw is None:
        with runtime_block("Data reading", "Step0B load GPR MAT -> raw cube"):
            files = sorted([f for f in os.listdir(DIRS["GPR_MAT"]) if f.endswith(".mat")], key=numeric_sort_key)
            if not files:
                raise FileNotFoundError(f"[GPR] No .mat files found in: {DIRS['GPR_MAT']}")

            raw_list = []
            for f in tqdm(files, desc="[GPR] Loading MAT"):
                a = load_mat_2d(os.path.join(DIRS["GPR_MAT"], f))  # (Nt,H)
                raw_list.append(a)

            gpr_raw = np.stack(raw_list, axis=2).astype(np.float32)  # (Nt,H,W)
            print(f"[GPR] raw cube shape: {gpr_raw.shape}")

            _ensure_dir(DIRS["GPR_STACK"])
            np.save(out_raw, gpr_raw)
            print(f"[GPR] saved raw cube -> {out_raw}")

    # ---- 3) 从 raw cube 提取特征 ----
    with runtime_block("Feature extraction", "Step0B extract GPR trace features"):
        Nt, H, W = gpr_raw.shape
        sample_feat, feat_names = gpr_trace_features(gpr_raw[:, 0, 0], return_names=True)
        N_gpr = int(sample_feat.size)

        gpr_feat = np.zeros((H, W, N_gpr), dtype=np.float32)
        for y in tqdm(range(H), desc="[GPR] Feature Extract (rows)"):
            for x in range(W):
                gpr_feat[y, x, :] = gpr_trace_features(gpr_raw[:, y, x])

        np.save(out_stack, gpr_feat.astype(np.float32))
        np.save(out_names, feat_names)
        print(f"[GPR] Feature stack saved: {gpr_feat.shape} -> {out_stack}, N_gpr={N_gpr}")
        assert feat_names.size == N_gpr, f"feat_names({feat_names.size}) != N_gpr({N_gpr})"


# ===========================================================
# 8) TransformerAE
# ===========================================================
class TransformerAE(nn.Module):
    def __init__(
        self,
        n_feat: int,
        embed_dim: int = 32,
        num_heads: int = 2,
        ff_dim: int = 128,
        bottleneck_dim: int = 16,
        num_layers: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_feat = n_feat
        self.embed_dim = embed_dim

        self.embed = nn.Linear(1, embed_dim)
        self.pos_emb = nn.Parameter(torch.zeros(1, n_feat, embed_dim))

        enc_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            # norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.to_latent = nn.Linear(embed_dim, bottleneck_dim)
        self.from_latent = nn.Linear(bottleneck_dim, n_feat * embed_dim)

        dec_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerEncoder(dec_layer, num_layers=num_layers)

        self.out_proj = nn.Linear(embed_dim, 1)

    def forward(self, x):
        B, N = x.shape
        x_tok = x.unsqueeze(-1)      # [B, N, 1]
        h = self.embed(x_tok)        # [B, N, D]
        h = h + self.pos_emb
        h_enc = self.encoder(h)      # [B, N, D]

        pooled = self.avg_pool(h_enc.transpose(1, 2)).squeeze(-1)  # [B, D]
        z = self.to_latent(pooled)   # [B, bottleneck]

        h_dec = self.from_latent(z).view(B, N, self.embed_dim)     # [B, N, D]
        h_dec = h_dec + self.pos_emb
        h_dec = self.decoder(h_dec)

        x_hat = self.out_proj(h_dec).squeeze(-1)  # [B, N]
        return x_hat

    def encode(self, x):
        # x: [B, N]
        B, N = x.shape
        x_tok = x.unsqueeze(-1)  # [B, N, 1]
        h = self.embed(x_tok)  # [B, N, D]
        h = h + self.pos_emb
        h_enc = self.encoder(h)  # [B, N, D]
        pooled = self.avg_pool(h_enc.transpose(1, 2)).squeeze(-1)  # [B, D]
        z = self.to_latent(pooled)  # [B, bottleneck]
        return z



class CNNAE1D(nn.Module):
    """
    1D CNN Autoencoder for feature-vector tokens.
    Input : x [B, N]  (N = #features)
    Treat as 1D signal with C=1 channel, L=N.
    Output: x_hat [B, N]
    """
    def __init__(
        self,
        n_feat: int,
        bottleneck_dim: int = 32,
        base_ch: int = 32,
        num_blocks: int = 3,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.n_feat = int(n_feat)
        self.bottleneck_dim = int(bottleneck_dim)

        # -------- Encoder --------
        enc_layers = []
        in_ch = 1
        ch = base_ch
        L = self.n_feat

        for _ in range(num_blocks):
            enc_layers += [
                nn.Conv1d(in_ch, ch, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm1d(ch),
                nn.GELU(),
            ]
            if dropout > 0:
                enc_layers.append(nn.Dropout(dropout))

            in_ch = ch
            ch = min(ch * 2, 256)
            L = (L + 1) // 2  # stride=2 length update (ceil)

        self.enc = nn.Sequential(*enc_layers)
        self.enc_out_ch = in_ch
        self.enc_out_len = L

        # ✅ 关键改动：全局池化到 [B, C]
        self.avg_pool = nn.AdaptiveAvgPool1d(1)

        # ✅ 关键改动：pooled([B,C]) -> z([B,bottleneck])
        self.to_latent = nn.Sequential(
            nn.Linear(self.enc_out_ch, self.bottleneck_dim),
            nn.GELU(),
        )

        # -------- Decoder --------
        # 仍然把 z 映射回 [B, C*L']，保证后续 ConvTranspose 能工作
        self.from_latent = nn.Sequential(
            nn.Linear(self.bottleneck_dim, self.enc_out_ch * self.enc_out_len),
            nn.GELU(),
        )

        dec_layers = []
        ch = self.enc_out_ch
        out_ch = max(ch // 2, 1)

        for _ in range(num_blocks):
            dec_layers += [
                nn.ConvTranspose1d(
                    ch, out_ch,
                    kernel_size=3, stride=2, padding=1,
                    output_padding=1,
                    bias=False
                ),
                nn.BatchNorm1d(out_ch),
                nn.GELU(),
            ]
            if dropout > 0:
                dec_layers.append(nn.Dropout(dropout))

            ch = out_ch
            out_ch = max(out_ch // 2, 1)

        self.dec = nn.Sequential(*dec_layers)
        self.out_proj = nn.Conv1d(ch, 1, kernel_size=1)

    def forward(self, x):
        # x: [B, N]
        B, N = x.shape
        assert N == self.n_feat, f"Input N={N} != n_feat={self.n_feat}"

        h = x.unsqueeze(1)      # [B, 1, N]
        h = self.enc(h)         # [B, C, L']

        # ✅ pooled → z（和 TAE 一致的写法）
        pooled = self.avg_pool(h).squeeze(-1)   # [B, C]
        z = self.to_latent(pooled)              # [B, bottleneck]

        h = self.from_latent(z)                 # [B, C*L']
        h = h.view(B, self.enc_out_ch, self.enc_out_len)
        h = self.dec(h)                         # [B, ?, ~N]
        h = self.out_proj(h)                    # [B, 1, ~N]
        x_hat = h.squeeze(1)                    # [B, ~N]

        # 长度对齐（保持你原来逻辑）
        if x_hat.shape[1] > N:
            x_hat = x_hat[:, :N]
        elif x_hat.shape[1] < N:
            pad = N - x_hat.shape[1]
            x_hat = torch.nn.functional.pad(x_hat, (0, pad), mode="replicate")

        return x_hat

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, N] -> z: [B, bottleneck_dim]
        """
        h = self.enc(x.unsqueeze(1))            # [B, C, L']
        pooled = self.avg_pool(h).squeeze(-1)   # [B, C]
        z = self.to_latent(pooled)              # [B, bottleneck]
        return z

class DNNAE(nn.Module):
    """
    MLP Autoencoder for feature vectors.
    Input : x [B, N]
    Output: x_hat [B, N]
    """
    def __init__(
        self,
        n_feat: int,
        bottleneck_dim: int = 32,
        h1: int = 256,
        h2: int = 128,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.n_feat = int(n_feat)

        # 为了对不同 N_feat 更稳：隐藏层不小于 n_feat*2 / n_feat*4
        h1 = int(max(h1, self.n_feat * 4))
        h2 = int(max(h2, self.n_feat * 2))

        act = nn.GELU()
        do = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()

        # Encoder
        self.enc = nn.Sequential(
            nn.Linear(self.n_feat, h1),
            nn.LayerNorm(h1),
            act,
            do,

            nn.Linear(h1, h2),
            nn.LayerNorm(h2),
            act,
            do,

            nn.Linear(h2, bottleneck_dim),
            act,
        )

        # Decoder
        self.dec = nn.Sequential(
            nn.Linear(bottleneck_dim, h2),
            nn.LayerNorm(h2),
            act,
            do,

            nn.Linear(h2, h1),
            nn.LayerNorm(h1),
            act,
            do,

            nn.Linear(h1, self.n_feat),
        )

    def forward(self, x):
        # x: [B, N]
        B, N = x.shape
        assert N == self.n_feat, f"Input N={N} != n_feat={self.n_feat}"
        z = self.enc(x)
        x_hat = self.dec(z)
        return x_hat

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, N] -> z: [B, bottleneck_dim]
        """
        return self.enc(x)


def build_ae_model(n_feat: int, backbone: str = None):
    bb = str(backbone if backbone is not None else AE_BACKBONE).lower()

    if bb == "cnn":
        return CNNAE1D(
            n_feat=n_feat,
            bottleneck_dim=AE_BOTTLENECK_DIM,
            base_ch=CNN_BASE_CH,
            num_blocks=CNN_NUM_BLOCKS,
            dropout=CNN_DROPOUT,
        )

    elif bb == "dnn":
        return DNNAE(
            n_feat=n_feat,
            bottleneck_dim=AE_BOTTLENECK_DIM,
            h1=DNN_H1,
            h2=DNN_H2,
            dropout=DNN_DROPOUT,
        )

    else:  # "tae"
        return TransformerAE(
            n_feat=n_feat,
            embed_dim=64,
            num_heads=4,
            ff_dim=128,
            bottleneck_dim=AE_BOTTLENECK_DIM,
            num_layers=1
        )


def _prepare_ae_inputs(feature_stack, ae_out_dir, tag, stat_prefix, modality_label):
    H, W, n_feat = feature_stack.shape
    X_all = feature_stack.reshape(-1, n_feat).astype(np.float32)

    healthy_mask = get_unified_healthy_mask((H, W), ae_out_dir=ae_out_dir, tag=tag)
    X_train = X_all[healthy_mask.reshape(-1)]
    print(f"[{modality_label} AE] train samples = {X_train.shape[0]}")

    mean = X_train.mean(axis=0, keepdims=True)
    std = X_train.std(axis=0, keepdims=True) + 1e-6

    X_train_norm = (X_train - mean) / std
    X_all_norm = (X_all - mean) / std

    np.save(os.path.join(ae_out_dir, f"{stat_prefix}_mean.npy"), mean)
    np.save(os.path.join(ae_out_dir, f"{stat_prefix}_std.npy"), std)

    return X_all_norm, X_train_norm, healthy_mask


def _train_ae_model(X_train_norm, n_feat, ae_out_dir, modality_label, weight_name):
    loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train_norm)),
        batch_size=AE_BATCH_SIZE,
        shuffle=True,
        drop_last=False
    )

    model = build_ae_model(n_feat).to(DEVICE)

    if PRINT_MODEL_PARAMS:
        if PRINT_ALL_BACKBONES:
            for bb in ["cnn", "dnn", "tae"]:
                m_tmp = build_ae_model(n_feat, bb)
                log_model_params_once(ae_out_dir, modality_label, bb, n_feat, m_tmp)
                del m_tmp
        else:
            log_model_params_once(
                ae_out_dir,
                modality_label,
                str(AE_BACKBONE).lower(),
                n_feat,
                model
            )

    optimizer = torch.optim.Adam(model.parameters(), lr=AE_LR, weight_decay=AE_WEIGHT_DECAY)
    criterion = nn.MSELoss()

    print(f"[{modality_label} AE] training...")
    train_loss_hist = []

    for ep in range(AE_EPOCHS):
        model.train()
        run_loss = 0.0

        for (bx,) in loader:
            bx = bx.to(DEVICE)
            optimizer.zero_grad()
            recon = model(bx)
            loss = criterion(recon, bx)
            loss.backward()
            optimizer.step()
            run_loss += loss.item() * bx.size(0)

        epoch_loss = run_loss / len(X_train_norm)
        train_loss_hist.append(epoch_loss)

        if ep == 0 or (ep + 1) % 10 == 0:
            print(f"  Epoch {ep + 1}/{AE_EPOCHS}, Loss={epoch_loss:.6e}")

    torch.save(model.state_dict(), os.path.join(ae_out_dir, weight_name))
    return model, train_loss_hist

# ===========================================================
# 9) Step5A: Knock AE -> prob_knock.npy
# ===========================================================
def step5A_AE_knock():
    print("\n=== Step5A: Knock AE (Top-60, keep ZCR) ===")

    ae_out_dir = _get_ae_out_dir()

    stack_path = os.path.join(DIRS["STACK"], "feat_stack.npy")
    names_path = os.path.join(DIRS["STACK"], "feat_names.npy")
    if not (os.path.exists(stack_path) and os.path.exists(names_path)):
        raise FileNotFoundError("[Knock] Missing feat_stack.npy / feat_names.npy")

    with runtime_block("Data reading", "Step5A load knock feature stack"):
        feat_stack = np.load(stack_path).astype(np.float32)      # (H,W,N_all)
        feat_names = np.load(names_path, allow_pickle=True)      # (N_all,)

    H, W, _ = feat_stack.shape
    print(f"[Knock] raw feature stack: {H}×{W}×{feat_stack.shape[2]}")

    # =========================================================
    # Top-60 Knock feature selection (11 per channel × 4)
    # =========================================================
    with runtime_block("Preprocessing", "Step5A select + normalize knock features"):
        idx_sel = []
        for i, name in enumerate(feat_names):
            name = str(name)
            for ch in USE_CHANNELS:
                if name.startswith(f"Ch{ch}_"):
                    for suf in KNOCK_AE_FEATURE_SUFFIXES:
                        if name.endswith("_" + suf):
                            idx_sel.append(i)
                            break

        idx_sel = np.array(sorted(set(idx_sel)), dtype=int)

        feat_names_sel = feat_names[idx_sel]
        feat_stack_sel = feat_stack[:, :, idx_sel]

        N_sel = feat_stack_sel.shape[2]
        print(f"[Knock] Top-60 selected features: N_sel = {N_sel}")

        X_all_norm, X_train_norm, healthy_mask = _prepare_ae_inputs(
            feat_stack_sel,
            ae_out_dir=ae_out_dir,
            tag="knock",
            stat_prefix="knock",
            modality_label="Knock"
        )

    with runtime_block("TAE training/inference", "Step5A train knock AE"):
        model, train_loss_hist = _train_ae_model(
            X_train_norm,
            n_feat=N_sel,
            ae_out_dir=ae_out_dir,
            modality_label="Knock",
            weight_name="knock_ae_weights.pth"
        )

    # ✅ 导出训练曲线 + 更新统一CSV
    export_ae_train_logs(ae_out_dir, modality_key="knock", loss_hist=train_loss_hist)

    # =========================
    # Train-only latent z + t-SNE (Knock)
    # =========================
    if TAE_EXPORT_LATENT and hasattr(model, "encode"):
        out_lat = _exp_dir("AE_Latent", f"Knock_{str(AE_BACKBONE).lower()}")  # ✅按骨干区分
        Xtr = X_train_norm

        model.eval()
        with torch.no_grad():
            z_tr = model.encode(torch.from_numpy(Xtr).to(DEVICE)).cpu().numpy()
            recon_tr = model(torch.from_numpy(Xtr).to(DEVICE)).cpu().numpy()

        score_tr = np.mean(np.abs(Xtr - recon_tr), axis=1)

        # ✅不抽样：你训练集就 116 点，直接全画，保证“tsne点数=训练点数”
        xy, _ = tsne_fit_transform_safe(
            z_tr,
            seed=BASE_SEED,
            perplexity=TAE_TSNE_PERPLEXITY,
            init="pca",
            iters=2000
        )
        if xy is not None:
            plt.figure(figsize=(7, 6))
            plt.scatter(xy[:, 0], xy[:, 1], s=18, c="tab:blue", alpha=0.85)
            plt.title(f"Train-only t-SNE (Knock, {str(AE_BACKBONE).lower()})  n={z_tr.shape[0]}")
            plt.xlabel("t-SNE-1");
            plt.ylabel("t-SNE-2")
            plt.tight_layout()
            plt.savefig(os.path.join(out_lat, "Knock_trainOnly_tSNE.png"), dpi=200)
            plt.close()

        # 统计：训练集中 score 的上 3%（可作为“混入少量异常”的候选）
        thr97 = np.quantile(score_tr, 0.97)
        n_out = int((score_tr > thr97).sum())
        with open(os.path.join(out_lat, "Knock_trainOnly_summary.txt"), "w", encoding="utf-8") as f:
            f.write(f"n_train={len(score_tr)}\n")
            f.write(f"thr_q97={thr97}\n")
            f.write(f"n_score_above_q97={n_out}\n")
        print(f"[Latent] Train-only t-SNE saved -> {out_lat}")

    # =========================================================
    # Reconstruction error → probability
    # =========================================================
    with runtime_block("TAE training/inference", "Step5A infer knock AE"):
        model.eval()
        with torch.no_grad():
            recon_all = model(torch.from_numpy(X_all_norm).to(DEVICE)).cpu().numpy()

        err  = np.abs(X_all_norm - recon_all)
    # ===== TAE latent t-SNE =====
    # ===== TAE latent t-SNE + 训练集异常点位置导出 =====
    if (str(AE_BACKBONE).lower() == "tae") and TAE_EXPORT_LATENT and hasattr(model, "encode"):
        # --- 1) 全体点的瓶颈向量 + score ---
        with torch.no_grad():
            z_all = model.encode(torch.from_numpy(X_all_norm).to(DEVICE)).cpu().numpy()

        # --- 2) 训练集点对应的 z_train / score_train（用于定位训练集内异常）---
        train_mask_1d = healthy_mask.reshape(-1).astype(bool)
        z_train = z_all[train_mask_1d]

        out_lat = _exp_dir("TAE_Latent", "Knock")


        # 训练点在原始网格上的(row,col)——如果你的训练掩膜是 healthy_mask (H,W)
        flat_idx = np.where(healthy_mask.reshape(-1))[0]
        rc_tr = np.column_stack(np.unravel_index(flat_idx, (H, W)))  # (N_train, 2)

        export_train_tsne_dbscan(
            z_train=z_train,  # ✅ 用当前 block 里真正的 z_train
            out_dir=out_lat,
            prefix="Knock",
            rc_train=rc_tr,
            seed=BASE_SEED,  # ✅ 统一随机种子
        )

    with runtime_block("TAE training/inference", "Step5A export knock AE outputs"):
        prob = _percentile_prob(err, 5, 95)

        prob_knock = prob.reshape(H, W, N_sel).astype(np.float32)

        np.save(os.path.join(ae_out_dir, "prob_knock.npy"), prob_knock)
        np.save(os.path.join(ae_out_dir, "knock_feat_names.npy"), feat_names_sel)

    if TAE_EXPORT_LATENT and hasattr(model, "encode"):
        out_lat = _exp_dir("AE_Latent", f"Knock_{str(AE_BACKBONE).lower()}")  # ✅按骨干分目录
        Xtr = X_train_norm

        model.eval()
        with torch.no_grad():
            z_tr = model.encode(torch.from_numpy(Xtr).to(DEVICE)).cpu().numpy()

        # 训练点在原始网格(row,col)
        flat_idx = np.where(healthy_mask.reshape(-1))[0]
        rc_tr = np.column_stack(np.unravel_index(flat_idx, (H, W)))  # (N_train, 2)

        # ✅直接用你已有的函数：会输出 Knock_train_tSNE_DBSCAN.png
        export_train_tsne_dbscan(
            z_train=z_tr,
            out_dir=out_lat,
            prefix="Knock",
            rc_train=rc_tr,
            seed=BASE_SEED,
        )

    print(f"[Knock AE] done -> prob_knock.npy, N_feat={N_sel}")

    export_knock_sensor_ae_maps(
        prob_knock,
        feat_names_sel,
        out_dir=_exp_dir("Knock_AE_PerSensor")
    )


# ===========================================================
# 10) Step5B: GPR AE -> prob_gpr.npy
# ===========================================================
def step5B_AE_gpr():
    print("\n=== Step5B: GPR AE (Top-40 features) ===")

    ae_out_dir = _get_ae_out_dir()

    gpr_stack_path = os.path.join(DIRS["GPR_STACK"], "gpr_feat_stack.npy")
    gpr_names_path = os.path.join(DIRS["GPR_STACK"], "gpr_feat_names.npy")
    if not (os.path.exists(gpr_stack_path) and os.path.exists(gpr_names_path)):
        raise FileNotFoundError("[GPR] Missing gpr_feat_stack.npy / gpr_feat_names.npy")

    with runtime_block("Data reading", "Step5B load GPR feature stack"):
        gpr_stack = np.load(gpr_stack_path).astype(np.float32)     # (H,W,N_all)
        gpr_names = np.load(gpr_names_path, allow_pickle=True)

    H, W, _ = gpr_stack.shape
    print(f"[GPR] raw feature stack: {H}×{W}×{gpr_stack.shape[2]}")

    # =========================================================
    # Top-40 GPR feature selection (12)
    # =========================================================
    with runtime_block("Preprocessing", "Step5B select + normalize GPR features"):
        idx_sel = [i for i, n in enumerate(gpr_names) if str(n) in GPR_AE_FEATURES]
        idx_sel = np.array(idx_sel, dtype=int)

        gpr_stack = gpr_stack[:, :, idx_sel]
        gpr_names = gpr_names[idx_sel]

        N_gpr = gpr_stack.shape[2]
        print(f"[GPR] Top-40 selected features: N_gpr = {N_gpr}")

        X_all_norm, X_train_norm, healthy_mask = _prepare_ae_inputs(
            gpr_stack,
            ae_out_dir=ae_out_dir,
            tag="gpr",
            stat_prefix="gpr",
            modality_label="GPR"
        )
    with runtime_block("TAE training/inference", "Step5B train GPR AE"):
        model, train_loss_hist = _train_ae_model(
            X_train_norm,
            n_feat=N_gpr,
            ae_out_dir=ae_out_dir,
            modality_label="GPR",
            weight_name="gpr_ae_weights.pth"
        )
    if TAE_EXPORT_LATENT and hasattr(model, "encode"):
        out_lat = _exp_dir("AE_Latent", f"GPR_{str(AE_BACKBONE).lower()}")  # ✅改对
        Xtr = X_train_norm  # 训练集（healthy_mask 选出来的点）

        model.eval()
        with torch.no_grad():
            z_tr = model.encode(torch.from_numpy(Xtr).to(DEVICE)).cpu().numpy()
            recon_tr = model(torch.from_numpy(Xtr).to(DEVICE)).cpu().numpy()

        score_tr = np.mean(np.abs(Xtr - recon_tr), axis=1)  # 重构误差作为异常分

        # 保存原始数据
        np.save(os.path.join(out_lat, "z_trainOnly.npy"), z_tr.astype(np.float32))
        np.save(os.path.join(out_lat, "score_trainOnly.npy"), score_tr.astype(np.float32))

        # ✅输出：t-SNE + DBSCAN（你已写好 export_train_tsne_dbscan）
        export_train_tsne_dbscan(
            z_train=z_tr,
            out_dir=out_lat,
            prefix=f"Knock_{str(AE_BACKBONE).lower()}",
            rc_train=None,  # ✅你刚说不要括号坐标：直接传 None
            seed=BASE_SEED,
        )

    # ✅ 导出训练曲线 + 更新统一CSV
    export_ae_train_logs(ae_out_dir, modality_key="gpr", loss_hist=train_loss_hist)

    # =========================================================
    # Reconstruction error → probability
    # =========================================================
    with runtime_block("TAE training/inference", "Step5B infer GPR AE"):
        model.eval()
        with torch.no_grad():
            recon_all = model(torch.from_numpy(X_all_norm).to(DEVICE)).cpu().numpy()

        err  = np.abs(X_all_norm - recon_all)

    # ===== TAE latent t-SNE =====
    if (str(AE_BACKBONE).lower() == "tae") and TAE_EXPORT_LATENT and hasattr(model, "encode"):
        with torch.no_grad():
            z_all = model.encode(torch.from_numpy(X_all_norm).to(DEVICE)).cpu().numpy()
        score_all = err.mean(axis=1)
        out_lat = _exp_dir("TAE_Latent", "GPR")
        export_latent_tsne(
            z_all=z_all,
            score_all=score_all,
            train_mask_1d=healthy_mask.reshape(-1).astype(bool),
            out_dir=out_lat,
            tag="GPR"
        )

    with runtime_block("TAE training/inference", "Step5B export GPR AE outputs"):
        prob = _percentile_prob(err, 5, 95)

        prob_gpr = prob.reshape(H, W, N_gpr).astype(np.float32)

    with runtime_block("TAE training/inference", "Step5B export GPR AE files"):

    # ===== 调用 =====
        export_gpr_ae_maps(
            prob_gpr,
            out_dir=_exp_dir("GPR_AE")
        )

        np.save(os.path.join(ae_out_dir, "prob_gpr.npy"), prob_gpr)
        np.save(os.path.join(ae_out_dir, "gpr_feat_names.npy"), gpr_names)

        if str(AE_BACKBONE).lower() == "tae":
            out_fig = _exp_dir("TAE_ProbMaps", "GPR")
            export_tae_prob_maps_smallset(
                prob3d=prob_gpr,
                feat_names=gpr_names,
                healthy_mask=healthy_mask,
                out_dir=out_fig,
                prefix="GPR",
                force_features=TAE_GPR_FIG_FEATURES,
                topk=TAE_EXPORT_TOPK,
            )

    print(f"[GPR AE] done -> prob_gpr.npy, N_feat={N_gpr}")

# ===========================================================
# 11) Step6: Bayes 融合（模态内 + 跨模态）
# ===========================================================

def collect_alpha_beta(
    records: list,
    modality: str,
    feature_names,
    alpha,
    beta
):
    for i, fname in enumerate(feature_names):
        records.append({
            "Modality": modality,
            "Feature": str(fname),
            "Alpha": float(alpha[i]),
            "Beta":  float(beta[i]),
            "Reliability": float(np.log((alpha[i] + 1e-6) / (beta[i] + 1e-6)))
        })

def export_alpha_beta_csv(records, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    df = pd.DataFrame(records)
    csv_path = os.path.join(out_dir, "Alpha_Beta_GPR_and_KnockAll.csv")
    df.to_csv(csv_path, index=False)
    print(f"[✓] Alpha/Beta CSV saved -> {csv_path}")
    return df


def plot_alpha_beta_bar(df, modality, out_dir, top_k=20):
    sub = df[df["Modality"] == modality].copy()
    if sub.empty:
        return

    sub = sub.sort_values("Reliability", ascending=False).head(top_k)
    x = np.arange(len(sub))
    width = 0.35

    plt.figure(figsize=(max(10, len(sub) * 0.4), 4))
    plt.bar(x - width / 2, sub["Alpha"], width, label="Alpha (TPR)")
    plt.bar(x + width / 2, sub["Beta"], width, label="Beta (FPR)")

    plt.xticks(x, sub["Feature"], rotation=45, ha="right", fontsize=8)
    plt.ylabel("Probability")
    plt.title(f"Alpha / Beta per Feature – {modality}")
    plt.legend()
    plt.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    out_png = os.path.join(out_dir, f"AlphaBeta_{modality}.png")
    plt.savefig(out_png, dpi=150)
    plt.close()

    print(f"[✓] Bar plot saved -> {out_png}")


def rel_weight(alpha, beta, eps=1e-6):
    alpha_m = float(np.clip(np.mean(alpha), eps, 1.0 - eps))
    beta_m = float(np.clip(np.mean(beta), eps, 1.0 - eps))
    return np.log(alpha_m / (beta_m + eps))


def logit_fuse(p1, p2, w1=1.0, w2=1.0, eps=1e-6):
    p1 = np.clip(p1, eps, 1 - eps)
    p2 = np.clip(p2, eps, 1 - eps)
    l1 = np.log(p1 / (1 - p1))
    l2 = np.log(p2 / (1 - p2))
    lf = w1 * l1 + w2 * l2
    return 1.0 / (1.0 + np.exp(-lf))


def average_fuse_threshold(p1, p2, thresh=0.5):
    post = 0.5 * (np.asarray(p1, dtype=np.float32) + np.asarray(p2, dtype=np.float32))
    post = np.clip(post, 0.0, 1.0)
    mask = (post > thresh).astype(float)
    return post, mask


def mean_prob_maps_threshold(prob_maps, thresh=0.5):
    prob_maps = np.asarray(prob_maps, dtype=np.float32)
    if prob_maps.ndim != 3:
        raise ValueError(f"prob_maps must be 3D (H, W, C), got {prob_maps.shape}")
    post = np.mean(prob_maps, axis=2)
    post = np.clip(post, 0.0, 1.0)
    mask = (post > thresh).astype(float)
    return post, mask

def compute_object_level_metrics(pred_mask, gt_mask, iou_thresh=0.5):
    """
    Object-level metrics (fragment-tolerant):
    - A GT object is considered detected if the UNION of all overlapping
      predicted fragments reaches IoU >= iou_thresh.
    """

    gt_lab, gt_num = label(gt_mask)
    pr_lab, pr_num = label(pred_mask)

    def iou(a, b):
        inter = np.logical_and(a, b).sum()
        union = np.logical_or(a, b).sum()
        return inter / (union + 1e-12)

    matched_gt = 0
    matched_pred = set()

    for gi in range(1, gt_num + 1):
        g_obj = (gt_lab == gi)

        # ① 找到所有与该 GT 有重叠的预测碎块
        overlapping_preds = []
        for pi in range(1, pr_num + 1):
            p_obj = (pr_lab == pi)
            if np.logical_and(g_obj, p_obj).any():
                overlapping_preds.append((pi, p_obj))

        if not overlapping_preds:
            continue

        # ② 合并这些预测碎块
        p_union = np.zeros_like(g_obj, dtype=bool)
        for pi, p_obj in overlapping_preds:
            p_union |= p_obj

        # ③ 用“合并后的预测区域”计算 IoU
        if iou(g_obj, p_union) >= iou_thresh:
            matched_gt += 1
            for pi, _ in overlapping_preds:
                matched_pred.add(pi)

    object_recall = matched_gt / (gt_num + 1e-12)
    false_positive_objects = pr_num - len(matched_pred)

    return {
        "Object_Recall": object_recall,
        "GT_Objects": gt_num,
        "Pred_Objects": pr_num,
        "FP_Objects": false_positive_objects,
        "Matched_Objects": matched_gt
    }

def evaluate_all_metrics(mask, GT, prefix=""):
    """
    mask, GT: 2D binary (0/1)
    prefix: modality tag, e.g. 'GPR', 'Knock', 'Final'
    """
    # ---- pixel-level ----
    pix_m = evaluate_metrics(mask, GT)

    # ---- object-level ----
    obj_m = compute_object_level_metrics(
        pred_mask=mask.astype(int),
        gt_mask=GT.astype(int),
        iou_thresh=0.2
    )

    # ---- relaxed IoU ----
    relax_m = {
        "Relaxed_IoU_r1": relaxed_iou(mask, GT, radius=1),
        "Relaxed_IoU_r2": relaxed_iou(mask, GT, radius=2),
        "Relaxed_IoU_r3": relaxed_iou(mask, GT, radius=3),
    }

    # ---- merge ----
    row = {
        "Modality": prefix,

        "IoU": pix_m["IoU"],
        "F1": pix_m["F1"],
        "Precision": pix_m["Precision"],
        "Recall": pix_m["Recall"],
        "TP": pix_m["TP"],
        "FP": pix_m["FP"],
        "FN": pix_m["FN"],
    }

    row.update(obj_m)
    row.update(relax_m)

    return row

def export_knock_sensor_ae_maps(
    prob_knock: np.ndarray,
    feat_names_sel,
    out_dir: str,
    channels=(0, 1, 2, 3)
):
    """
    prob_knock: (29,64,N_sel)
    """
    _ensure_dir(out_dir)

    for ch in channels:
        idx = [
            i for i, n in enumerate(feat_names_sel)
            if str(n).startswith(f"Ch{ch}_")
        ]
        if not idx:
            print(f"[EXPORT] Ch{ch}: no features, skip")
            continue

        ch_prob = np.mean(prob_knock[:, :, idx], axis=2)  # (29,64)

        export_map_2d(
            ch_prob,
            out_dir,
            base_name=f"AE_Prob_Ch{ch}",
            title=f"Knock AE Probability – Ch{ch}"
        )


# def compute_object_level_metrics(pred_mask, gt_mask, iou_thresh=0.02):
#     """
#     pred_mask, gt_mask: 2D binary (0/1)
#     """
#     # ---- 连通域 ----
#     gt_lab, gt_num = label(gt_mask)
#     pr_lab, pr_num = label(pred_mask)
#
#     def iou(a, b):
#         inter = np.logical_and(a, b).sum()
#         union = np.logical_or(a, b).sum()
#         return inter / (union + 1e-12)
#
#     matched_gt = 0
#     matched_pred = set()
#
#     for gi in range(1, gt_num + 1):
#         g_obj = (gt_lab == gi)
#         for pi in range(1, pr_num + 1):
#             if pi in matched_pred:
#                 continue
#             p_obj = (pr_lab == pi)
#             if iou(g_obj, p_obj) >= iou_thresh:
#                 matched_gt += 1
#                 matched_pred.add(pi)
#                 break
#
#     object_recall = matched_gt / (gt_num + 1e-12)
#     false_positive_objects = pr_num - len(matched_pred)
#
#     return {
#         "Object_Recall": object_recall,
#         "GT_Objects": gt_num,
#         "Pred_Objects": pr_num,
#         "FP_Objects": false_positive_objects,
#         "Matched_Objects": matched_gt
#     }

def relaxed_iou(pred_mask, gt_mask, radius=2):
    """
    radius: pixel tolerance
    """
    struct = np.ones((2 * radius + 1, 2 * radius + 1), dtype=bool)
    gt_dilated = binary_dilation(gt_mask, structure=struct)

    inter = np.logical_and(pred_mask, gt_dilated).sum()
    union = np.logical_or(pred_mask, gt_dilated).sum()

    return inter / (union + 1e-12)

def load_and_check_gt(gt_path, detected_shape):
    """
    支持：
      - .xlsx / .xls  (29×64)
      - .csv
      - .mat
    detected_shape: (H, W) of current detection result
    """
    if gt_path is None:
        print("[INFO] 未配置 GT_PATH，跳过评估。")
        return None

    if not os.path.exists(gt_path):
        print(f"[INFO] 未检测到 GT 文件: {gt_path} (跳过评估)")
        return None

    ext = os.path.splitext(gt_path)[1].lower()

    try:
        # ===== Excel / CSV =====
        if ext in [".xlsx", ".xls", ".csv"]:
            if ext == ".csv":
                gt = pd.read_csv(gt_path, header=None).values
            else:
                gt = pd.read_excel(gt_path, header=None).values

            gt = (gt > 0).astype(int)

        # ===== MAT =====
        elif ext == ".mat":
            mat_data = sio.loadmat(gt_path)
            key = [k for k in mat_data.keys() if not k.startswith("__")][0]
            gt = (mat_data[key] > 0).astype(int)

        else:
            raise ValueError(f"Unsupported GT format: {ext}")

        gt_h, gt_w = gt.shape
        det_h, det_w = detected_shape

        print(f"[GT] Loaded GT: {gt_h}×{gt_w}, Detected: {det_h}×{det_w}")

        # ===== 自动尺度匹配 =====
        if (gt_h, gt_w) != (det_h, det_w):
            print("[GT] Resizing GT to match detection grid...")
            gt = zoom(gt, (det_h / gt_h, det_w / gt_w), order=0)

        return gt.astype(int)

    except Exception as e:
        print(f"[GT Error] 加载失败: {e}")
        return None


def evaluate_metrics(mask, gt):
    if gt is None:
        return None
    tn, fp, fn, tp = confusion_matrix(gt.flatten(), mask.flatten(), labels=[0, 1]).ravel()
    precision = tp / (tp + fp + 1e-12)
    recall = tp / (tp + fn + 1e-12)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-12)
    iou = tp / (tp + fp + fn + 1e-12)
    return {"IoU": iou, "F1": f1, "Precision": precision, "Recall": recall, "TP": tp, "FP": fp, "FN": fn}

def bayesPixelFusion3_optimized(prob_all, thresh=0.5, maxIter=50):
    H, W, N = prob_all.shape
    N_pix = H * W

    Y = prob_all.reshape(N_pix, N).T
    Y = np.clip(Y, 1e-6, 1 - 1e-6)

    Post = np.mean(Y, axis=0)
    alpha_final, beta_final = None, None
    loss_hist = []

    for _ in range(maxIter):
        w1 = Post
        w0 = 1 - Post
        s_w1 = np.sum(w1) + 2
        s_w0 = np.sum(w0) + 2

        alpha = (Y @ w1 + 1) / s_w1
        beta  = (Y @ w0 + 1) / s_w0
        alpha = np.clip(alpha, 1e-3, 1 - 1e-3)
        beta  = np.clip(beta,  1e-3, 1 - 1e-3)

        alp_col = alpha[:, None]
        bet_col = beta[:, None]

        term1 = Y * alp_col + (1 - Y) * (1 - alp_col)
        term0 = Y * bet_col + (1 - Y) * (1 - bet_col)

        logL1 = np.sum(np.log(term1 + 1e-12), axis=0)
        logL0 = np.sum(np.log(term0 + 1e-12), axis=0)

        pi0 = (np.sum(Post) + 1) / (N_pix + 2)
        logit_prior = np.log(pi0 / (1 - pi0))

        delta = logL1 - logL0 + logit_prior
        Post_new = 1.0 / (1.0 + np.exp(-delta))

        diff = np.mean(np.abs(Post_new - Post))
        loss_hist.append(diff)

        Post = 0.7 * Post_new + 0.3 * Post
        alpha_final, beta_final = alpha, beta
        if diff < 1e-5:
            break

    if np.mean(alpha_final) < np.mean(beta_final):
        Post = 1 - Post
        alpha_final, beta_final = 1 - beta_final, 1 - alpha_final

    post_img = Post.reshape(H, W)
    mask_img = (post_img > thresh).astype(float)
    return post_img, mask_img, alpha_final, beta_final, loss_hist

def _save_compare_three_separate(out_dir, tag, post, mask, GT):
    """
    直接保存三张独立图：
      - Pred
      - GT
      - TP_FP_FN
    不裁剪、不依赖拼图
    """
    _ensure_dir(out_dir)

    post_v = _repeat_columns(post)
    mask_v = _repeat_columns(mask)
    gt_v = _repeat_columns(GT)

    # =======================
    # 1) Prediction
    # =======================
    plt.figure(figsize=(5, 4))
    plt.imshow(post_v, cmap="cividis", vmin=0, vmax=1)
    plt.colorbar()
    plt.title(f"Pred – {tag}")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{tag}_Pred.png"), dpi=150)
    plt.close()

    # =======================
    # 2) Ground Truth
    # =======================
    plt.figure(figsize=(5, 4))
    plt.imshow(gt_v, cmap="cividis", vmin=0, vmax=1)
    plt.colorbar()
    plt.title("Ground Truth")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{tag}_GT.png"), dpi=150)
    plt.close()

    # =======================
    # 3) TP / FP / FN
    # =======================
    diff = np.zeros((GT.shape[0], GT.shape[1], 3), dtype=np.float32)
    diff[..., 1] = (mask == 1) & (GT == 1)  # TP (Green)
    diff[..., 0] = (mask == 1) & (GT == 0)  # FP (Red)
    diff[..., 2] = (mask == 0) & (GT == 1)  # FN (Blue)

    diff_v = _repeat_columns(diff)

    plt.figure(figsize=(5, 4))
    plt.imshow(diff_v)
    plt.title("G=TP, R=FP, B=FN")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{tag}_TP_FP_FN.png"), dpi=150)
    plt.close()


def _save_bayes_pack(out_dir, tag, post, mask, alpha, beta, loss, GT=None):
    _ensure_dir(out_dir)

    np.savetxt(os.path.join(out_dir, f"{tag}_Post.csv"), post)
    np.savetxt(os.path.join(out_dir, f"{tag}_Mask.csv"), mask)
    np.savetxt(os.path.join(out_dir, f"{tag}_alpha.csv"), alpha)
    np.savetxt(os.path.join(out_dir, f"{tag}_beta.csv"), beta)
    np.savetxt(os.path.join(out_dir, f"{tag}_loss.csv"), loss)

    plt.figure(figsize=(5, 4))
    post_vis = _repeat_columns(post)

    plt.imshow(post_vis, cmap="cividis", vmin=0, vmax=1)
    plt.colorbar()
    plt.title(f"Post - {tag}")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{tag}_Post.png"), dpi=150)
    plt.close()

    if GT is not None:
        m = evaluate_metrics(mask, GT)
        with open(os.path.join(out_dir, f"{tag}_metrics.txt"), "w", encoding="utf-8") as f:
            f.write(str(m) + "\n")

        fig, ax = plt.subplots(1, 3, figsize=(12, 4))
        mask_vis = _repeat_columns(mask)
        GT_vis = _repeat_columns(GT)

        diff = np.zeros((GT.shape[0], GT.shape[1], 3))
        diff[..., 1] = (mask == 1) & (GT == 1)  # TP
        diff[..., 0] = (mask == 1) & (GT == 0)  # FP
        diff[..., 2] = (mask == 0) & (GT == 1)  # FN
        diff_vis = _repeat_columns(diff)

        # ================== 绘图 ==================
        ax[0].imshow(mask_vis, cmap="cividis", vmin=0, vmax=1)
        ax[0].set_title(f"Pred\n(IoU={m['IoU']:.3f}, F1={m['F1']:.3f})", fontsize=9)

        ax[1].imshow(GT_vis, cmap="cividis", vmin=0, vmax=1)
        ax[1].set_title("Ground Truth", fontsize=9)

        ax[2].imshow(diff_vis)
        ax[2].set_title("G=TP, R=FP, B=FN", fontsize=9)

        for a in ax:
            a.axis("off")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{tag}_Compare_GT.png"), dpi=150)
        plt.close()
        _save_compare_three_separate(
            out_dir=out_dir,
            tag=tag,
            post=post,
            mask=mask,
            GT=GT
        )
    else:
        plt.figure(figsize=(5, 4))
        plt.imshow(_repeat_columns(mask), cmap="cividis", vmin=0, vmax=1)
        plt.title(f"Mask - {tag}")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{tag}_Mask.png"), dpi=150)
        plt.close()

    plt.figure(figsize=(5, 3))
    plt.plot(loss)
    plt.title(f"Convergence - {tag}")
    plt.xlabel("Iteration")
    plt.ylabel("Diff")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{tag}_loss.png"), dpi=120)
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.plot(alpha, "o-", label="Alpha (TPR)", markersize=3)
    plt.plot(beta,  "s-", label="Beta (FPR)", markersize=3)
    plt.title(f"Reliability - {tag}")
    plt.xlabel("Index")
    plt.ylabel("Prob")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{tag}_reliability.png"), dpi=150)
    plt.close()

def _get_bayes_root():
    """
    让 Bayes 输出按 EXP_TAG 分目录，避免 k=1..10 的结果互相覆盖
    """
    if EXP_TAG is None:
        d = DIRS["BAYES"]
    else:
        d = os.path.join(DIRS["BAYES"], f"exp_{EXP_TAG}")
    os.makedirs(d, exist_ok=True)
    return d


def step6_bayes_fusion_multimodal():
    w_k = 1.0
    w_g = 1.0

    BAYES_ROOT = _get_bayes_root()   # ✅ 新增：本次实验(k)的输出根目录

    print("\n=== Step6: Bayes Fusion (Knock + GPR + Per-Sensor) ===")

    ae_in_dir = _get_ae_out_dir()

    # =========================================================
    # Load probabilities
    # =========================================================
    prob_gpr_path   = os.path.join(ae_in_dir, "prob_gpr.npy")
    prob_knock_path = os.path.join(ae_in_dir, "prob_knock.npy")

    if not os.path.exists(prob_gpr_path):
        raise FileNotFoundError("[Step6] prob_gpr.npy not found. Run Step5B first.")

    prob_gpr = np.load(prob_gpr_path).astype(np.float32)
    Hg, Wg, _ = prob_gpr.shape
    print(f"[Step6] GPR prob shape = {Hg}×{Wg}")

    prob_knock = None
    if os.path.exists(prob_knock_path):
        prob_knock = np.load(prob_knock_path).astype(np.float32)
        Hk, Wk, _ = prob_knock.shape
        print(f"[Step6] Knock prob shape = {Hk}×{Wk}")
    else:
        print("[Step6] prob_knock.npy not found → only GPR intra-modal fusion.")

    # =========================================================
    # Load GT
    # =========================================================
    GT = load_and_check_gt(GT_PATH, (Hg, Wg))

    metrics_rows = []

    # =========================================================
    # ① GPR intra-modal Bayes
    # =========================================================
    print("[Bayes] Intra-modal: GPR")
    post_gpr, mask_gpr, a_g, b_g, loss_g = bayesPixelFusion3_optimized(prob_gpr)
    out_gpr = os.path.join(BAYES_ROOT, "IntraModal_GPR")

    save_bayes_iter_loss(loss_g, out_gpr, "GPR")  # ✅加这一行

    alpha_beta_records = []

    gpr_feat_names = np.load(
        os.path.join(ae_in_dir, "gpr_feat_names.npy"),
        allow_pickle=True
    )

    collect_alpha_beta(
        alpha_beta_records,
        modality="GPR",
        feature_names=gpr_feat_names,
        alpha=a_g,
        beta=b_g
    )

    out_gpr = os.path.join(BAYES_ROOT, "IntraModal_GPR")
    _save_bayes_pack(out_gpr, "GPR", post_gpr, mask_gpr, a_g, b_g, loss_g, GT)

    export_map_2d(
        post_gpr,
        _exp_dir("BAYES_PostMaps"),
        "Post_GPR",
        "Bayes Post – GPR (145×64)"
    )
    out_pm = _exp_dir("BAYES_PostMaps")

    if GT is not None:
        metrics_rows.append(
            evaluate_all_metrics(mask_gpr.astype(int), GT, prefix="GPR")
        )

    print(f"[Average] GPR feature-map mean baseline, N={prob_gpr.shape[2]}")
    post_gpr_avg, mask_gpr_avg = mean_prob_maps_threshold(prob_gpr, thresh=0.5)
    out_gpr_avg = os.path.join(BAYES_ROOT, "IntraModal_GPR_Average")
    _save_bayes_pack(
        out_gpr_avg,
        "GPR_Average10",
        post_gpr_avg,
        mask_gpr_avg,
        alpha=np.array([]),
        beta=np.array([]),
        loss=np.array([]),
        GT=GT
    )
    export_map_2d(
        post_gpr_avg,
        _exp_dir("BAYES_PostMaps"),
        "Post_GPR_Average10",
        "Average Post - GPR 10 feature maps"
    )

    if GT is not None:
        metrics_rows.append(
            evaluate_all_metrics(mask_gpr_avg.astype(int), GT, prefix="GPR_Average10")
        )

    # =========================================================
    # If no Knock → stop here
    # =========================================================
    if prob_knock is None:
        print("[Step6] Knock not available, finish with GPR only.")
        return

    feat_names_sel = np.load(
        os.path.join(ae_in_dir, "knock_feat_names.npy"),
        allow_pickle=True
    )

    # =========================================================
    # ② Knock intra-modal (all channels together)
    # =========================================================
    print("[Bayes] Intra-modal: Knock (all channels)")
    post_knock, mask_knock, a_k, b_k, loss_k = bayesPixelFusion3_optimized(prob_knock)
    export_map_raw_and_vis(
        post_knock, out_pm,
        "Post_Knock_All_29x64",
        "Bayes Post – Knock (All Sensors)"
    )
    out_knock = os.path.join(BAYES_ROOT, "IntraModal_Knock_All")

    save_bayes_iter_loss(loss_k, out_knock, "Knock_All")  # ✅加这一行

    collect_alpha_beta(
        alpha_beta_records,
        modality="Knock_All",
        feature_names=feat_names_sel,
        alpha=a_k,
        beta=b_k
    )

    post_knock_up = resize_hw(post_knock, (Hg, Wg), order=1)
    mask_knock_up = (resize_hw(mask_knock, (Hg, Wg), order=0) > 0.5).astype(float)

    out_knock = os.path.join(BAYES_ROOT, "IntraModal_Knock_All")
    _save_bayes_pack(
        out_knock,
        "Knock_All",
        post_knock_up,
        mask_knock_up,
        a_k,
        b_k,
        loss_k,
        GT
    )

    export_map_raw_and_vis(post_knock_up, out_pm, "Post_Knock_All", "Bayes Post – Knock (All Sensors)")


    if GT is not None:
        metrics_rows.append(
            evaluate_all_metrics(mask_knock_up.astype(int), GT, prefix="Knock_All")
        )

    print(f"[Average] Knock feature-map mean baseline, N={prob_knock.shape[2]}")
    post_knock_avg = np.mean(prob_knock, axis=2)
    mask_knock_avg = (post_knock_avg > 0.5).astype(float)
    export_map_raw_and_vis(
        post_knock_avg,
        out_pm,
        "Post_Knock_Average30_29x64",
        "Average Post - Knock 30 feature maps"
    )
    post_knock_avg_up = resize_hw(post_knock_avg, (Hg, Wg), order=1)
    mask_knock_avg_up = (resize_hw(mask_knock_avg, (Hg, Wg), order=0) > 0.5).astype(float)
    out_knock_avg = os.path.join(BAYES_ROOT, "IntraModal_Knock_Average")
    _save_bayes_pack(
        out_knock_avg,
        "Knock_Average30",
        post_knock_avg_up,
        mask_knock_avg_up,
        alpha=np.array([]),
        beta=np.array([]),
        loss=np.array([]),
        GT=GT
    )
    export_map_2d(
        post_knock_avg_up,
        _exp_dir("BAYES_PostMaps"),
        "Post_Knock_Average30",
        "Average Post - Knock 30 feature maps"
    )

    if GT is not None:
        metrics_rows.append(
            evaluate_all_metrics(mask_knock_avg_up.astype(int), GT, prefix="Knock_Average30")
        )

    # =========================================================
    # ③ Knock per-sensor Bayes
    # =========================================================
    print("[Bayes] Intra-modal: Knock per sensor")

    for ch in [0, 1, 2, 3]:
        idx = [
            i for i, n in enumerate(feat_names_sel)
            if str(n).startswith(f"Ch{ch}_")
        ]
        if not idx:
            print(f"[Bayes] Ch{ch}: no features, skip.")
            continue

        print(f"[Bayes] Knock sensor Ch{ch}, N_feat={len(idx)}")

        prob_ch = prob_knock[:, :, idx]

        post_ch, mask_ch, a_c, b_c, loss_c = bayesPixelFusion3_optimized(prob_ch)
        out_ch = os.path.join(BAYES_ROOT, f"IntraModal_Knock_Ch{ch}")

        save_bayes_iter_loss(loss_c, out_ch, f"Knock_Ch{ch}")  # ✅加这一行

        post_ch_up = resize_hw(post_ch, (Hg, Wg), order=1)
        post_ch_up = np.clip(post_ch_up, 0.0, 1.0)
        mask_ch_up = (resize_hw(mask_ch, (Hg, Wg), order=0) > 0.5).astype(float)

        out_ch = os.path.join(BAYES_ROOT, f"IntraModal_Knock_Ch{ch}")
        _save_bayes_pack(
            out_ch,
            f"Knock_Ch{ch}",
            post_ch_up,
            mask_ch_up,
            a_c,
            b_c,
            loss_c,
            GT
        )

        export_map_2d(
            post_ch_up,
            _exp_dir("BAYES_PostMaps"),
            f"Post_Knock_Ch{ch}",
            f"Bayes Post – Knock Ch{ch}"
        )

        if GT is not None:
            metrics_rows.append(
                evaluate_all_metrics(
                    mask_ch_up.astype(int),
                    GT,
                    prefix=f"Knock_Ch{ch}"
                )
            )

    # =========================================================
    # ④ Cross-modal fusion (Knock_all + GPR)
    # =========================================================
    print("[Bayes] Cross-modal fusion: Knock_All + GPR")

    raw_w_k = max(rel_weight(a_k, b_k), 1e-6)
    raw_w_g = max(rel_weight(a_g, b_g), 1e-6)
    w_sum = raw_w_k + raw_w_g
    w_k = raw_w_k / w_sum
    w_g = raw_w_g / w_sum
    print(
        f"[Fusion Weights] "
        f"Knock raw={raw_w_k:.3f}, GPR raw={raw_w_g:.3f}; "
        f"normalized -> Knock={w_k:.3f}, GPR={w_g:.3f}"
    )

    post_final = logit_fuse(post_knock_up, post_gpr, w1=raw_w_k, w2=raw_w_g)
    mask_final = (post_final > 0.5).astype(float)
    # =========================================================
    # CrossModal (Post-level): stack([post_knock_up, post_gpr]) -> Bayes
    # =========================================================
    print("[Bayes] Cross-modal fusion (Post-level): stack([post_knock_up, post_gpr])")

    prob_cross_post = np.stack([post_knock_up, post_gpr], axis=2).astype(np.float32)  # (145,64,2)
    prob_cross_post = np.clip(prob_cross_post, 1e-6, 1.0 - 1e-6)

    post_final_post, mask_final_post, a_cm, b_cm, loss_cm = bayesPixelFusion3_optimized(
        prob_cross_post, thresh=0.5, maxIter=50
    )

    collect_alpha_beta(
        alpha_beta_records,
        modality="CrossModal_PostLevel",
        feature_names=["Knock_All_Post", "GPR_Post"],
        alpha=a_cm,
        beta=b_cm
    )

    out_cm = os.path.join(BAYES_ROOT, "CrossModal_PostLevel_145x64")

    # ✅ 导出“迭代曲线”（csv + png）
    save_bayes_iter_loss(loss_cm, out_cm, "Cross_PostLevel")

    # （可选）你原来的保存包：会再额外保存 loss.png / loss.csv
    _save_bayes_pack(
        out_cm,
        "Final_PostLevel",
        post_final_post,
        mask_final_post,
        a_cm,
        b_cm,
        loss_cm,
        GT=GT
    )

    out_bayes = os.path.join(BAYES_ROOT, "CrossModal_Bayes_145x64")
    _save_bayes_pack(
        out_bayes,
        "Final_Bayes",
        post_final_post,
        mask_final_post,
        a_cm,
        b_cm,
        loss_cm,
        GT=GT
    )

    export_map_2d(
        post_final_post,
        _exp_dir("BAYES_PostMaps"),
        "Post_Final_Bayes",
        "Bayes Post - Final_Bayes (Knock + GPR)"
    )

    df_ab = export_alpha_beta_csv(
        alpha_beta_records,
        out_dir=os.path.join(BAYES_ROOT, "AlphaBeta")
    )

    for modality in ["GPR", "Knock_All", "CrossModal_PostLevel"]:
        plot_alpha_beta_bar(
            df_ab,
            modality=modality,
            out_dir=os.path.join(BAYES_ROOT, "AlphaBeta", "BarPlots")
        )

    w_k = rel_weight(a_k, b_k)  # Knock 权重
    w_g = rel_weight(a_g, b_g)  # GPR 权重

    print(f"[Fusion Weights] Knock w={w_k:.3f}, GPR w={w_g:.3f}")

    out_final = os.path.join(BAYES_ROOT, "CrossModal_Final_145x64")
    _save_bayes_pack(
        out_final,
        "Final_Weighted",
        post_final,
        mask_final,
        alpha=np.array([raw_w_k, raw_w_g]),
        beta=np.array([0.0, 0.0]),
        loss=np.array([]),
        GT=GT
    )

    export_map_2d(
        post_final,
        _exp_dir("BAYES_PostMaps"),
        "Post_Final_Weighted",
        "Bayes Post – Final (Knock + GPR)"
    )

    prob_knock_up_all = resize_hw(prob_knock, (Hg, Wg), order=1)
    prob_all_40 = np.concatenate([prob_knock_up_all, prob_gpr], axis=2)
    print(
        f"[Average] All-feature mean baseline: "
        f"Knock={prob_knock_up_all.shape[2]}, GPR={prob_gpr.shape[2]}, Total={prob_all_40.shape[2]}"
    )
    post_final_avg, mask_final_avg = mean_prob_maps_threshold(prob_all_40, thresh=0.5)
    out_avg = os.path.join(BAYES_ROOT, "CrossModal_Average40_145x64")
    _save_bayes_pack(
        out_avg,
        "Final_Average40",
        post_final_avg,
        mask_final_avg,
        alpha=np.array([]),
        beta=np.array([]),
        loss=np.array([0.0]),
        GT=GT
    )

    export_map_2d(
        post_final_avg,
        _exp_dir("BAYES_PostMaps"),
        "Post_Final_Average40",
        "Average Post - 40 feature maps (Knock 30 + GPR 10)"
    )

    if GT is not None:
        metrics_rows.append(
            evaluate_all_metrics(mask_final.astype(int), GT, prefix="Final_Weighted")
        )
        metrics_rows.append(
            evaluate_all_metrics(mask_final_post.astype(int), GT, prefix="Final_Bayes")
        )
        metrics_rows.append(
            evaluate_all_metrics(mask_final_avg.astype(int), GT, prefix="Final_Average40")
        )

    # =========================================================
    # Save all metrics
    # =========================================================
    if metrics_rows:
        df = pd.DataFrame(metrics_rows)
        csv_path = os.path.join(out_final, "All_Modalities_Metrics.csv")
        csv_path_root = os.path.join(BAYES_ROOT, "All_Modalities_Metrics.csv")
        df.to_csv(csv_path, index=False)
        df.to_csv(csv_path_root, index=False)

        print("\n[✓] Metrics summary:")
        print(df)

    print("[✓] Step6 Bayes fusion completed.")



# ===========================================================
# 12) Main
# ===========================================================
set_global_seed(BASE_SEED)

if __name__ == "__main__":
    t0 = time.time()
    run_mode = apply_run_mode_from_env()
    _reset_runtime_profile()

    # =========================================================
    # 前处理（只跑一次，不随 k 变化）
    # =========================================================
    if RUN_SWITCH["step1_txt2npy"]:
        timed_runtime_call("Data reading", "Step1 TXT -> NPY", step1_convert_txt_to_npy)
    if RUN_SWITCH["step2_feature"]:
        timed_runtime_call("Feature extraction", "Step2 knock feature extraction", step2_extract_features)
    if RUN_SWITCH["step3_matrix"]:
        timed_runtime_call("Preprocessing", "Step3 knock matrix transform", step3_matrix_transform)
    if RUN_SWITCH["step4_stack"]:
        timed_runtime_call("Preprocessing", "Step4 knock feature stacking", step4_stack_feature_matrices)

    if RUN_SWITCH["step0B_gpr_extract_features"]:
        step0B_gpr_extract_features_to_stack()

    # =========================================================
    # 实验标签由顶部配置区统一控制
    # =========================================================
    EXP_TAG = DEFAULT_EXP_TAG
    set_global_seed(BASE_SEED)

    if RUN_SWITCH["step5A_AE_knock"]:
        step5A_AE_knock()
    if RUN_SWITCH["step5B_AE_gpr"]:
        step5B_AE_gpr()
    if RUN_SWITCH["step6_bayes_fusion"]:
        timed_runtime_call("Fusion", "Step6 fusion comparison", step6_bayes_fusion_multimodal)

    total_seconds = time.time() - t0
    write_runtime_profile(total_seconds=total_seconds, run_mode=run_mode)

    print("\n[Runtime] Stage totals:")
    for stage in RUNTIME_STAGE_ORDER:
        print(f"  - {stage:<24} {RUNTIME_STAGE_SECONDS[stage]:8.2f}s")

    print(f"\n=========== ALL DONE ===========  time={total_seconds:.1f}s")

