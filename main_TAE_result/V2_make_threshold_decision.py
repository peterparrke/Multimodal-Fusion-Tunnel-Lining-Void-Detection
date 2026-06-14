from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, to_rgb
from scipy.ndimage import binary_dilation, label, zoom


# ============================================================
# 基础路径与阈值
# 这里控制输入/输出位置，以及二值化阈值
# ============================================================
THRESHOLD = 0.5
ROOT = Path(r"C:\GUO\jinqiang\Paper2\New_plot\result_new")
OUT_DIR = ROOT / "plot_2"
GT_PATH = ROOT / "Groundtruth_4.xlsx"
BAYES_RESULTS_ROOT = ROOT / "bayes_results" / "sdcq_2" / "exp_col_list"
INPUTS = {
    "gpr_ae_mean": ROOT / "sdcq_2" / "exp_col_list" / "GPR_AE" / "GPR_AE_Mean.npy",
    "knock_ae_ch1": ROOT / "sdcq_2" / "exp_col_list" / "Knock_AE_PerSensor" / "AE_Prob_Ch1.npy",
    "knock_ae_ch2": ROOT / "sdcq_2" / "exp_col_list" / "Knock_AE_PerSensor" / "AE_Prob_Ch2.npy",
    "knock_ae_ch3": ROOT / "sdcq_2" / "exp_col_list" / "Knock_AE_PerSensor" / "AE_Prob_Ch3.npy",
}
FINAL_MASK_INPUTS = {
    "CNNFinal": ROOT / "CNNFinal_Mask.csv",
    "DNNFinal": ROOT / "DNNFinal_Mask.csv",
    "TAEFinal": ROOT / "TAEFinal_Mask.csv",
}


# ============================================================
# 手动可调的配色 / 样式配置
# 后续如果你想改颜色、图大小、图例位置，优先改这里
# ============================================================
GLOBAL_COLORS = {
    # 全局通用颜色
    "panel_background": "#EDF4F1",
    "groundtruth_positive": "#DBCB92",
    "groundtruth_outline": "#0C454F",
    # "groundtruth_outline": "#000000",
}

# 统一在这里手动调整三类结果的配色：
# 1. probability_colors: 概率图 0->1 的渐变色，按“浅 -> 中 -> 深”填写
# 2. binary_positive: 二值图中的阳性颜色
# 3. tp / fp / fn: 像素级对比图中的 TP / FP / FN 颜色
STYLE_THEMES = {
    # GPR 系列：蓝色
    "gpr": {
        "probability_colors": ["#EEF5FF", "#92B3E4", "#055F73"],
        "binary_positive": "#2563EB",
        "tp": "#ddf5b1",
        "fp": "#F97F5F",
        "fn": "#8080b0",
    },
    # Knock 系列：紫红色
    "knock": {
        "probability_colors": ["#FFF1F6", "#CEC8AC", "#81785A"],
        "binary_positive": "#34675E",
        "tp": "#ddf5b1",
        "fp": "#F97F5F",
        "fn": "#8080b0",
    },
    # Final / Fusion 系列：绿色
    "final": {
        "probability_colors": ["#F0FDF4", "#F0C0A7", "#439466"],
        "binary_positive": "#0C2F20",
        # "tp": "#00FF00",
        # "fp": "#F66E14",
        # "fn": "#233AE7",
        "tp": "#ddf5b1",
        "fp": "#F97F5F",
        "fn": "#8080b0",
    },
}

LAYOUT = {
    # 所有矩阵图统一按 64:29 的视觉比例来设置画布
    "matrix_display_ratio": 64 / 29,
    "heatmap_figsize": (9.5, 9.5 * 29 / 64),
    "heatmap_dpi": 200,
    "panel_figsize": (7.2, 7.2 * 29 / 64),
    "panel_dpi": 220,

    # 指标框字体、子图边距、图例位置
    "heatmap_adjust": {"left": 0.02, "right": 0.94, "top": 0.90, "bottom": 0.06},
}

RENDER = {
    "origin": "upper",
    "aspect": "auto",
    "interpolation": "nearest",
    "resample": False,
}


# ============================================================
# 通用小工具函数
# 这些函数主要用于减少重复代码
# ============================================================
def save_array_csv(array: np.ndarray, path: Path, fmt: str) -> None:
    """把矩阵保存成 CSV，便于后续检查或导入其他软件。"""
    np.savetxt(path, array, fmt=fmt, delimiter=",")


def render_image(ax: plt.Axes, array: np.ndarray, **kwargs):
    """统一图像渲染入口，避免每次 imshow 都重复写参数。"""
    options = dict(RENDER)
    options.update(kwargs)
    return ax.imshow(array, **options)


def infer_style_key(name: str) -> str:
    """
    根据名称自动判断应该使用哪套配色主题。
    - 含 gpr -> gpr
    - 含 knock / ch1 / ch2 / ch3 -> knock
    - 其他默认按 final 处理
    """
    lower_name = name.lower()
    if "gpr" in lower_name:
        return "gpr"
    if "knock" in lower_name or "ch1" in lower_name or "ch2" in lower_name or "ch3" in lower_name:
        return "knock"
    return "final"


def matrix_hw(array: np.ndarray) -> tuple[int, int]:
    """返回矩阵的高和宽。"""
    return int(array.shape[0]), int(array.shape[1])


def target_coordinate_shape(height: int, width: int) -> tuple[int, int]:
    """
    将矩阵像素尺寸映射为你关心的物理坐标尺寸。
    例如：
    - 145 行对应 29 个扫描位置
    - 64 列保持 64 个测点
    - 320 列也映射回 64 列坐标
    """
    target_height = 29 if height == 145 else height
    target_width = 64 if width in (64, 320) else width
    return target_height, target_width


def apply_matrix_display_ratio(ax: plt.Axes, array: np.ndarray) -> None:
    """
    让所有矩阵图都按照“坐标比例”显示，而不是按照原始像素数显示。
    这样不同来源的矩阵图视觉比例就能统一。
    """
    height, width = matrix_hw(array)
    target_height, target_width = target_coordinate_shape(height, width)
    data_ratio = height / width
    target_ratio = target_height / target_width
    ax.set_aspect(target_ratio / data_ratio)


def binary_cmap(positive_color: str) -> ListedColormap:
    """生成二值图的配色：背景色 + 前景色。"""
    return ListedColormap([GLOBAL_COLORS["panel_background"], positive_color])


def probability_cmap(style_key: str) -> LinearSegmentedColormap:
    """生成某个模态专属的 0-1 概率渐变色。"""
    return LinearSegmentedColormap.from_list(
        f"{style_key}_probability",
        STYLE_THEMES[style_key]["probability_colors"],
        N=256,
    )


def agreement_rgb(mask: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """根据预测与 GT 的关系，生成 TP / FP / FN 的彩色对比图。"""
    style_key = "final"
    rgb = np.ones((gt.shape[0], gt.shape[1], 3), dtype=np.float32)
    rgb[np.logical_and(mask == 1, gt == 1)] = np.array(to_rgb(STYLE_THEMES[style_key]["tp"]), dtype=np.float32)
    rgb[np.logical_and(mask == 1, gt == 0)] = np.array(to_rgb(STYLE_THEMES[style_key]["fp"]), dtype=np.float32)
    rgb[np.logical_and(mask == 0, gt == 1)] = np.array(to_rgb(STYLE_THEMES[style_key]["fn"]), dtype=np.float32)
    return rgb


def agreement_rgb_with_style(mask: np.ndarray, gt: np.ndarray, style_key: str) -> np.ndarray:
    """根据指定模态主题，生成 TP / FP / FN 彩色对比图。"""
    rgb = np.ones((gt.shape[0], gt.shape[1], 3), dtype=np.float32)
    rgb[np.logical_and(mask == 1, gt == 1)] = np.array(to_rgb(STYLE_THEMES[style_key]["tp"]), dtype=np.float32)
    rgb[np.logical_and(mask == 1, gt == 0)] = np.array(to_rgb(STYLE_THEMES[style_key]["fp"]), dtype=np.float32)
    rgb[np.logical_and(mask == 0, gt == 1)] = np.array(to_rgb(STYLE_THEMES[style_key]["fn"]), dtype=np.float32)
    return rgb


def save_panel(
    array: np.ndarray,
    path: Path,
    title: str | None = None,
    cmap=None,
    vmin=None,
    vmax=None,
    contour_gt: np.ndarray | None = None,
    is_rgb: bool = False,
) -> None:
    """
    统一单张图的保存逻辑。
    这个函数可以保存：
    - 二值检测图
    - GT 图
    - TP/FP/FN 彩色对比图
    """
    fig, ax = plt.subplots(figsize=LAYOUT["panel_figsize"], dpi=LAYOUT["panel_dpi"])
    fig.patch.set_facecolor("white")

    if is_rgb:
        render_image(ax, array)
    else:
        render_image(ax, array, cmap=cmap, vmin=vmin, vmax=vmax)

    apply_matrix_display_ratio(ax, array)

    if contour_gt is not None:
        ax.contour(
            contour_gt.astype(float),
            levels=[0.5],
            colors=GLOBAL_COLORS["groundtruth_outline"],
            linewidths=1.2,
            linestyles="dashed",
        )

    if title:
        ax.set_title(title)
    ax.axis("off")
    ax.set_position([0.0, 0.0, 1.0, 1.0])
    fig.savefig(path, bbox_inches="tight", pad_inches=0, facecolor="white")
    plt.close(fig)


def save_heatmap(array: np.ndarray, path: Path, title: str, cmap, vmin: float, vmax: float) -> None:
    """保存连续值矩阵的热图，例如概率图、投票图。"""
    fig, ax = plt.subplots(figsize=LAYOUT["heatmap_figsize"], dpi=LAYOUT["heatmap_dpi"])
    image = render_image(ax, array, cmap=cmap, vmin=vmin, vmax=vmax)
    apply_matrix_display_ratio(ax, array)
    ax.set_title(title)
    ax.axis("off")
    fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
    fig.subplots_adjust(**LAYOUT["heatmap_adjust"])
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ============================================================
# 数据读取与指标计算
# ============================================================
def describe_array(name: str, array: np.ndarray) -> str:
    """生成矩阵的统计描述，写入 summary 文本。"""
    return (
        f"{name}: shape={array.shape}, dtype={array.dtype}, "
        f"min={float(np.min(array)):.6f}, max={float(np.max(array)):.6f}, "
        f"mean={float(np.mean(array)):.6f}"
    )


def load_and_resize_gt(gt_path: Path, target_shape: tuple[int, int]) -> np.ndarray | None:
    """读取 Groundtruth_4，并按目标矩阵尺寸做最近邻缩放。"""
    if not gt_path.exists():
        return None

    gt = pd.read_excel(gt_path, header=None).values
    gt = (gt > 0).astype(np.uint8)

    if gt.shape != target_shape:
        gt = zoom(gt, (target_shape[0] / gt.shape[0], target_shape[1] / gt.shape[1]), order=0)

    return gt.astype(np.uint8)


def evaluate_metrics(mask: np.ndarray, gt: np.ndarray) -> dict[str, float | int]:
    """计算像素级指标：IoU / F1 / Precision / Recall / TP / FP / FN。"""
    mask = mask.astype(np.uint8)
    gt = gt.astype(np.uint8)

    tp = int(np.logical_and(mask == 1, gt == 1).sum())
    fp = int(np.logical_and(mask == 1, gt == 0).sum())
    fn = int(np.logical_and(mask == 0, gt == 1).sum())

    precision = tp / (tp + fp + 1e-12)
    recall = tp / (tp + fn + 1e-12)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-12)
    iou = tp / (tp + fp + fn + 1e-12)

    return {
        "IoU": float(iou),
        "F1": float(f1),
        "Precision": float(precision),
        "Recall": float(recall),
        "TP": tp,
        "FP": fp,
        "FN": fn,
    }


def compute_object_level_metrics(pred_mask: np.ndarray, gt_mask: np.ndarray, iou_thresh: float = 0.2) -> dict[str, float | int]:
    """计算目标级指标：GT 目标召回、预测目标个数、误检目标个数等。"""
    gt_lab, gt_num = label(gt_mask)
    pr_lab, pr_num = label(pred_mask)

    matched_gt = 0
    matched_pred = set()

    for gt_index in range(1, gt_num + 1):
        gt_object = gt_lab == gt_index
        overlapping_pred_objects = []

        for pred_index in range(1, pr_num + 1):
            pred_object = pr_lab == pred_index
            if np.logical_and(gt_object, pred_object).any():
                overlapping_pred_objects.append((pred_index, pred_object))

        if not overlapping_pred_objects:
            continue

        merged_pred_object = np.zeros_like(gt_object, dtype=bool)
        for pred_index, pred_object in overlapping_pred_objects:
            merged_pred_object |= pred_object

        inter = np.logical_and(gt_object, merged_pred_object).sum()
        union = np.logical_or(gt_object, merged_pred_object).sum()
        if inter / (union + 1e-12) >= iou_thresh:
            matched_gt += 1
            for pred_index, _ in overlapping_pred_objects:
                matched_pred.add(pred_index)

    return {
        "Object_Recall": float(matched_gt / (gt_num + 1e-12)),
        "GT_Objects": int(gt_num),
        "Pred_Objects": int(pr_num),
        "FP_Objects": int(pr_num - len(matched_pred)),
        "Matched_Objects": int(matched_gt),
    }


def relaxed_iou(pred_mask: np.ndarray, gt_mask: np.ndarray, radius: int) -> float:
    """计算带容忍半径的 relaxed IoU。"""
    structure = np.ones((2 * radius + 1, 2 * radius + 1), dtype=bool)
    gt_dilated = binary_dilation(gt_mask, structure=structure)
    inter = np.logical_and(pred_mask, gt_dilated).sum()
    union = np.logical_or(pred_mask, gt_dilated).sum()
    return float(inter / (union + 1e-12))


def evaluate_all_metrics(mask: np.ndarray, gt: np.ndarray, prefix: str) -> dict[str, float | int | str]:
    """汇总像素级、目标级、relaxed IoU 三类指标。"""
    row = {"Modality": prefix}
    row.update(evaluate_metrics(mask, gt))
    row.update(compute_object_level_metrics(mask.astype(int), gt.astype(int), iou_thresh=0.2))
    row.update(
        {
            "Relaxed_IoU_r1": relaxed_iou(mask, gt, radius=1),
            "Relaxed_IoU_r2": relaxed_iou(mask, gt, radius=2),
            "Relaxed_IoU_r3": relaxed_iou(mask, gt, radius=3),
        }
    )
    return row


# ============================================================
# 图片导出
# ============================================================
def metrics_to_text(metrics: dict[str, float | int]) -> str:
    """把关键指标转成左上角文本框里的字符串。"""
    return (
        f"IoU={metrics['IoU']:.3f}\n"
        f"F1={metrics['F1']:.3f}\n"
        f"Precision={metrics['Precision']:.3f}\n"
        f"Recall={metrics['Recall']:.3f}"
    )


def save_detection_gt_agreement_images(tag: str, output_dir: Path, mask: np.ndarray, gt: np.ndarray) -> None:
    """
    将原来的三联图拆成三张单图分别保存：
    1. Binary detection
    2. Ground truth
    3. Pixel-wise agreement
    """
    style_key = infer_style_key(f"{output_dir}_{tag}")
    save_panel(
        mask,
        output_dir / f"{tag}_BinaryDetection.png",
        title=None,
        cmap=binary_cmap(STYLE_THEMES[style_key]["binary_positive"]),
        vmin=0,
        vmax=1,
        contour_gt=gt,
    )

    save_panel(
        gt,
        output_dir / f"{tag}_GroundTruth.png",
        title=None,
        cmap=binary_cmap(GLOBAL_COLORS["groundtruth_positive"]),
        vmin=0,
        vmax=1,
        contour_gt=gt,
    )

    save_panel(
        agreement_rgb_with_style(mask, gt, style_key),
        output_dir / f"{tag}_PixelAgreement.png",
        title=None,
        contour_gt=gt,
        is_rgb=True,
    )

    stale_compare_path = output_dir / f"{tag}_Compare_GT.png"
    if stale_compare_path.exists():
        # 删除旧版三联图，避免和当前拆分后的三张图混淆
        stale_compare_path.unlink()


def save_mask_only_figure(mask: np.ndarray, title: str, path: Path, contour_gt: np.ndarray | None = None) -> None:
    """单独保存 mask 图，保持当前统一配色。"""
    style_key = infer_style_key(str(path))
    save_panel(
        mask,
        path,
        title=None,
        cmap=binary_cmap(STYLE_THEMES[style_key]["binary_positive"]),
        vmin=0,
        vmax=1,
        contour_gt=contour_gt,
    )


def batch_render_bayes_mask_figures(root_dir: Path, gt_path: Path) -> list[str]:
    """
    批量扫描 bayes_results 目录中的 *_Mask.csv，
    为每个结果目录输出：
    - Mask.png
    - BinaryDetection.png
    - GroundTruth.png
    - PixelAgreement.png
    """
    outputs: list[str] = []

    for mask_path in sorted(root_dir.rglob("*_Mask.csv")):
        tag = mask_path.stem.removesuffix("_Mask")
        output_dir = mask_path.parent
        mask = np.loadtxt(mask_path).astype(np.uint8)
        gt = load_and_resize_gt(gt_path, mask.shape)

        mask_png_path = output_dir / f"{tag}_Mask.png"
        save_mask_only_figure(mask, f"{tag} mask", mask_png_path, contour_gt=gt)
        outputs.append(str(mask_png_path))

        if gt is None:
            continue

        save_detection_gt_agreement_images(tag, output_dir, mask, gt)
        outputs.extend(
            [
                str(output_dir / f"{tag}_BinaryDetection.png"),
                str(output_dir / f"{tag}_GroundTruth.png"),
                str(output_dir / f"{tag}_PixelAgreement.png"),
            ]
        )

    return outputs


def render_final_mask_comparisons(mask_files: dict[str, Path], gt_path: Path, output_dir: Path) -> tuple[list[str], list[dict[str, float | int | str]]]:
    """
    读取根目录下给定的最终二值 mask，并生成它们与 Groundtruth_4 的对比图。
    每个矩阵输出：
    - Mask.png
    - BinaryDetection.png
    - GroundTruth.png
    - PixelAgreement.png
    """
    outputs: list[str] = []
    metric_rows: list[dict[str, float | int | str]] = []

    for tag, mask_path in mask_files.items():
        if not mask_path.exists():
            continue

        mask = np.loadtxt(mask_path).astype(np.float32)
        mask = (mask > 0.5).astype(np.uint8)
        gt = load_and_resize_gt(gt_path, mask.shape)

        mask_png_path = output_dir / f"{tag}_Mask.png"
        save_mask_only_figure(mask, f"{tag} mask", mask_png_path, contour_gt=gt)
        outputs.append(str(mask_png_path))

        if gt is None:
            continue

        save_detection_gt_agreement_images(tag, output_dir, mask, gt)
        outputs.extend(
            [
                str(output_dir / f"{tag}_BinaryDetection.png"),
                str(output_dir / f"{tag}_GroundTruth.png"),
                str(output_dir / f"{tag}_PixelAgreement.png"),
            ]
        )
        metric_rows.append(evaluate_all_metrics(mask, gt, prefix=tag))

    return outputs, metric_rows


# ============================================================
# 主流程
# ============================================================
def main() -> None:
    """主入口：读取矩阵、生成平均融合结果、评估、批量导出图片。"""
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    probability_maps: dict[str, np.ndarray] = {}
    target_shape = None

    # 1) 读取 4 个输入概率矩阵，并检查尺寸一致
    for name, path in INPUTS.items():
        probability_map = np.asarray(np.load(path), dtype=np.float32)
        if target_shape is None:
            target_shape = probability_map.shape
        elif probability_map.shape != target_shape:
            raise ValueError(f"Shape mismatch: expected {target_shape}, got {probability_map.shape} for {path}")
        probability_maps[name] = np.clip(probability_map, 0.0, 1.0)

    # 2) 对每一路概率图按阈值生成二值 mask
    binary_masks = {
        name: (probability_map > THRESHOLD).astype(np.uint8)
        for name, probability_map in probability_maps.items()
    }

    # 3) 做平均融合，并计算最终二值决策
    stacked_probability_maps = np.stack([probability_maps[name] for name in INPUTS], axis=2)
    fused_mean_probability = np.mean(stacked_probability_maps, axis=2).astype(np.float32)
    fused_decision_mask = (fused_mean_probability > THRESHOLD).astype(np.uint8)
    positive_vote_count = np.sum(stacked_probability_maps > THRESHOLD, axis=2).astype(np.uint8)

    # 4) 保存每一路输入概率图和对应的 mask
    for name, probability_map in probability_maps.items():
        style_key = infer_style_key(name)
        np.save(OUT_DIR / f"{name}_prob.npy", probability_map)
        save_array_csv(probability_map, OUT_DIR / f"{name}_prob.csv", fmt="%.6f")
        save_heatmap(
            probability_map,
            OUT_DIR / f"{name}_prob.png",
            title=f"{name} probability",
            cmap=probability_cmap(style_key),
            vmin=0.0,
            vmax=1.0,
        )

    gt = load_and_resize_gt(GT_PATH, fused_decision_mask.shape)

    for name, mask in binary_masks.items():
        np.save(OUT_DIR / f"{name}_mask_thr_0p5.npy", mask)
        save_array_csv(mask, OUT_DIR / f"{name}_mask_thr_0p5.csv", fmt="%d")
        save_mask_only_figure(mask, f"{name} decision mask (thr=0.5)", OUT_DIR / f"{name}_mask_thr_0p5.png", contour_gt=gt)

    # 5) 保存融合后的概率图、mask、投票图
    np.save(OUT_DIR / "fused_mean_prob.npy", fused_mean_probability)
    save_array_csv(fused_mean_probability, OUT_DIR / "fused_mean_prob.csv", fmt="%.6f")
    save_heatmap(
        fused_mean_probability,
        OUT_DIR / "fused_mean_prob.png",
        title="Fused mean probability",
        cmap=probability_cmap("final"),
        vmin=0.0,
        vmax=1.0,
    )

    np.save(OUT_DIR / "fused_decision_mask_thr_0p5.npy", fused_decision_mask)
    save_array_csv(fused_decision_mask, OUT_DIR / "fused_decision_mask_thr_0p5.csv", fmt="%d")
    save_mask_only_figure(fused_decision_mask, "Fused decision mask (thr=0.5)", OUT_DIR / "fused_decision_mask_thr_0p5.png", contour_gt=gt)

    np.save(OUT_DIR / "positive_vote_count.npy", positive_vote_count)
    save_array_csv(positive_vote_count, OUT_DIR / "positive_vote_count.csv", fmt="%d")
    save_heatmap(
        positive_vote_count,
        OUT_DIR / "positive_vote_count.png",
        title="Positive vote count (>0.5)",
        cmap=probability_cmap("final"),
        vmin=0.0,
        vmax=float(len(INPUTS)),
    )

    metric_row = None
    rendered_bayes_images: list[str] = []
    rendered_final_mask_images: list[str] = []
    final_mask_metric_rows: list[dict[str, float | int | str]] = []

    # 6) 如果存在 GT，则保存 GT 并输出评估图和指标
    if gt is not None:
        np.save(OUT_DIR / "groundtruth_4_resized.npy", gt)
        save_array_csv(gt, OUT_DIR / "groundtruth_4_resized.csv", fmt="%d")
        save_panel(
            gt,
            OUT_DIR / "groundtruth_4_resized.png",
            title=None,
            cmap=binary_cmap(GLOBAL_COLORS["groundtruth_positive"]),
            vmin=0,
            vmax=1,
            contour_gt=gt,
        )

        metric_row = evaluate_all_metrics(fused_decision_mask, gt, prefix="AverageFusion_4Maps")
        pd.DataFrame([metric_row]).to_csv(OUT_DIR / "average_fusion_metrics_with_gt.csv", index=False)
        save_detection_gt_agreement_images("AverageFusion", OUT_DIR, fused_decision_mask, gt)
        stale_compare_path = OUT_DIR / "average_fusion_vs_groundtruth4.png"
        if stale_compare_path.exists():
            stale_compare_path.unlink()

        rendered_final_mask_images, final_mask_metric_rows = render_final_mask_comparisons(FINAL_MASK_INPUTS, GT_PATH, OUT_DIR)
        if final_mask_metric_rows:
            pd.DataFrame(final_mask_metric_rows).to_csv(OUT_DIR / "final_mask_metrics_with_gt.csv", index=False)

    # 7) 批量处理 bayes_results 下所有 mask 结果图
    if BAYES_RESULTS_ROOT.exists():
        rendered_bayes_images = batch_render_bayes_mask_figures(BAYES_RESULTS_ROOT, GT_PATH)

    # 8) 输出 summary 文本
    summary_lines = [
        "Threshold decision summary",
        f"Threshold: {THRESHOLD}",
        "Rule: mean the 4 probability maps, then apply decision mask = (mean_prob > 0.5).",
        "",
        "Inputs:",
    ]
    summary_lines.extend(describe_array(name, array) for name, array in probability_maps.items())
    summary_lines.extend(
        [
            "",
            describe_array("fused_mean_prob", fused_mean_probability),
            f"fused_decision positive_pixels={int(np.sum(fused_decision_mask))}",
            f"fused_decision negative_pixels={int(fused_decision_mask.size - np.sum(fused_decision_mask))}",
            "",
            "Per-map positive pixels after threshold:",
        ]
    )
    summary_lines.extend(f"{name}: {int(np.sum(mask))}" for name, mask in binary_masks.items())

    if metric_row is not None:
        summary_lines.extend(
            [
                "",
                "Metrics against Groundtruth_4:",
            ]
        )
        summary_lines.extend(f"{key}: {value}" for key, value in metric_row.items())

    if rendered_bayes_images:
        summary_lines.extend(["", f"Rendered Bayes images: {len(rendered_bayes_images)}"])

    if rendered_final_mask_images:
        summary_lines.extend(["", f"Rendered final-mask comparison images: {len(rendered_final_mask_images)}"])

    (OUT_DIR / "decision_summary.txt").write_text("\n".join(summary_lines), encoding="utf-8")

    print(f"Saved outputs to: {OUT_DIR}")
    print(f"Final positive pixels: {int(np.sum(fused_decision_mask))}")


if __name__ == "__main__":
    main()
