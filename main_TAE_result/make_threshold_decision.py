from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap, to_rgb
from matplotlib.patches import Patch
from scipy.ndimage import binary_dilation, label, zoom


THRESHOLD = 0.5
ROOT = Path(r"C:\GUO\jinqiang\Paper2\New_plot\result_new")
GT_PATH = ROOT / "Groundtruth_4.xlsx"
BAYES_RESULTS_ROOT = ROOT / "bayes_results" / "sdcq_2" / "exp_col_list"
INPUTS = {
    "gpr_ae_mean": ROOT / "sdcq_2" / "exp_col_list" / "GPR_AE" / "GPR_AE_Mean.npy",
    "knock_ae_ch1": ROOT / "sdcq_2" / "exp_col_list" / "Knock_AE_PerSensor" / "AE_Prob_Ch1.npy",
    "knock_ae_ch2": ROOT / "sdcq_2" / "exp_col_list" / "Knock_AE_PerSensor" / "AE_Prob_Ch2.npy",
    "knock_ae_ch3": ROOT / "sdcq_2" / "exp_col_list" / "Knock_AE_PerSensor" / "AE_Prob_Ch3.npy",
}
OUT_DIR = ROOT / "plot"

# ============================================================
# Manually editable plotting configuration
# Change these values when you want to adjust colors or layout.
# ============================================================
PLOT_COLORS = {
    "panel_background": "#F2F4F7",
    "axis_spine": "#5B6B7A",
    "axis_tick": "#4E5D6C",
    "metric_box_background": "#1B1F24",
    "metric_box_text": "#FFFFFF",
    "probability_cmap": "viridis",
    "groundtruth_cmap": "cividis",
    "detection_positive": "#0072B2",
    "groundtruth_positive": "#D55E00",
    "tp": "#009E73",
    "fp": "#E69F00",
    "fn": "#CC79A7",
    "outline": "#3A4757",
}

PLOT_LAYOUT = {
    "heatmap_figsize": (12, 5),
    "heatmap_dpi": 200,
    "compare_figsize": (12, 4.2),
    "compare_dpi": 220,
    "single_mask_figsize": (5, 4),
    "single_mask_dpi": 220,
    "compare_adjust": {"left": 0.055, "right": 0.985, "top": 0.88, "bottom": 0.28, "wspace": 0.16},
    "single_mask_adjust": {"left": 0.14, "right": 0.98, "top": 0.88, "bottom": 0.16},
    "legend_anchor_y": -0.20,
    "metric_box_fontsize": 8.8,
    "outline_linewidth": 1.0,
}

PIXEL_RENDER = {
    "origin": "upper",
    "aspect": "auto",
    "interpolation": "nearest",
    "resample": False,
}


@dataclass(frozen=True)
class CompareTitles:
    detection: str = "Binary detection"
    ground_truth: str = "Ground truth"
    agreement: str = "Pixel-wise agreement"


COMPARE_TITLES = CompareTitles()


def save_array_csv(array: np.ndarray, path: Path, fmt: str) -> None:
    np.savetxt(path, array, fmt=fmt, delimiter=",")


def render_pixel_image(ax: plt.Axes, array: np.ndarray, **kwargs):
    render_kwargs = dict(PIXEL_RENDER)
    render_kwargs.update(kwargs)
    return ax.imshow(array, **render_kwargs)


def make_binary_cmap(positive_color: str) -> ListedColormap:
    return ListedColormap([PLOT_COLORS["panel_background"], positive_color])


def make_agreement_rgb(mask: np.ndarray, gt: np.ndarray) -> np.ndarray:
    agreement = np.ones((gt.shape[0], gt.shape[1], 3), dtype=np.float32)
    agreement[np.logical_and(mask == 1, gt == 1)] = np.array(to_rgb(PLOT_COLORS["tp"]), dtype=np.float32)
    agreement[np.logical_and(mask == 1, gt == 0)] = np.array(to_rgb(PLOT_COLORS["fp"]), dtype=np.float32)
    agreement[np.logical_and(mask == 0, gt == 1)] = np.array(to_rgb(PLOT_COLORS["fn"]), dtype=np.float32)
    return agreement


def style_axis_frame(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(PLOT_COLORS["axis_spine"])
    ax.spines["bottom"].set_color(PLOT_COLORS["axis_spine"])
    ax.spines["left"].set_linewidth(1.0)
    ax.spines["bottom"].set_linewidth(1.0)
    ax.tick_params(colors=PLOT_COLORS["axis_tick"])


def save_heatmap(array: np.ndarray, path: Path, title: str, cmap: str, vmin: float, vmax: float) -> None:
    fig, ax = plt.subplots(figsize=PLOT_LAYOUT["heatmap_figsize"], dpi=PLOT_LAYOUT["heatmap_dpi"])
    im = render_pixel_image(
        ax,
        array,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_title(title)
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def describe_array(name: str, array: np.ndarray) -> str:
    return (
        f"{name}: shape={array.shape}, dtype={array.dtype}, "
        f"min={float(np.min(array)):.6f}, max={float(np.max(array)):.6f}, "
        f"mean={float(np.mean(array)):.6f}"
    )


def load_and_check_gt(gt_path: Path, detected_shape: tuple[int, int]) -> np.ndarray | None:
    if not gt_path.exists():
        return None

    gt = pd.read_excel(gt_path, header=None).values
    gt = (gt > 0).astype(np.uint8)

    if gt.shape != detected_shape:
        gt = zoom(gt, (detected_shape[0] / gt.shape[0], detected_shape[1] / gt.shape[1]), order=0)

    return gt.astype(np.uint8)


def evaluate_metrics(mask: np.ndarray, gt: np.ndarray) -> dict[str, float | int]:
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
    gt_lab, gt_num = label(gt_mask)
    pr_lab, pr_num = label(pred_mask)

    def iou(a: np.ndarray, b: np.ndarray) -> float:
        inter = np.logical_and(a, b).sum()
        union = np.logical_or(a, b).sum()
        return float(inter / (union + 1e-12))

    matched_gt = 0
    matched_pred = set()

    for gi in range(1, gt_num + 1):
        g_obj = gt_lab == gi
        overlapping_preds = []

        for pi in range(1, pr_num + 1):
            p_obj = pr_lab == pi
            if np.logical_and(g_obj, p_obj).any():
                overlapping_preds.append((pi, p_obj))

        if not overlapping_preds:
            continue

        p_union = np.zeros_like(g_obj, dtype=bool)
        for pi, p_obj in overlapping_preds:
            p_union |= p_obj

        if iou(g_obj, p_union) >= iou_thresh:
            matched_gt += 1
            for pi, _ in overlapping_preds:
                matched_pred.add(pi)

    return {
        "Object_Recall": float(matched_gt / (gt_num + 1e-12)),
        "GT_Objects": int(gt_num),
        "Pred_Objects": int(pr_num),
        "FP_Objects": int(pr_num - len(matched_pred)),
        "Matched_Objects": int(matched_gt),
    }


def relaxed_iou(pred_mask: np.ndarray, gt_mask: np.ndarray, radius: int) -> float:
    struct = np.ones((2 * radius + 1, 2 * radius + 1), dtype=bool)
    gt_dilated = binary_dilation(gt_mask, structure=struct)
    inter = np.logical_and(pred_mask, gt_dilated).sum()
    union = np.logical_or(pred_mask, gt_dilated).sum()
    return float(inter / (union + 1e-12))


def evaluate_all_metrics(mask: np.ndarray, gt: np.ndarray, prefix: str) -> dict[str, float | int | str]:
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


def _set_scan_axes(ax: plt.Axes, height: int, width: int) -> None:
    if width == 320:
        x_labels = np.array([1, 16, 32, 48, 64])
        x_pos = (x_labels - 1) * (width - 1) / (64 - 1)
    else:
        x_labels = np.linspace(1, width, 5, dtype=int)
        x_pos = x_labels - 1

    if height == 145:
        y_labels = np.array([1, 29, 58, 87, 116, 145])
    else:
        y_labels = np.linspace(1, height, 6, dtype=int)
    y_pos = y_labels - 1

    ax.set_xticks(x_pos)
    ax.set_xticklabels([str(v) for v in x_labels])
    ax.set_yticks(y_pos)
    ax.set_yticklabels([str(v) for v in y_labels])
    ax.set_xlabel("Measurement column")
    ax.set_ylabel("Scan row")


def _target_coordinate_shape(height: int, width: int) -> tuple[int, int]:
    target_h = 29 if height == 145 else height
    target_w = 64 if width in (64, 320) else width
    return target_h, target_w


def _set_coordinate_ratio(ax: plt.Axes, height: int, width: int) -> None:
    target_h, target_w = _target_coordinate_shape(height, width)
    data_ratio = height / width
    target_ratio = target_h / target_w
    ax.set_aspect(target_ratio / data_ratio)


def save_compare_with_gt(mask: np.ndarray, gt: np.ndarray, metrics: dict[str, float | int], path: Path) -> None:
    h, w = gt.shape

    detection_cmap = make_binary_cmap(PLOT_COLORS["detection_positive"])
    gt_cmap = make_binary_cmap(PLOT_COLORS["groundtruth_positive"])
    agreement = make_agreement_rgb(mask, gt)

    fig, axs = plt.subplots(1, 3, figsize=PLOT_LAYOUT["compare_figsize"], dpi=PLOT_LAYOUT["compare_dpi"])
    fig.patch.set_facecolor("white")

    render_pixel_image(
        axs[0],
        mask,
        cmap=detection_cmap,
        vmin=0,
        vmax=1,
    )
    axs[0].contour(
        gt.astype(float),
        levels=[0.5],
        colors=PLOT_COLORS["outline"],
        linewidths=PLOT_LAYOUT["outline_linewidth"],
        linestyles="dotted",
    )
    axs[0].set_title(COMPARE_TITLES.detection)
    metric_text = (
        f"IoU={metrics['IoU']:.3f}\n"
        f"F1={metrics['F1']:.3f}\n"
        f"Precision={metrics['Precision']:.3f}\n"
        f"Recall={metrics['Recall']:.3f}"
    )
    axs[0].text(
        0.03,
        0.97,
        metric_text,
        transform=axs[0].transAxes,
        va="top",
        ha="left",
        fontsize=PLOT_LAYOUT["metric_box_fontsize"],
        color=PLOT_COLORS["metric_box_text"],
        bbox={"boxstyle": "round,pad=0.25", "facecolor": PLOT_COLORS["metric_box_background"], "alpha": 0.72, "edgecolor": "none"},
    )

    render_pixel_image(
        axs[1],
        gt,
        cmap=gt_cmap,
        vmin=0,
        vmax=1,
    )
    axs[1].set_title(COMPARE_TITLES.ground_truth)

    render_pixel_image(
        axs[2],
        agreement,
    )
    axs[2].set_title(COMPARE_TITLES.agreement)
    legend_handles = [
        Patch(facecolor=PLOT_COLORS["tp"], edgecolor=PLOT_COLORS["tp"], label="TP"),
        Patch(facecolor=PLOT_COLORS["fp"], edgecolor=PLOT_COLORS["fp"], label="FP"),
        Patch(facecolor=PLOT_COLORS["fn"], edgecolor=PLOT_COLORS["fn"], label="FN"),
    ]
    axs[2].legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, PLOT_LAYOUT["legend_anchor_y"]),
        ncol=3,
        frameon=False,
        handlelength=1.0,
        columnspacing=1.2,
    )

    for ax in axs:
        _set_scan_axes(ax, h, w)
        _set_coordinate_ratio(ax, h, w)
        style_axis_frame(ax)

    fig.subplots_adjust(**PLOT_LAYOUT["compare_adjust"])
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_mask_only_figure(mask: np.ndarray, title: str, path: Path) -> None:
    detection_cmap = make_binary_cmap(PLOT_COLORS["detection_positive"])

    fig, ax = plt.subplots(1, 1, figsize=PLOT_LAYOUT["single_mask_figsize"], dpi=PLOT_LAYOUT["single_mask_dpi"])
    fig.patch.set_facecolor("white")
    render_pixel_image(
        ax,
        mask,
        cmap=detection_cmap,
        vmin=0,
        vmax=1,
    )
    ax.set_title(title)
    _set_scan_axes(ax, mask.shape[0], mask.shape[1])
    _set_coordinate_ratio(ax, mask.shape[0], mask.shape[1])
    style_axis_frame(ax)
    fig.subplots_adjust(**PLOT_LAYOUT["single_mask_adjust"])
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def batch_render_bayes_mask_figures(root_dir: Path, gt_path: Path) -> list[str]:
    outputs: list[str] = []
    mask_files = sorted(root_dir.rglob("*_Mask.csv"))
    if not mask_files:
        return outputs

    for mask_path in mask_files:
        tag = mask_path.stem.removesuffix("_Mask")
        out_dir = mask_path.parent
        mask = np.loadtxt(mask_path).astype(np.uint8)
        save_mask_only_figure(mask, f"{tag} mask", out_dir / f"{tag}_Mask.png")
        outputs.append(str(out_dir / f"{tag}_Mask.png"))

        gt = load_and_check_gt(gt_path, mask.shape)
        if gt is None:
            continue

        metrics = evaluate_metrics(mask, gt)
        save_compare_with_gt(
            mask,
            gt,
            metrics,
            out_dir / f"{tag}_Compare_GT.png",
        )
        outputs.append(str(out_dir / f"{tag}_Compare_GT.png"))

    return outputs


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    prob_maps = {}
    shape = None
    for name, path in INPUTS.items():
        array = np.asarray(np.load(path), dtype=np.float32)
        if shape is None:
            shape = array.shape
        elif array.shape != shape:
            raise ValueError(f"Shape mismatch: expected {shape}, got {array.shape} for {path}")
        prob_maps[name] = np.clip(array, 0.0, 1.0)

    binary_masks = {
        name: (array > THRESHOLD).astype(np.uint8)
        for name, array in prob_maps.items()
    }

    stacked_probs = np.stack([prob_maps[name] for name in INPUTS], axis=2)
    fused_mean_prob = np.mean(stacked_probs, axis=2).astype(np.float32)
    fused_decision = (fused_mean_prob > THRESHOLD).astype(np.uint8)
    positive_votes = np.sum(stacked_probs > THRESHOLD, axis=2).astype(np.uint8)

    for name, array in prob_maps.items():
        np.save(OUT_DIR / f"{name}_prob.npy", array)
        save_array_csv(array, OUT_DIR / f"{name}_prob.csv", fmt="%.6f")
        save_heatmap(
            array,
            OUT_DIR / f"{name}_prob.png",
            title=f"{name} probability",
            cmap=PLOT_COLORS["probability_cmap"],
            vmin=0.0,
            vmax=1.0,
        )

    for name, mask in binary_masks.items():
        np.save(OUT_DIR / f"{name}_mask_thr_0p5.npy", mask)
        save_array_csv(mask, OUT_DIR / f"{name}_mask_thr_0p5.csv", fmt="%d")
        save_heatmap(
            mask,
            OUT_DIR / f"{name}_mask_thr_0p5.png",
            title=f"{name} decision mask (thr=0.5)",
            cmap="gray",
            vmin=0.0,
            vmax=1.0,
        )

    np.save(OUT_DIR / "fused_mean_prob.npy", fused_mean_prob)
    save_array_csv(fused_mean_prob, OUT_DIR / "fused_mean_prob.csv", fmt="%.6f")
    save_heatmap(
        fused_mean_prob,
        OUT_DIR / "fused_mean_prob.png",
        title="Fused mean probability",
        cmap=PLOT_COLORS["probability_cmap"],
        vmin=0.0,
        vmax=1.0,
    )

    np.save(OUT_DIR / "fused_decision_mask_thr_0p5.npy", fused_decision)
    save_array_csv(fused_decision, OUT_DIR / "fused_decision_mask_thr_0p5.csv", fmt="%d")
    save_heatmap(
        fused_decision,
        OUT_DIR / "fused_decision_mask_thr_0p5.png",
        title="Fused decision mask (thr=0.5)",
        cmap="gray",
        vmin=0.0,
        vmax=1.0,
    )

    np.save(OUT_DIR / "positive_vote_count.npy", positive_votes)
    save_array_csv(positive_votes, OUT_DIR / "positive_vote_count.csv", fmt="%d")
    save_heatmap(
        positive_votes,
        OUT_DIR / "positive_vote_count.png",
        title="Positive vote count (>0.5)",
        cmap="magma",
        vmin=0.0,
        vmax=float(len(INPUTS)),
    )

    gt = load_and_check_gt(GT_PATH, fused_decision.shape)
    metric_row = None
    bayes_rendered = []
    if gt is not None:
        np.save(OUT_DIR / "groundtruth_4_resized.npy", gt)
        save_array_csv(gt, OUT_DIR / "groundtruth_4_resized.csv", fmt="%d")
        save_heatmap(
            gt,
            OUT_DIR / "groundtruth_4_resized.png",
            title="Groundtruth_4 resized to decision grid",
            cmap=PLOT_COLORS["groundtruth_cmap"],
            vmin=0.0,
            vmax=1.0,
        )

        metric_row = evaluate_all_metrics(fused_decision.astype(np.uint8), gt.astype(np.uint8), prefix="AverageFusion_4Maps")
        pd.DataFrame([metric_row]).to_csv(OUT_DIR / "average_fusion_metrics_with_gt.csv", index=False)
        save_compare_with_gt(
            fused_decision.astype(np.uint8),
            gt.astype(np.uint8),
            metric_row,
            OUT_DIR / "average_fusion_vs_groundtruth4.png",
        )

    if BAYES_RESULTS_ROOT.exists():
        bayes_rendered = batch_render_bayes_mask_figures(BAYES_RESULTS_ROOT, GT_PATH)

    summary_lines = [
        "Threshold decision summary",
        f"Threshold: {THRESHOLD}",
        "Rule: mean the 4 probability maps, then apply decision mask = (mean_prob > 0.5).",
        "",
        "Inputs:",
    ]
    summary_lines.extend(describe_array(name, array) for name, array in prob_maps.items())
    summary_lines.extend(
        [
            "",
            describe_array("fused_mean_prob", fused_mean_prob),
            f"fused_decision positive_pixels={int(np.sum(fused_decision))}",
            f"fused_decision negative_pixels={int(fused_decision.size - np.sum(fused_decision))}",
            "",
            "Per-map positive pixels after threshold:",
        ]
    )
    summary_lines.extend(
        f"{name}: {int(np.sum(mask))}" for name, mask in binary_masks.items()
    )

    if metric_row is not None:
        summary_lines.extend(
            [
                "",
                "Metrics against Groundtruth_4 (same definitions as evaluate_all_metrics in the main code):",
            ]
        )
        summary_lines.extend(f"{key}: {value}" for key, value in metric_row.items())

    if bayes_rendered:
        summary_lines.extend(
            [
                "",
                f"Rendered Bayes mask figures: {len(bayes_rendered)}",
            ]
        )

    (OUT_DIR / "decision_summary.txt").write_text("\n".join(summary_lines), encoding="utf-8")

    print(f"Saved outputs to: {OUT_DIR}")
    print(f"Final positive pixels: {int(np.sum(fused_decision))}")


if __name__ == "__main__":
    main()
