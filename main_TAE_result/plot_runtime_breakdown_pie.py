from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# ============================================================
# 手动配置区
# 这一部分放的是最常需要人工修改的内容：
# 1. 输入/输出路径
# 2. 图标题
# 3. 阶段名称映射
# 4. 各阶段颜色
# 5. 绘图样式参数
# ============================================================
ROOT = Path(r"C:\GUO\jinqiang\Paper2\New_plot\result_new")
INPUT_CSV = ROOT / "runtime_breakdown_measured.csv"
OUTPUT_PNG = ROOT / "runtime_breakdown_measured_pie.png"

# 如果论文里需要特定标题，直接改这里。
TITLE = "Runtime Breakdown"
SHOW_TOTAL_IN_TITLE = True

# 如果论文里阶段顺序需要固定，直接改这里。
# 名称必须与 CSV 第一列 Stage 中的原始名称一致。
STAGE_ORDER = [
    "Data reading",
    "Preprocessing",
    "Feature extraction",
    "TAE training/inference",
    "Fusion",
]

# 把 CSV 里的原始名称映射成论文里展示的名称。
# 如果你的论文里用中文，也可以直接把右边改成中文。
STAGE_DISPLAY_NAMES = {
    "Data reading": "Data loading",
    "Preprocessing": "Pre-processing",
    "Feature extraction": "Feature extraction",
    "TAE training/inference": "TAE training and inference",
    "Fusion": "Bayesian fusion",
}

# 每个阶段对应的颜色。
# 如果想和论文配色统一，优先改这里。
STAGE_COLORS = {
    "Data reading": "#0B5D6E",
    "Preprocessing": "#1E8C93",
    "Feature extraction": "#8CC9B5",
    "TAE training/inference": "#E6D39A",
    "Fusion": "#E79A00",
}

# 画图参数。
FIGSIZE = (12, 9)
DPI = 220
START_ANGLE = 90
COUNTERCLOCK = False
LABEL_DISTANCE = 1.15
PCT_DISTANCE = 0.75
LABEL_FONTSIZE = 26
VALUE_FONTSIZE = 26
TITLE_FONTSIZE = 26
WEDGE_LINEWIDTH = 1.2
WEDGE_EDGECOLOR = "white"

# 如果论文排版需要特定字体，可以在这里改。
# 例如：FONT_FAMILY = "Times New Roman"
FONT_FAMILY = "Times New Roman"


def format_autopct(total_seconds: float):
    """返回饼图扇区文字格式：百分比 + 对应秒数。"""

    def _autopct(pct: float) -> str:
        seconds = total_seconds * pct / 100.0
        return f"{pct:.1f}%\n({seconds:.2f}s)"

    return _autopct


def load_runtime_data(csv_path: Path) -> pd.DataFrame:
    """读取并清洗运行时间分解表。"""
    if not csv_path.exists():
        raise FileNotFoundError(f"Input file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    required_columns = {"Stage", "Seconds"}
    if not required_columns.issubset(df.columns):
        raise ValueError(f"CSV must contain columns: {sorted(required_columns)}")

    df = df.copy()
    df["Seconds"] = pd.to_numeric(df["Seconds"], errors="coerce")
    df = df.dropna(subset=["Seconds"])
    df = df[df["Seconds"] > 0]

    if df.empty:
        raise ValueError("No positive runtime values found in the CSV.")

    return df


def apply_stage_order(df: pd.DataFrame) -> pd.DataFrame:
    """按手动设定的阶段顺序重排数据。"""
    order_index = {name: idx for idx, name in enumerate(STAGE_ORDER)}
    df = df.copy()
    df["sort_key"] = df["Stage"].map(lambda name: order_index.get(name, len(order_index)))
    df = df.sort_values(["sort_key", "Stage"]).drop(columns="sort_key")
    return df


def prepare_display_columns(df: pd.DataFrame) -> pd.DataFrame:
    """生成展示名称和颜色列，便于统一绘图。"""
    df = apply_stage_order(df)
    df = df.copy()
    df["DisplayName"] = df["Stage"].map(lambda name: STAGE_DISPLAY_NAMES.get(name, name))
    df["Color"] = df["Stage"].map(lambda name: STAGE_COLORS.get(name, "#999999"))
    return df


def build_title(total_seconds: float) -> str:
    """根据配置生成标题。"""
    if SHOW_TOTAL_IN_TITLE:
        return f"{TITLE} (total = {total_seconds:.2f}s)"
    return TITLE


def main() -> None:
    if FONT_FAMILY:
        plt.rcParams["font.family"] = FONT_FAMILY

    df = load_runtime_data(INPUT_CSV)
    df = prepare_display_columns(df)
    total_seconds = float(df["Seconds"].sum())

    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    wedges, texts, autotexts = ax.pie(
        df["Seconds"],
        labels=df["DisplayName"],
        colors=df["Color"],
        autopct=format_autopct(total_seconds),
        startangle=START_ANGLE,
        counterclock=COUNTERCLOCK,
        wedgeprops={"edgecolor": WEDGE_EDGECOLOR, "linewidth": WEDGE_LINEWIDTH},
        textprops={"fontsize": LABEL_FONTSIZE},
        pctdistance=PCT_DISTANCE,
        labeldistance=LABEL_DISTANCE,
    )

    for autotext in autotexts:
        autotext.set_color("white")
        autotext.set_fontsize(VALUE_FONTSIZE)
        autotext.set_weight("bold")

    for text in texts:
        text.set_fontsize(LABEL_FONTSIZE)
        text.set_weight("bold")

    ax.set_title(build_title(total_seconds), fontsize=TITLE_FONTSIZE)
    ax.axis("equal")

    fig.tight_layout()
    fig.savefig(OUTPUT_PNG, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"Saved pie chart to: {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
