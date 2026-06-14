from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "plot_2"
OUT_DIR.mkdir(exist_ok=True)
OUT_PATH = OUT_DIR / "tae_flowchart_from_manuscript.png"

WIDTH = 2400
HEIGHT = 1700

BG = "#F7FAFC"
TITLE = "#12324A"
TEXT = "#1F2937"
MUTED = "#5B6472"
PANEL_BORDER = "#D5DCE5"
SHADOW = "#DDE6EF"
HEADER_BLUE = "#D8ECFF"
HEADER_ORANGE = "#FFE5CF"
BLUE_FILL = "#EEF6FF"
BLUE_EDGE = "#2B6CB0"
ORANGE_FILL = "#FFF5EB"
ORANGE_EDGE = "#C05621"
GREEN_FILL = "#EAFBF5"
GREEN_EDGE = "#2F855A"
GREY_FILL = "#F8FAFC"
GREY_EDGE = "#718096"
ARROW_BLUE = "#316DAA"
ARROW_ORANGE = "#C7662D"
BADGE_FILL = "#E6FFFA"
BADGE_EDGE = "#319795"


def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = []
    if bold:
        candidates.extend(
            [
                Path("C:/Windows/Fonts/arialbd.ttf"),
                Path("C:/Windows/Fonts/segoeuib.ttf"),
                Path("C:/Windows/Fonts/calibrib.ttf"),
            ]
        )
    else:
        candidates.extend(
            [
                Path("C:/Windows/Fonts/arial.ttf"),
                Path("C:/Windows/Fonts/segoeui.ttf"),
                Path("C:/Windows/Fonts/calibri.ttf"),
            ]
        )

    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size=size)

    return ImageFont.load_default()


FONT_TITLE = load_font(44, bold=True)
FONT_SUBTITLE = load_font(24, bold=False)
FONT_PANEL = load_font(28, bold=True)
FONT_BADGE = load_font(22, bold=True)
FONT_BOX_TITLE = load_font(24, bold=True)
FONT_BOX = load_font(22, bold=False)
FONT_FOOTER = load_font(20, bold=False)


def wrap_text(draw: ImageDraw.ImageDraw, text: str, font, max_width: int):
    words = text.split()
    lines = []
    current = ""
    for word in words:
        candidate = word if not current else f"{current} {word}"
        if draw.textlength(candidate, font=font) <= max_width:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def draw_multiline_centered(draw, x0, y0, x1, y1, title, body, fill, edge):
    draw.rounded_rectangle(
        [x0, y0, x1, y1],
        radius=26,
        fill=fill,
        outline=edge,
        width=4,
    )

    center_x = (x0 + x1) / 2
    available = int(x1 - x0 - 64)
    lines = []
    if title:
        lines.append(("title", title))
    if body:
        for line in wrap_text(draw, body, FONT_BOX, available):
            lines.append(("body", line))

    title_h = FONT_BOX_TITLE.size + 6
    body_h = FONT_BOX.size + 5
    total_h = 0
    for kind, _ in lines:
        total_h += title_h if kind == "title" else body_h
    total_h -= 5

    y = y0 + (y1 - y0 - total_h) / 2 - 2
    for kind, line in lines:
        font = FONT_BOX_TITLE if kind == "title" else FONT_BOX
        color = edge if kind == "title" else TEXT
        bbox = draw.textbbox((0, 0), line, font=font)
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        draw.text((center_x - w / 2, y), line, font=font, fill=color)
        y += h + (8 if kind == "title" else 5)


def draw_arrow(draw, x, y0, y1, color):
    draw.line((x, y0, x, y1), fill=color, width=5)
    head = 12
    draw.polygon(
        [
            (x, y1),
            (x - head, y1 - 18),
            (x + head, y1 - 18),
        ],
        fill=color,
    )


def draw_panel(draw, x0, y0, x1, y1, header_fill, title_text):
    shadow_offset = 10
    draw.rounded_rectangle(
        [x0 + shadow_offset, y0 + shadow_offset, x1 + shadow_offset, y1 + shadow_offset],
        radius=36,
        fill=SHADOW,
    )
    draw.rounded_rectangle(
        [x0, y0, x1, y1],
        radius=36,
        fill="white",
        outline=PANEL_BORDER,
        width=4,
    )
    draw.rounded_rectangle(
        [x0 + 28, y0 + 24, x0 + 380, y0 + 88],
        radius=24,
        fill=header_fill,
        outline="white",
    )
    draw.text((x0 + 56, y0 + 39), title_text, font=FONT_PANEL, fill=TITLE)


def main():
    image = Image.new("RGB", (WIDTH, HEIGHT), BG)
    draw = ImageDraw.Draw(image)

    draw.text((110, 60), "Transformer-based Autoencoder Flowchart", font=FONT_TITLE, fill=TITLE)
    draw.text(
        (112, 116),
        "Based on Section 4.2 and Fig. 9 of the manuscript. One TAE is trained per modality (GPR or vibro-acoustic).",
        font=FONT_SUBTITLE,
        fill=MUTED,
    )

    banner = (110, 170, WIDTH - 110, 290)
    draw.rounded_rectangle(banner, radius=30, fill=GREY_FILL, outline=GREY_EDGE, width=3)
    banner_title = "Shared input context"
    banner_body = (
        "Raw GPR and vibro-acoustic signals are first converted into physics-informed feature tensors. "
        "Each measurement point is then reshaped into one N-dimensional feature vector for point-wise TAE modeling."
    )
    draw.text((148, 196), banner_title, font=FONT_BOX_TITLE, fill=GREY_EDGE)
    banner_lines = wrap_text(draw, banner_body, FONT_BOX, WIDTH - 380)
    y = 232
    for line in banner_lines:
        draw.text((148, y), line, font=FONT_BOX, fill=TEXT)
        y += 28

    badge = (WIDTH - 680, 196, WIDTH - 150, 252)
    draw.rounded_rectangle(badge, radius=22, fill=BADGE_FILL, outline=BADGE_EDGE, width=3)
    draw.text((WIDTH - 650, 212), "Point-wise modeling, not full-field image input", font=FONT_BADGE, fill=BADGE_EDGE)

    left_panel = (110, 340, 1120, 1540)
    right_panel = (1280, 340, 2290, 1540)
    draw_panel(draw, *left_panel, HEADER_BLUE, "Training phase")
    draw_panel(draw, *right_panel, HEADER_ORANGE, "Inference phase")

    train_boxes = [
        ("Input sample", "Point-wise feature vector x in R^N", GREY_FILL, GREY_EDGE),
        ("Training data", "Use intact samples only to learn the healthy feature distribution", BLUE_FILL, BLUE_EDGE),
        ("Tokenization", "Treat each scalar feature as one token and linearly embed 1 -> 64", BLUE_FILL, BLUE_EDGE),
        ("Position encoding", "Add learnable positional embedding to preserve feature order", BLUE_FILL, BLUE_EDGE),
        ("Transformer encoder", "1 layer, 4 attention heads, FFN dimension = 128", BLUE_FILL, BLUE_EDGE),
        ("Latent compression", "Global average pooling across tokens, then compress to a 28-D bottleneck z", GREEN_FILL, GREEN_EDGE),
        ("Linear expansion", "Expand z back to an N x 64 token sequence", BLUE_FILL, BLUE_EDGE),
        ("Transformer decoder", "1 layer, 4 attention heads, FFN dimension = 128", BLUE_FILL, BLUE_EDGE),
        ("Reconstruction", "Linear output projection gives reconstructed vector x_hat", BLUE_FILL, BLUE_EDGE),
        ("Optimization", "Minimize MSE reconstruction loss between x and x_hat", GREY_FILL, GREY_EDGE),
    ]

    infer_boxes = [
        ("Full dataset", "Apply the trained TAE to all measurement points in the modality-specific feature tensor", GREY_FILL, GREY_EDGE),
        ("Trained TAE", "Reuse the encoder-decoder weights learned from intact data", ORANGE_FILL, ORANGE_EDGE),
        ("Forward pass", "Reconstruct x_hat for each point sample", ORANGE_FILL, ORANGE_EDGE),
        ("Error map", "Compute absolute reconstruction error e = |x - x_hat|", ORANGE_FILL, ORANGE_EDGE),
        ("Normalization", "Use percentile-based normalization for cross-feature comparability", ORANGE_FILL, ORANGE_EDGE),
        ("Probability conversion", "Convert normalized errors into feature-wise anomaly / void probabilities", ORANGE_FILL, ORANGE_EDGE),
        ("Spatial remapping", "Map the probability vectors back to the measurement grid", ORANGE_FILL, ORANGE_EDGE),
        ("Output to fusion", "Generate multi-channel probability maps for the subsequent Bayesian fusion stage", GREY_FILL, GREY_EDGE),
    ]

    def layout_boxes(panel, boxes, arrow_color):
        px0, py0, px1, py1 = panel
        box_w = px1 - px0 - 120
        box_h = 84
        gap = 22
        start_x = px0 + 60
        start_y = py0 + 110
        centers = []
        prev_bottom = None

        for idx, (title, body, fill, edge) in enumerate(boxes):
            x0 = start_x
            x1 = start_x + box_w
            y0 = start_y + idx * (box_h + gap)
            y1 = y0 + box_h
            draw_multiline_centered(draw, x0, y0, x1, y1, title, body, fill, edge)
            centers.append(((x0 + x1) / 2, y0, y1))
            if prev_bottom is not None:
                draw_arrow(draw, int((x0 + x1) / 2), prev_bottom + 3, y0 - 6, arrow_color)
            prev_bottom = y1
        return centers

    train_centers = layout_boxes(left_panel, train_boxes, ARROW_BLUE)
    infer_centers = layout_boxes(right_panel, infer_boxes, ARROW_ORANGE)

    left_anchor_x = int((left_panel[0] + left_panel[2]) / 2)
    right_anchor_x = int((right_panel[0] + right_panel[2]) / 2)
    draw.line((left_anchor_x, banner[3], left_anchor_x, train_centers[0][1] - 10), fill=GREY_EDGE, width=4)
    draw.polygon(
        [
            (left_anchor_x, train_centers[0][1] - 10),
            (left_anchor_x - 10, train_centers[0][1] - 26),
            (left_anchor_x + 10, train_centers[0][1] - 26),
        ],
        fill=GREY_EDGE,
    )
    draw.line((right_anchor_x, banner[3], right_anchor_x, infer_centers[0][1] - 10), fill=GREY_EDGE, width=4)
    draw.polygon(
        [
            (right_anchor_x, infer_centers[0][1] - 10),
            (right_anchor_x - 10, infer_centers[0][1] - 26),
            (right_anchor_x + 10, infer_centers[0][1] - 26),
        ],
        fill=GREY_EDGE,
    )

    mid_y = 520
    draw.line(
        (left_panel[2] - 70, mid_y, right_panel[0] + 70, mid_y),
        fill="#A0AEC0",
        width=4,
    )
    draw.polygon(
        [
            (right_panel[0] + 70, mid_y),
            (right_panel[0] + 54, mid_y - 10),
            (right_panel[0] + 54, mid_y + 10),
        ],
        fill="#A0AEC0",
    )
    label = "trained weights transferred to inference"
    label_bbox = draw.textbbox((0, 0), label, font=FONT_FOOTER)
    label_w = label_bbox[2] - label_bbox[0]
    draw.rounded_rectangle(
        [
            (WIDTH - label_w) / 2 - 18,
            mid_y - 28,
            (WIDTH + label_w) / 2 + 18,
            mid_y + 18,
        ],
        radius=18,
        fill="white",
        outline="#CBD5E0",
        width=2,
    )
    draw.text(((WIDTH - label_w) / 2, mid_y - 18), label, font=FONT_FOOTER, fill=MUTED)

    footer = (
        "Key settings from the manuscript and implementation: embedding dim = 64, attention heads = 4, "
        "FFN dim = 128, bottleneck = 28, batch size = 64, epochs = 80."
    )
    footer_lines = wrap_text(draw, footer, FONT_FOOTER, WIDTH - 220)
    fy = 1590
    for line in footer_lines:
        bbox = draw.textbbox((0, 0), line, font=FONT_FOOTER)
        w = bbox[2] - bbox[0]
        draw.text(((WIDTH - w) / 2, fy), line, font=FONT_FOOTER, fill=MUTED)
        fy += 24

    image.save(OUT_PATH, quality=95)
    print(f"saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
