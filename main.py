import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from env import OUTPUT_FOLDER
from src.pipeline.edge_detector import (
    CannyDetector,
    DoGEdgeDetector,
    FDoGDetector,
    SobelDetector,
    XDoGDetector,
)
from src.pipeline.orchestrator import EdgePipeline
from src.pipeline.postprocessing import Dilate, Pixelate
from src.pipeline.preprocess import BilateralFilter, ETFComputer, GaussianBlur

# ===== PIPELINE =====


def build_pipeline(config):
    pre_map = {
        "gaussian": GaussianBlur(),
        "bilateral": BilateralFilter(),
        "etf": ETFComputer(),
    }
    det_map = {
        "canny": CannyDetector(),
        "sobel": SobelDetector(),
        "xdog": XDoGDetector(),
        "fdog": FDoGDetector(),
        "dog": DoGEdgeDetector(),
    }
    post_map = {"pixelate": Pixelate(), "dilate": Dilate()}
    pre = [pre_map[n] for n in config["pre"]]
    det = det_map[config["detector"]]
    post = [post_map[n] for n in config["post"]]
    return EdgePipeline(pre, det, post)


def config_to_name(cfg):
    parts = []
    if cfg["pre"]:
        parts.append("+".join(cfg["pre"]))
    parts.append(cfg["detector"])
    if cfg["post"]:
        parts.append("+".join(cfg["post"]))
    return "_".join(parts)


# ===== CONTOURS =====


def simplify_contours(contours, hierarchy, epsilon=2.0, filters=None):
    if filters is None:
        filters = {}

    min_vertices = filters.get("min_vertices", 3)
    require_convex = filters.get("require_convex", True)
    require_quadrilateral = filters.get("require_quadrilateral", True)

    enable_min_area = filters.get("enable_min_area", False)
    min_area = filters.get("min_area", 10)

    enable_min_aspect_ratio = filters.get("enable_min_aspect_ratio", False)
    min_aspect_ratio = filters.get("min_aspect_ratio", 0.1)

    simplified = []
    if hierarchy is None or len(contours) == 0:
        return simplified

    hierarchy = hierarchy[0]
    for i, c in enumerate(contours):
        approx = cv2.approxPolyDP(c, epsilon, True)
        if len(approx) < min_vertices:  # remove degenerate contours
            continue

        parent = hierarchy[i][3]

        if require_convex and not cv2.isContourConvex(approx):  # remove non-convex contours
            continue

        if require_quadrilateral and len(approx) != 4:  # keep only quadrilaterals
            continue

        if enable_min_area and cv2.contourArea(approx) < min_area:  # remove very small contours
            continue

        if enable_min_aspect_ratio:  # remove very elongated contours
            rect = cv2.minAreaRect(approx)
            w, h = rect[1]
            if max(w, h) == 0:
                continue
            if min(w, h) / max(w, h) < min_aspect_ratio:
                continue

        simplified.append(
            {"contour": approx, "vertex_count": len(approx), "filled": parent != -1}
        )
    return simplified


def draw_contours(edges, contours):
    canvas = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    for c in contours:
        if c["filled"]:
            cv2.drawContours(canvas, [c["contour"]], -1, (0, 200, 0), -1)

    for c in contours:
        if not c["filled"]:
            cv2.drawContours(canvas, [c["contour"]], -1, (0, 0, 255), 1)

    return canvas


# ===== HISTOGRAM =====


def save_vertex_histogram(contours, output_path, title):
    sns.set(style="whitegrid")

    bins = list(range(0, 11)) + list(range(11, 105, 5))
    filled_counts = []
    open_counts = []

    for c in contours:
        vc = c["vertex_count"]
        if c["filled"]:
            filled_counts.append(vc)
        else:
            open_counts.append(vc)

    hist_filled, bin_edges = np.histogram(filled_counts, bins=bins)
    hist_open, _ = np.histogram(open_counts, bins=bins)

    labels = []
    for i in range(len(bin_edges) - 1):
        s = bin_edges[i]
        e = bin_edges[i + 1] - 1
        labels.append(str(s) if s == e else f"{s}-{e}")

    x = np.arange(len(labels))

    plt.figure(figsize=(12, 5))
    plt.bar(x, hist_open, label="Open", color="#fdae6b")
    plt.bar(x, hist_filled, bottom=hist_open, label="Filled", color="#2ca25f")
    plt.xlabel("Vertex Count")
    plt.ylabel("Number of Contours")
    plt.title(title)
    plt.xticks(x, labels, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def save_vertex_histogram_csv(contours, output_path):
    bins = list(range(0, 11)) + list(range(11, 105, 5))
    filled_counts = []
    open_counts = []

    for c in contours:
        vc = c["vertex_count"]

        if c["filled"]:
            filled_counts.append(vc)
        else:
            open_counts.append(vc)

    hist_filled, bin_edges = np.histogram(filled_counts, bins=bins)
    hist_open, _ = np.histogram(open_counts, bins=bins)
    labels = []

    for i in range(len(bin_edges) - 1):
        s = bin_edges[i]
        e = bin_edges[i + 1] - 1

        if s == e:
            labels.append(str(s))
        else:
            labels.append(f"{s}-{e}")

    total = hist_open + hist_filled
    df = pd.DataFrame(
        {
            "vertex_bin": labels,
            "open_count": hist_open,
            "filled_count": hist_filled,
            "total_count": total,
        }
    )
    df.to_csv(output_path, index=False)

# ===== EDGE =====
def extract_edges(contours):
    edges = []
    edges_directions = []

    for c in contours:
        pts = c["contour"].reshape(-1, 2)
        for i in range(4):
            p1 = pts[i]
            p2 = pts[(i + 1) % 4]
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            edges.append((p1, p2))
            edges_directions.append((dx, dy))
    return edges, edges_directions

def plot_edge_direction_on_unit_circle(
    edges_directions,
    output_path,
    title,
    num_bins=36,
    top_k=4,
):
    if not edges_directions:
        print(f"No edge directions for {title}")
        return

    # Convert to array and normalize
    dirs = np.array(edges_directions, dtype=float)
    norms = np.linalg.norm(dirs, axis=1)
    dirs = dirs[norms > 0] / norms[norms > 0][:, None]

    # Compute angles (mod π → opposite directions merged)
    angles = np.arctan2(dirs[:, 1], dirs[:, 0]) % np.pi
    degrees = np.degrees(angles)

    # ---- Polar histogram ----
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="polar")

    counts, bin_edges, _ = ax.hist(
        angles,
        bins=num_bins,
        color="#3182bd",
        alpha=0.75,
        edgecolor="black",
        linewidth=0.5,
    )

    ax.set_title(title, pad=15)
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(-1)
    ax.set_thetagrids(range(0, 180, 15))
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)

    # ---- Top direction bins ----
    print(f"\nTop {top_k} directions — {title}:")

    bin_width = 180 / num_bins
    bin_ids = (degrees // bin_width) * bin_width

    direction_counts = {}
    for b in bin_ids:
        key = int(b)
        direction_counts[key] = direction_counts.get(key, 0) + 1

    top_dirs = sorted(direction_counts.items(), key=lambda x: x[1], reverse=True)[
        :top_k
    ]

    for deg, cnt in top_dirs:
        print(f"{deg:3d}°–{int(deg+bin_width):3d}° : {cnt}")


# ===== CONFIG =====

CONTOUR_FILTER_CONFIG = {
    "min_vertices": 3,
    "require_convex": True,
    "require_quadrilateral": True,
    "enable_min_area": False,
    "min_area": 10,
    "enable_min_aspect_ratio": False,
    "min_aspect_ratio": 0.1,
}

configs = [
    {"pre": [], "detector": "canny", "post": []},
    {"pre": ["gaussian"], "detector": "canny", "post": []},
    {"pre": ["bilateral"], "detector": "canny", "post": []},
    # {"pre": [], "detector": "sobel", "post": []},             // Doesn't have sufficient contour amount for analysis
    # {"pre": ["gaussian"], "detector": "sobel", "post": []},   // Doesn't have sufficient contour amount for analysis
    # {"pre": ["bilateral"], "detector": "sobel", "post": []},  // Doesn't have sufficient contour amount for analysis
]

# ===== OUTPUT FOLDERS =====

EDGE_DIR = os.path.join(OUTPUT_FOLDER, "edges")
CONTOUR_DIR = os.path.join(OUTPUT_FOLDER, "contours")
HIST_DIR = os.path.join(OUTPUT_FOLDER, "histograms")
CSV_DIR = os.path.join(OUTPUT_FOLDER, "csv")

os.makedirs(CSV_DIR, exist_ok=True)
os.makedirs(EDGE_DIR, exist_ok=True)
os.makedirs(CONTOUR_DIR, exist_ok=True)
os.makedirs(HIST_DIR, exist_ok=True)

# ===== MAIN =====

img = cv2.imread("data\\house1\\images\\2026-03-23_19.31.05.png")
kernel = np.ones((3, 3), np.uint8)

for cfg in configs:
    pipeline = build_pipeline(cfg)
    edges = pipeline.run(img)
    name = config_to_name(cfg)
    cv2.imwrite(os.path.join(EDGE_DIR, f"{name}.png"), edges)

    edges_clean = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    contours, hierarchy = cv2.findContours(
        edges_clean, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE
    )

    contours = simplify_contours(contours, hierarchy, filters=CONTOUR_FILTER_CONFIG)
    contour_img = draw_contours(edges_clean, contours)
    cv2.imwrite(os.path.join(CONTOUR_DIR, f"{name}.png"), contour_img)

    save_vertex_histogram(
        contours,
        os.path.join(HIST_DIR, f"{name}_vertex_hist.png"),
        title=f"{name} - Vertex Count Histogram",
    )
    save_vertex_histogram_csv(
        contours, os.path.join(CSV_DIR, f"{name}_vertex_hist.csv")
    )
    
    edges_directions = extract_edges(contours)[1]
    plot_edge_direction_on_unit_circle(
        edges_directions,
        os.path.join(HIST_DIR, f"{name}_edge_direction.png"),
        title=f"{name} - Edge Direction Distribution",
    )
