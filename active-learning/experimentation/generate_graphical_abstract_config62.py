#!/usr/bin/env python3
"""
Generate a 660x295 JPG Graphical Abstract (GA) for config 62 using the
final F1 scores per run (Active vs Passive) from the statistical test data.

Output: experimentation/data/graphical_abstract_config62.jpg (< 45 KB)
"""

import os
import pandas as pd
import matplotlib.pyplot as plt


PROJECT_ROOT = "/Users/lucasbraga/Documents/GitHub/active-learning"
DATA_DIR = f"{PROJECT_ROOT}/active-learning/experimentation/data"
ALT_DATA_DIR = f"{PROJECT_ROOT}/active-learning/remove-gridlines"


def find_csv_path() -> str:
    """Return the absolute CSV path for statistical_test_data_config62.csv."""
    candidates = [
        f"{DATA_DIR}/statistical_test_data_config62.csv",
        f"{ALT_DATA_DIR}/statistical_test_data_config62.csv",
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError("statistical_test_data_config62.csv not found in expected locations")


def ensure_dir(path: str) -> None:
    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)


def save_jpg_under_size(fig, out_path: str, width_px: int, height_px: int, max_bytes: int = 45000) -> str:
    """Save figure as JPEG with descending quality until under max_bytes."""
    # Matplotlib size control: inches = pixels / dpi
    dpi = 100
    fig.set_size_inches(width_px / dpi, height_px / dpi)

    qualities = [85, 80, 75, 70, 65, 60, 55, 50, 45, 40, 35, 30]

    ensure_dir(out_path)
    for q in qualities:
        fig.savefig(
            out_path,
            format="jpg",
            dpi=dpi,
            bbox_inches="tight",
            pil_kwargs={
                "quality": q,
                "optimize": True,
                "progressive": True,
                # 2 = 4:2:0 subsampling, good tradeoff for size
                "subsampling": 2,
            },
        )
        size = os.path.getsize(out_path)
        if size <= max_bytes:
            return f"Saved {out_path} at quality {q} with size {size} bytes"
    # If still too large, keep the smallest and report
    size = os.path.getsize(out_path)
    return f"Saved {out_path} at minimum quality with size {size} bytes (exceeds {max_bytes})"


def main() -> None:
    csv_path = find_csv_path()
    df = pd.read_csv(csv_path)

    # Expect columns: Run, Active_F1, Passive_F1
    runs = df["Run"].tolist()
    active = df["Active_F1"].tolist()
    passive = df["Passive_F1"].tolist()

    fig, ax = plt.subplots()

    ax.plot(runs, active, "o-", label="Active Learning", linewidth=2, markersize=6)
    ax.plot(runs, passive, "s-", label="Passive Learning", linewidth=2, markersize=6)
    ax.set_xlabel("Run")
    ax.set_ylabel("F1 Score")
    ax.set_title("F1 Score by Run")
    ax.legend(loc="upper left")
    ax.grid(False)

    plt.tight_layout()

    out_path = f"{DATA_DIR}/graphical_abstract_config62.jpg"
    msg = save_jpg_under_size(fig, out_path, width_px=660, height_px=295, max_bytes=45000)
    print(msg)


if __name__ == "__main__":
    main()


