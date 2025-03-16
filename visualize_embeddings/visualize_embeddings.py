from os.path import dirname, join, splitext, exists, basename, relpath, abspath
import os
import polars
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Ellipse
from scipy.stats import chi2
from typing import Dict, Literal
from tqdm import tqdm

import shutil
BASE_DIR = dirname(__file__)
SOURCE_DIR = abspath(
    join(BASE_DIR, "..", "output")
)  
TARGET_DIR = abspath(join(BASE_DIR, "output"))  
IMAGE_DIR = join(BASE_DIR, "images")  
def create_figure(figsize=(14, 12)):  
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_axisbelow(True)
    return fig, ax


def setup_axes(ax, title, xlabel="t-SNE Component 1", ylabel="t-SNE Component 2"):
    ax.set_title(title, fontsize=14, pad=20, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=12, labelpad=10)
    ax.set_ylabel(ylabel, fontsize=12, labelpad=10)
    ax.margins(0.15)  

    return ax


def finalize_plot(fig, ax, legend_title="Reward Types", legend_loc="best"):
    if ax.get_legend_handles_labels()[0]:
        ax.legend(
            title=legend_title,
            title_fontsize=12,
            fontsize=10,
            loc=legend_loc,
            framealpha=0.7,
            edgecolor="#cccccc",
        )
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    fig.tight_layout()
    return fig, ax
def draw_confidence_ellipse(x, y, ax, n_std=2.0, facecolor="none", **kwargs):
    if x.size != y.size:
        raise ValueError("x and y must be the same size")
    if len(x) < 3:
        return None
    mean_x, mean_y = np.mean(x), np.mean(y)
    cov = np.cov(x, y)
    eigenvals, eigenvecs = np.linalg.eigh(cov)
    if np.min(eigenvals) <= 0:
        min_eig = max(np.max(eigenvals) * 0.001, 1e-6)
        eigenvals = np.maximum(eigenvals, min_eig)
    chisquare_val = chi2.ppf(0.95, 2)
    width = 2 * n_std * np.sqrt(eigenvals[0] * chisquare_val)
    height = 2 * n_std * np.sqrt(eigenvals[1] * chisquare_val)
    angle = np.degrees(np.arctan2(eigenvecs[1][0], eigenvecs[0][0]))
    ellipse = Ellipse(
        (mean_x, mean_y),
        width=width,
        height=height,
        angle=angle,
        facecolor=facecolor,
        **kwargs,
    )
    return ax.add_patch(ellipse)
colors = plt.cm.tab10(range(10))  
def process_file(file_path, source_dir, target_dir):
    rel_path = relpath(file_path, source_dir)
    output_dir = join(target_dir, dirname(rel_path))
    os.makedirs(output_dir, exist_ok=True)  
    run_name = splitext(basename(file_path))[0]
    tqdm.write(f"Processing file: {file_path}")
    tqdm.write(f"Ouput directory: {output_dir}")

    try:
        pdf = polars.read_csv(file_path)
        grouped_pdf = pdf.group_by("reward_enum")
        coords: Dict[str, Dict[Literal["tsne_x", "tsne_y"], np.ndarray]] = {}
        for name, pdf_ in grouped_pdf:
            reward_enum = name[0]
            tsne_x = pdf_["tsne_x"].to_numpy()
            tsne_y = pdf_["tsne_y"].to_numpy()
            coords[reward_enum] = {"tsne_x": tsne_x, "tsne_y": tsne_y}
    except Exception as e:
        tqdm.write(f"File reading error: {file_path}")
        tqdm.write(f"Err: {e}")
        return
    plot_types = ["basic", "clusters", "contours"]
    with tqdm(
        total=len(plot_types), desc=f"'{run_name}' plot graph", leave=False
    ) as pbar:
        fig, ax = create_figure()
        for i, (reward_enum, data) in enumerate(coords.items()):
            color = colors[i % len(colors)]
            ax.scatter(
                data["tsne_x"],
                data["tsne_y"],
                c=[color],
                marker="o",
                alpha=0.7,
                s=40,
                edgecolors="none",
                label=f"Reward: {reward_enum}",
            )
        title = f"t-SNE Visualization of Embedded Data - {run_name}"
        setup_axes(ax, title)
        finalize_plot(fig, ax)
        fig.savefig(
            join(output_dir, f"{run_name}_basic.png"),
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.5,  
        )
        plt.close(fig)
        pbar.update(1)
        fig, ax = create_figure()
        for i, (reward_enum, data) in enumerate(coords.items()):
            color = colors[i % len(colors)]
            ax.scatter(
                data["tsne_x"],
                data["tsne_y"],
                c=[color],
                marker="o",
                alpha=0.7,
                s=40,
                edgecolors="none",
                label=f"Reward: {reward_enum}",
            )
            if len(data["tsne_x"]) >= 3:
                draw_confidence_ellipse(
                    data["tsne_x"],
                    data["tsne_y"],
                    ax,
                    n_std=2.0,
                    edgecolor=color,
                    facecolor="none",
                    alpha=0.5,
                    linewidth=1.0,
                    linestyle="--",
                    zorder=1,
                )
        title = f"t-SNE Visualization with Cluster Regions - {run_name}"
        setup_axes(ax, title)
        finalize_plot(fig, ax)
        fig.savefig(
            join(output_dir, f"{run_name}_clusters.png"),
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.5,  
        )
        plt.close(fig)
        pbar.update(1)
        fig, ax = create_figure()
        for i, (reward_enum, data) in enumerate(coords.items()):
            color = colors[i % len(colors)]
            x, y = data["tsne_x"], data["tsne_y"]
            if len(x) > 10:
                sns.kdeplot(
                    x=x,
                    y=y,
                    ax=ax,
                    levels=3,
                    fill=False,
                    color=color,
                    alpha=0.5,
                    linewidths=1.0,
                    zorder=1,
                )
            ax.scatter(
                x,
                y,
                c=[color],
                marker="o",
                alpha=0.7,
                s=40,
                edgecolors="none",
                label=f"Reward: {reward_enum}",
            )
        title = f"t-SNE Visualization with Density Contours - {run_name}"
        setup_axes(ax, title)
        finalize_plot(fig, ax)
        fig.savefig(
            join(output_dir, f"{run_name}_contours.png"),
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.5,  
        )
        plt.close(fig)
        pbar.update(1)

    tqdm.write(f"'{run_name}' file processed")
def main(source_dir=SOURCE_DIR, target_dir=TARGET_DIR):
    csv_files = []
    for root, _, files in os.walk(source_dir):
        for file in files:
            if file.endswith(".csv"):
                file_path = join(root, file)
                if exists(file_path):
                    csv_files.append(file_path)
                else:
                    tqdm.write(f"Can't find file: {file_path}")

    if exists(target_dir):
        shutil.rmtree(target_dir)  
    for file_path in tqdm(csv_files, desc="Processing CSV file", unit="files"):
        process_file(file_path, source_dir, target_dir)


if __name__ == "__main__":
    main()
