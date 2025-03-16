import os
import argparse
from glob import glob
from typing import Dict, Literal, List, Set, Optional
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import polars as pl
from polars import selectors
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from matplotlib.patches import Ellipse
from scipy.stats import chi2
from tqdm import tqdm
parser = argparse.ArgumentParser(description="BERT Embedding Visualization Tool")
parser.add_argument(
    "--max_rewards_per_plot",
    type=int,
    default=0,
    help="Maximum number of rewards to display per plot (None or 0 displays all rewards, default: 0)",
)

args = parser.parse_args()
MAX_REWARDS_PER_PLOT = (
    args.max_rewards_per_plot if args.max_rewards_per_plot > 0 else None
)
MODEL_TYPE: Literal["albert", "bert", "electra", "roberta"] = "bert"
RANDOM_STATE = 42
PERPLEXITY = 30
N_COMPONENTS = 2
N_JOBS = -1 
BASE_DIR = os.path.dirname(__file__)
IMAGE_DIR = os.path.join(BASE_DIR, "visualization_results")
SOURCE_DIR = os.path.join(BASE_DIR, "..", "instruct", "test", "bert")
TSNE_SCATTER_DIR = os.path.join(IMAGE_DIR, "tsne_scatter")

os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(TSNE_SCATTER_DIR, exist_ok=True)
def create_figure(figsize=(10, 8)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_axisbelow(True)
    return fig, ax


def setup_axes(ax, title, xlabel="t-SNE Component 1", ylabel="t-SNE Component 2"):
    ax.set_title(title, fontsize=14, pad=20, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=12, labelpad=10)
    ax.set_ylabel(ylabel, fontsize=12, labelpad=10)
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
    fig.tight_layout()
    return fig, ax
def generate_colors(n):
    cmap = plt.cm.tab10
    return [cmap(i % cmap.N) for i in range(n)]
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
print("Loading data...")
path_list = glob(os.path.join(SOURCE_DIR, "*.csv"))
pdf_list: List[pl.DataFrame] = []
for path in tqdm(path_list, desc="Loading CSV files"):
    try:
        data = pl.read_csv(path)
        pdf_list.append(data)
    except Exception as e:
        print(f"Error occurred while loading file {path}: {e}")
        continue

if not pdf_list:
    raise ValueError("No data was loaded. Please check the file paths.")

print(f"Total {len(pdf_list)} files loaded")
pdf = pl.concat(pdf_list)
print(f"Data merging completed. Total {len(pdf)} rows")
print("Preprocessing data...")
pdf = pdf.unique()
print(f"Data after preprocessing: {len(pdf)} rows")

unique_rewards = pdf["reward_enum"].unique().to_list()
print(f"Unique reward_enum values: {unique_rewards}")

embed_cols = [col for col in pdf.columns if col.startswith("embed_")]
if not embed_cols:
    raise ValueError("Embedding columns not found. Please check the data structure.")
print("Performing t-SNE transformation...")

embeddings = pdf.select(selectors.starts_with("embed_")).to_numpy()
scaler = StandardScaler()
normalized_embeddings = scaler.fit_transform(embeddings)

try:
    try:
        from cuml.manifold import TSNE as cuTSNE

        print("Using CuML t-SNE...")
        tsne = cuTSNE(
            n_components=N_COMPONENTS, perplexity=PERPLEXITY, random_state=RANDOM_STATE
        )
        tsne_results = tsne.fit_transform(normalized_embeddings)
    except (ImportError, ModuleNotFoundError):
        print("Using scikit-learn t-SNE...")
        tsne = TSNE(
            n_components=N_COMPONENTS,
            perplexity=PERPLEXITY,
            n_jobs=N_JOBS,
            random_state=RANDOM_STATE,
        )
        tsne_results = tsne.fit_transform(normalized_embeddings)
except Exception as e:
    print(f"Error occurred during t-SNE transformation: {e}")
    print("Retrying with simplified parameters...")
    tsne = TSNE(n_components=N_COMPONENTS, random_state=RANDOM_STATE)
    tsne_results = tsne.fit_transform(normalized_embeddings)

pdf = pdf.with_columns(
    [
        pl.Series(name="tsne_x", values=tsne_results[:, 0]),
        pl.Series(name="tsne_y", values=tsne_results[:, 1]),
    ]
)

tsne_result_path = os.path.join(BASE_DIR, "tsne_results.csv")
pdf.write_csv(tsne_result_path)
print(f"t-SNE results saved at: {tsne_result_path}")
def get_reward_elements(reward_enum: int) -> List[int]:
    if reward_enum == 0:
        return [0]
    return [int(digit) for digit in str(reward_enum)]


def split_into_batches(items, batch_size: Optional[int] = None):
    if batch_size is None:
        return [items]
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]


reward_groups: Dict[int, Set[int]] = {}
for reward_enum in unique_rewards:
    elements = get_reward_elements(reward_enum)
    group_len = len(elements)
    if group_len not in reward_groups:
        reward_groups[group_len] = set()
    reward_groups[group_len].add(reward_enum)

for combo_len in reward_groups.keys():
    combo_dir = os.path.join(TSNE_SCATTER_DIR, f"combo_len_{combo_len}")
    os.makedirs(combo_dir, exist_ok=True)
num_rewards = len(unique_rewards)
reward_colors = dict(zip(unique_rewards, generate_colors(num_rewards)))
print(
    f"Generating visualizations by combination length... (max_rewards_per_plot: {MAX_REWARDS_PER_PLOT or 'All'})"
)
for combo_len, reward_enums in reward_groups.items():
    print(
        f"\n=== Reward visualization for combination length {combo_len} ({len(reward_enums)} combinations) ==="
    )
    combo_dir = os.path.join(TSNE_SCATTER_DIR, f"combo_len_{combo_len}")
    reward_batches = split_into_batches(list(reward_enums), MAX_REWARDS_PER_PLOT)
    for batch_idx, reward_batch in enumerate(reward_batches):
        fig, ax = create_figure(figsize=(12, 10))
        for reward_enum in reward_batch:
            filtered_df = pdf.filter(pl.col("reward_enum") == reward_enum)
            if len(filtered_df) == 0:
                continue
            tsne_x = filtered_df["tsne_x"].to_numpy()
            tsne_y = filtered_df["tsne_y"].to_numpy()
            color = reward_colors[reward_enum]
            ax.scatter(
                tsne_x,
                tsne_y,
                c=[color],
                marker="o",
                alpha=0.7,
                s=40,
                edgecolors="none",
                label=f"Reward: {reward_enum}",
            )

        title_suffix = f" (Batch {batch_idx + 1})" if len(reward_batches) > 1 else ""
        title = f"t-SNE Scatter Plot - Combo Length {combo_len}{title_suffix}"
        setup_axes(ax, title)
        finalize_plot(fig, ax)

        file_suffix = f"_batch{batch_idx + 1}" if len(reward_batches) > 1 else ""
        save_path = os.path.join(
            combo_dir, f"tsne_scatter_all_combo{combo_len}{file_suffix}.png"
        )
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    for batch_idx, reward_batch in enumerate(reward_batches):
        fig, ax = create_figure(figsize=(12, 10))
        for reward_enum in reward_batch:
            filtered_df = pdf.filter(pl.col("reward_enum") == reward_enum)
            if len(filtered_df) == 0:
                continue
            tsne_x = filtered_df["tsne_x"].to_numpy()
            tsne_y = filtered_df["tsne_y"].to_numpy()
            color = reward_colors[reward_enum]
            ax.scatter(
                tsne_x,
                tsne_y,
                c=[color],
                marker="o",
                alpha=0.7,
                s=40,
                edgecolors="none",
                label=f"Reward: {reward_enum}",
            )
            if len(tsne_x) >= 3:
                draw_confidence_ellipse(
                    tsne_x,
                    tsne_y,
                    ax,
                    n_std=2.0,
                    edgecolor=color,
                    facecolor="none",
                    alpha=0.5,
                    linewidth=1.0,
                    linestyle="--",
                    zorder=1,
                )

        title_suffix = f" (Batch {batch_idx + 1})" if len(reward_batches) > 1 else ""
        title = f"t-SNE Visualization with Cluster Regions - Combo Length {combo_len}{title_suffix}"
        setup_axes(ax, title)
        finalize_plot(fig, ax)

        file_suffix = f"_batch{batch_idx + 1}" if len(reward_batches) > 1 else ""
        save_path = os.path.join(
            combo_dir, f"tsne_clusters_combo{combo_len}{file_suffix}.png"
        )
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    for batch_idx, reward_batch in enumerate(reward_batches):
        fig, ax = create_figure(figsize=(12, 10))
        for reward_enum in reward_batch:
            filtered_df = pdf.filter(pl.col("reward_enum") == reward_enum)
            if len(filtered_df) == 0:
                continue
            tsne_x = filtered_df["tsne_x"].to_numpy()
            tsne_y = filtered_df["tsne_y"].to_numpy()
            color = reward_colors[reward_enum]
            if len(tsne_x) > 10:
                sns.kdeplot(
                    x=tsne_x,
                    y=tsne_y,
                    ax=ax,
                    levels=3,
                    fill=False,
                    color=color,
                    alpha=0.5,
                    linewidths=1.0,
                    zorder=1,
                )
            ax.scatter(
                tsne_x,
                tsne_y,
                c=[color],
                marker="o",
                alpha=0.7,
                s=40,
                edgecolors="none",
                label=f"Reward: {reward_enum}",
            )

        title_suffix = f" (Batch {batch_idx + 1})" if len(reward_batches) > 1 else ""
        title = f"t-SNE Visualization with Density Contours - Combo Length {combo_len}{title_suffix}"
        setup_axes(ax, title)
        finalize_plot(fig, ax)

        file_suffix = f"_batch{batch_idx + 1}" if len(reward_batches) > 1 else ""
        save_path = os.path.join(
            combo_dir, f"tsne_contours_combo{combo_len}{file_suffix}.png"
        )
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
for combo_len, reward_enums in reward_groups.items():
    combo_dir = os.path.join(TSNE_SCATTER_DIR, f"combo_len_{combo_len}")
    readme_content = f"# Reward Combination Length {combo_len} Visualization\n\n"
    readme_content += f"This directory contains visualizations for combinations consisting of {combo_len} reward elements.\n\n"
    readme_content += "## Included Reward Combinations:\n\n"

    for reward_enum in sorted(reward_enums):
        elements = get_reward_elements(reward_enum)
        elements_str = ", ".join(map(str, elements))
        readme_content += f"- Reward {reward_enum}: [{elements_str}]\n"
    with open(os.path.join(combo_dir, "README.md"), "w", encoding="utf-8") as f:
        f.write(readme_content)

summary_content = "# BERT Embedding Visualization Summary\n\n"
summary_content += "## Data Distribution by Combination Length\n\n"
for combo_len, reward_enums in sorted(reward_groups.items()):
    summary_content += f"- Combination Length {combo_len}: {len(reward_enums)} combinations\n"
summary_content += "\nYou can find the visualization results for each reward combination in the corresponding directories.\n"
summary_content += f"\n## Visualization Settings\n\n"
summary_content += f"- max_rewards_per_plot: {MAX_REWARDS_PER_PLOT or 'No limit'}\n"
summary_content += f"- Model Type: {MODEL_TYPE}\n"
summary_content += f"- t-SNE Dimensions: {N_COMPONENTS}\n"
summary_content += f"- t-SNE Perplexity: {PERPLEXITY}\n"
with open(os.path.join(TSNE_SCATTER_DIR, "summary.md"), "w", encoding="utf-8") as f:
    f.write(summary_content)

print("All visualizations have been successfully generated.")
print("=== Visualization Results Location ===")
print(f"- Base Directory: {IMAGE_DIR}")
print(f"- t-SNE Scatter Plot: {TSNE_SCATTER_DIR}")
print("- Directories by Combination Length:")
for combo_len in reward_groups.keys():
    combo_dir = os.path.join(TSNE_SCATTER_DIR, f"combo_len_{combo_len}")
    print(f"  - Combination Length {combo_len}: {combo_dir}")

print("\nHow to Run the Script:")
print("1. To use the default value (max_rewards_per_plot=10):")
print("   python script_name.py")
print("2. To specify a custom value:")
print("   python script_name.py --max_rewards_per_plot 5")
print("3. To display all rewards without limits:")
print("   python script_name.py --max_rewards_per_plot 0")