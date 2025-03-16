import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

import matplotlib.colors as mcolors
import matplotlib.cm as cm
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use('Agg')

def create_scatter_plot(df, epoch, config, min_val=0, max_val=1,
                        xlim=(-10, 10), ylim=(-10, 10), postfix=""):

    """
    Function to plot Ground Truth vs Prediction.

    - Hue color varies based on the epoch value.
    - Colorbar range is set from min_val to max_val (n_epochs).
    """


    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(8, 6))
    norm = mcolors.Normalize(vmin=min_val, vmax=max_val)
    sm = cm.ScalarMappable(cmap="viridis", norm=norm)
    sm.set_array([])  
    _ = sns.scatterplot(
        data=df, x="ground_truth", y="prediction",
        hue="reward_id", palette="bright", alpha=0.5, ax=ax
    )
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label("Epoch")  

    ax.set_ylim(*ylim)
    ax.set_xlim(*xlim)
    ax.set_xlabel("Ground Truth")
    ax.set_ylabel("Prediction")

    ax.grid(True)
    plt.tight_layout()
    os.makedirs(os.path.join(config.exp_dir, config.figure_dir), exist_ok=True)
    fig_path = os.path.join(config.exp_dir, config.figure_dir, f"scatter_epoch_{epoch}{postfix}.png")
    plt.savefig(fig_path)
    plt.close(fig)

    return fig_path


def create_embedding_figure(embed_queue, reward_df: pd.DataFrame, epoch, config, postfix="") -> str:
    reward_ids = [e.reward_id for e in embed_queue]
    embeds = np.array([e.embedding for e in embed_queue])
    tsne = TSNE(n_components=2, random_state=42)
    tsne_embeds = tsne.fit_transform(embeds)

    inst_cols = reward_df.iloc[reward_ids][['reward_enum']].reset_index()
    tsne_df = pd.DataFrame(tsne_embeds, columns=['tsne_x', 'tsne_y']).reset_index()
    df = pd.concat([inst_cols, tsne_df], axis=1).drop(columns=['index'])

    # draw scatter plot
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.scatterplot(
        data=df, x="tsne_x", y="tsne_y",
        hue="reward_enum", palette="bright", alpha=0.9, ax=ax
    )

    ax.set_xlabel("Projection X")
    ax.set_ylabel("Projection Y")
    ax.grid(True)
    plt.tight_layout()
    os.makedirs(os.path.join(config.exp_dir, config.figure_dir), exist_ok=True)
    fig_path = os.path.join(config.exp_dir, config.figure_dir, f"embed_epoch_{epoch}{postfix}.png")
    plt.savefig(fig_path)
    plt.close(fig)

    return fig_path