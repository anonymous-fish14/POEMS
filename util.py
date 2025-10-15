from sklearn.manifold import TSNE
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import pearsonr, f_oneway
import numpy as np
import umap
import os
import torch
from matplotlib.colors import ListedColormap
import math
import sys

OMIC_COLOR_MAP = {
    "specific1": "#1f77b4",     # blue
    "specific2": "#ff7f0e",  # orange
    "specific3": "#2ca02c"     # green
}
OMIC_NAME_MAP = {
    "specific1": "mRNA",     # blue
    "specific2": "DNAMeth",  # orange
    "specific3": "miRNA"     # green
}
sparsity_threshold = 1e-2


def visualize_Ws(test_model, dir=None):
    """
    Enhanced visualization of W matrices, saving each row (type of plot) separately.
    Each figure row contains three omics side-by-side.
    """

    if dir:
        os.makedirs(dir, exist_ok=True)

    cmap_bin = ListedColormap(['red', 'white'])
    cmap_topk = ListedColormap(['white', 'red'])

    # --- Retrieve W matrices ---
    Ws = {}
    for omic_key in OMIC_COLOR_MAP.keys():
        W = test_model.specific_modules[omic_key].get_generator_mask().cpu().detach()
        W_clamped = W.clone()
        W_clamped[W_clamped.abs() <= sparsity_threshold] = 0.0
        Ws[omic_key] = W_clamped

    # Helper to iterate over modalities
    omic_items = list(OMIC_COLOR_MAP.items())

    # ===========================================================
    # ROW 1 — Sparsity masks
    # ===========================================================
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for i, (omic_key, color_value) in enumerate(omic_items):
        ax = axes[i]
        W_sparse = (Ws[omic_key].abs() <= sparsity_threshold).int()
        im = ax.imshow(W_sparse.T, cmap=cmap_bin, aspect='auto')
        ax.set_title(f"$|W_{{{OMIC_NAME_MAP[omic_key]}}}|$", fontsize=14)
        ax.set_xlabel("Input features", fontsize=12)
        if i == 0: ax.set_ylabel("Latent factors", fontsize=12)
        ax.legend(handles=[mpatches.Patch(color='red', label=f"Activation > {sparsity_threshold}")], fontsize=12)
    plt.tight_layout()
    if dir:
        plt.savefig(os.path.join(dir, "Ws_Row1_SparsityMask.pdf"), dpi=300, bbox_inches='tight')
        plt.close()

    # ===========================================================
    # ROW 2 — Histograms
    # ===========================================================
    fig, axes = plt.subplots(1, 3, figsize=(15, 4),sharey=True)
    for i, (omic_key, color_value) in enumerate(omic_items):
        ax = axes[i]
        ax.hist(Ws[omic_key].flatten(), bins=100, color=color_value, alpha=0.7)
        ax.set_title(f"Activation values in $|W_{{{OMIC_NAME_MAP[omic_key]}}}|$", fontsize=14)
        ax.set_xlabel("Value", fontsize=12)
        if i == 0: ax.set_ylabel("Frequency", fontsize=12)
    plt.tight_layout()
    if dir:
        plt.savefig(os.path.join(dir, "Ws_Row2_Histogram.pdf"), dpi=300, bbox_inches='tight')
        plt.close()


def plot_tsne(data, label, dir=None):
    plt.clf()
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(data)
    df = pd.DataFrame(columns=['tsne-2d-one', 'tsne-2d-two'])
    df['tsne-2d-one'] = tsne_results[:, 0]
    df['tsne-2d-two'] = tsne_results[:, 1]

    g = sns.scatterplot(
        x="tsne-2d-one", y="tsne-2d-two",
        hue=label,
        palette=sns.color_palette("hls", len(np.unique(label))),
        data=df,
        legend="full",
        alpha=0.7,
        s=40
    )
    for t, l in zip(g.legend_.texts, ["Normal-like", "Basal-like", "HER2-enriched", "Luminal A", "Luminal B"]):
        t.set_text(l)
    plt.title("t-SNE on latent representations ", fontsize=14)
    plt.xlabel("tsne-2d-one", fontsize=12)
    plt.ylabel("tsne-2d-two", fontsize=12)
    if(not (dir is None)):
        plt.savefig(dir+'tsne.pdf')
    else:
        plt.show()


def plot_umap(data, label, dir=None):
    plt.clf()
    embedding = umap.UMAP(random_state=42).fit_transform(data)
    df = pd.DataFrame(columns=['umap-2d-one', 'umap-2d-two'])
    df['umap-2d-one'] = embedding[:, 0]
    df['umap-2d-two'] = embedding[:, 1]

    g= sns.scatterplot(
        x="umap-2d-one", y="umap-2d-two",
        hue=label,
        palette=sns.color_palette("hls", len(np.unique(label))),
        data=df,
        legend="full",
        alpha=0.7,
        s=40
    )
    for t, l in zip(g.legend_.texts, ["Normal-like", "Basal-like", "HER2-enriched", "Luminal A", "Luminal B"]):
        t.set_text(l)
    plt.title("UMAP on latent representations", fontsize=14)
    plt.xlabel("umap-2d-one", fontsize=12)
    plt.ylabel("umap-2d-two", fontsize=12)
    if(not (dir is None)):
        plt.savefig(dir + 'umap.pdf')
    else:
        plt.show()

def visualize_final_embedding(embeddings,labels,dir=None):
    em_np = embeddings.cpu().numpy()
    labels = np.array(labels)
    sort_idx = np.argsort(labels)
    sorted_embeddings = em_np[sort_idx]
    sorted_labels = labels[sort_idx]
    
    plt.clf()
    plt.figure(figsize=(6, 3))
    ax = sns.heatmap(sorted_embeddings, cmap="coolwarm", cbar=True, yticklabels=False)
    cb = ax.collections[0].colorbar
    cb.ax.tick_params(labelsize=5)
    
    n_factors = sorted_embeddings.shape[1]
    ax.set_xticks(np.arange(n_factors) + 0.5)  # tick in center of each column
    ax.set_xticklabels([str(i) for i in range(n_factors)], fontsize=5, rotation=0)
    
    # Identify where to draw horizontal lines (cluster boundaries)
    unique_clusters, counts = np.unique(sorted_labels, return_counts=True)
    boundaries = np.cumsum(counts)[:-1]  # skip the last one

    for b in boundaries:
        ax.hlines(b, *ax.get_xlim(), colors='black', linestyles='dashed', linewidth=0.5)


    ax.set_xlabel("Latent factors",fontsize=6)
    ax.set_ylabel(f"Samples (cluster-sorted, n={len(labels)})",fontsize=6)
    plt.title("Latent embeddings heatmap",fontsize=7)
        
    if(not (dir is None)):
        plt.savefig(dir + 'final_em_test.pdf',bbox_inches='tight')
    else:
        plt.show()
    
    plt.tight_layout()
    plt.clf()
    plt.figure(figsize=(12, 6))
    sns.heatmap(pd.DataFrame(embeddings.cpu().numpy()).corr(), cmap="coolwarm", center=0, square=True, linewidths=0.5)
    plt.title("Latent Dimension Correlation Matrix")
    plt.xlabel("Latent Dimensions")
    plt.ylabel("Latent Dimensions")
    if(not (dir is None)):
        plt.savefig(dir + 'final_latent_corr_test.pdf')
    else:
        plt.show()

def plot_per_latent_dist_by_subtype(z, y, subtype_names, dir=None):
    """
    Plot a boxplot for each latent dim, showing subtype distributions side-by-side.

    Args:
        z: Tensor or numpy array, shape (n_samples, latent_dim)
        y: Tensor or array of subtype labels (ints from 0 to C-1)
        subtype_names: List of class names (length = #subtypes)
        dir: Optional path to save the figure
    """

    if isinstance(z, torch.Tensor): z = z.detach().cpu().numpy()
    if isinstance(y, torch.Tensor): y = y.detach().cpu().numpy()

    n_samples, n_latent = z.shape
    df = pd.DataFrame(z, columns=[f"z{i}" for i in range(n_latent)])
    df["Subtype"] = y

    if subtype_names:
        df["Subtype"] = df["Subtype"].map({i: name for i, name in enumerate(subtype_names)})

    # Plot layout
    ncols = 4
    nrows = math.ceil(n_latent / ncols)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 4, nrows * 4), sharey=False)
    palette = dict(zip(subtype_names, sns.color_palette("hls", len(subtype_names))))
    
    axes = axes.flatten() if n_latent > 1 else [axes]

    for i in range(n_latent):
        ax = axes[i]
        sns.boxplot(data=df, x="Subtype", y=f"z{i}", palette=palette, ax=ax)
        ax.set_title(f"Latent z{i}", fontsize=10)
        ax.tick_params(axis='x', rotation=30)
    
    # Turn off unused axes
    for j in range(n_latent, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    if dir:
        plt.savefig(dir+"PerLatentDist_BySubtype.pdf", dpi=300)
    else:
        plt.show()


def plot_per_subtype_latent_dist(z, y, subtype_names, dir=None):
    """
    For each subtype, draw a boxplot showing the distribution of all latent dimensions (z0, z1, ..., zn).

    Args:
        z: Tensor or ndarray, shape (n_samples, latent_dim)
        y: Tensor or array-like subtype labels (ints from 0 to C-1)
        subtype_names: Optional list of class names
        dir: Optional path to save final figure
    """

    if isinstance(z, torch.Tensor): z = z.detach().cpu().numpy()
    if isinstance(y, torch.Tensor): y = y.detach().cpu().numpy()

    n_samples, n_latent = z.shape
    df = pd.DataFrame(z, columns=[f"z{i}" for i in range(n_latent)])
    df["Subtype"] = y
    if subtype_names:
        df["Subtype"] = df["Subtype"].map({i: name for i, name in enumerate(subtype_names)})

    subtypes = df["Subtype"].unique()
    n_subtypes = len(subtypes)

    fig, axes = plt.subplots(nrows=n_subtypes, ncols=1, figsize=(max(10, n_latent * 0.5), 4 * n_subtypes), sharey=True)

    if n_subtypes == 1:
        axes = [axes]

    for idx, subtype in enumerate(subtypes):
        ax = axes[idx]
        df_sub = df[df["Subtype"] == subtype].drop(columns=["Subtype"])
        sns.boxplot(data=df_sub, palette="Set3", ax=ax)
        ax.set_title(f"Latent Distributions for Subtype: {subtype}", fontsize=12)
        ax.set_xlabel("Latent Dimension")
        ax.set_ylabel("Value")
        ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    if dir:
        os.makedirs(os.path.dirname(dir), exist_ok=True)
        plt.savefig(dir+"PerSubtype_LatentDist.pdf", dpi=300)
    else:
        plt.show()


def plot_subtype_correlations(X1,X2,X3, Z, y, subtype_names, dir=None):
    """
    Plots two heatmaps side-by-side: pairwise subtype correlation based on input features vs latent features.

    Args:
        X: Original input (n_samples, input_dim)
        Z: Latent representation (n_samples, latent_dim)
        y: Subtype labels (0 to C-1)
        subtype_names: List of class names
        dir: Optional PNG path to save output
    """
    if isinstance(X1, torch.Tensor): X1 = X1.detach().cpu().numpy()
    if isinstance(X2, torch.Tensor): X2 = X2.detach().cpu().numpy()
    if isinstance(X3, torch.Tensor): X3 = X3.detach().cpu().numpy()
    if isinstance(Z, torch.Tensor): Z = Z.detach().cpu().numpy()
    if isinstance(y, torch.Tensor): y = y.detach().cpu().numpy()

    subtype_labels = np.unique(y)
    subtype_names = subtype_names or [str(i) for i in subtype_labels]

    def compute_mean_vectors(data):
        return {i: data[y == i].mean(axis=0) for i in subtype_labels}

    def build_corr_matrix(means_dict):
        mat = np.zeros((len(subtype_labels), len(subtype_labels)))
        for i in range(len(subtype_labels)):
            for j in range(len(subtype_labels)):
                r, _ = pearsonr(means_dict[subtype_labels[i]], means_dict[subtype_labels[j]])
                mat[i, j] = r
        return mat

    input_means1 = compute_mean_vectors(X1)
    input_means2 = compute_mean_vectors(X2)
    input_means3 = compute_mean_vectors(X3)
    latent_means = compute_mean_vectors(Z)

    corr_input1 = build_corr_matrix(input_means1)
    corr_input2 = build_corr_matrix(input_means2)
    corr_input3 = build_corr_matrix(input_means3)
    corr_latent = build_corr_matrix(latent_means)

    # figure: 1x5 (last axis is colorbar)
    vmin, vmax = -1.0, 1.0
    titles = [
        "mRNA correlations",
        "DNAMeth correlations",
        "miRNA correlations",
        "Latent correlations"
    ]
    mats = [corr_input1, corr_input2, corr_input3, corr_latent]

    fig, axes = plt.subplots(
        1, 5, figsize=(3*5, 4),  # width scales with 4 plots
        gridspec_kw={"width_ratios": [1, 1, 1, 1, 0.04]},  # slim cbar axis
        squeeze=False
    )
    axes = axes[0]
    heat_axes = axes[:-1]
    cbar_ax = axes[-1]

    mappable = None
    for j, (ax, M, title) in enumerate(zip(heat_axes, mats, titles)):
        hm = sns.heatmap(
            M, annot=True, fmt=".2f",
            xticklabels=subtype_names, yticklabels=subtype_names,
            cmap="coolwarm", vmin=vmin, vmax=vmax,
            annot_kws={"size": 10},
            cbar=False, ax=ax, linewidths=0.5, linecolor='white'
        )
        ax.set_aspect('equal')
        if j == len(heat_axes) - 1:
            mappable = hm.collections[0]

        ax.set_title(title, fontsize=16, pad=6)
        ax.tick_params(axis='x', labelrotation=30, labelsize=12)
        ax.tick_params(axis='y', labelsize=12)

        # Keep y tick labels only on the first subplot to save horizontal space
        if j != 0:
            ax.set_yticklabels([])
            ax.tick_params(axis='y', left=False)

    # shared colorbar on the dedicated axis
    if mappable is not None:
        cb = fig.colorbar(mappable, cax=cbar_ax)
        cb.ax.tick_params(labelsize=10)

    # tighten layout & trim only the right if needed
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.05, right=0.98, left=0.06, top=0.90, bottom=0.12)

    if dir:
        os.makedirs(dir, exist_ok=True)
        plt.savefig(os.path.join(dir, "Subtype_Correlations.pdf"), dpi=300, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()
        

def plot_gating_alphas_stacked(alphas, dir=None):
    """
    Visualizes gating network outputs (alphas) as stacked bar plots per sample.
    Compact, publication-style version with larger text and adjusted layout.
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt

    # Convert to numpy
    alphas_np = alphas.detach().cpu().numpy()
    n_samples = alphas_np.shape[0]
    indices = np.arange(n_samples)

    # --- Figure setup (shorter y-axis, wider layout) ---
    fig, ax = plt.subplots(figsize=(6, 2)) 

    # --- Stacked bar plot ---
    ax.bar(indices, alphas_np[:, 0],
           label="mRNA", color=OMIC_COLOR_MAP["specific1"])
    ax.bar(indices, alphas_np[:, 1],
           bottom=alphas_np[:, 0],
           label="DNAMeth", color=OMIC_COLOR_MAP["specific2"])
    ax.bar(indices, alphas_np[:, 2],
           bottom=np.sum(alphas_np[:, :2], axis=1),
           label="miRNA", color=OMIC_COLOR_MAP["specific3"])

    # --- Labels & title ---
    ax.set_xlabel("Sample index", fontsize=4,labelpad=0)
    ax.set_ylabel("$\\alpha$ ", fontsize=4,labelpad=1)
    ax.set_title("Per-sample gating weights ($\\alpha$)", fontsize=5,pad=1)

    # --- Tick params ---
    ax.tick_params(axis='both', which='major',width=0.3, length=1.1,labelsize=3)
    ax.set_xticks(indices)
    ax.set_xticklabels(
        [str(i) if i in [indices[0], indices[-1]] else "" for i in indices],
        fontsize=2
    )
    
    ax.legend(fontsize=3)

    ax.set_ylim(0, 1.05)
    
    for spine in ['top', 'right','left', 'bottom']:
        ax.spines[spine].set_linewidth(0.25)     # make visible ones slightly thinner
    
    # --- Remove all internal margins ---
    # removes padding around the axes inside the figure
    ax.set_position([0, 0, 1, 1])   # (left, bottom, width, height) fills entire figure
    ax.margins(0, 0)                # no automatic margins on data limits
    ax.autoscale(enable=True, axis='both', tight=True)
    plt.tight_layout()
    
    # --- Save or show ---
    if dir:
        os.makedirs(dir, exist_ok=True)
        plt.savefig(os.path.join(dir, "GatingAlphas_StackedBar.pdf"),
                    dpi=300, bbox_inches='tight', pad_inches=0)
        plt.close()
    else:
        plt.show()

def plot_feature_importance(test_model,disease,omic_names=None,dir=None,top_k=10):
    """
    Analyze and visualize feature importances from the trained MocsSparseVAE model in a single figure.
    
    Args:
        test_model (MocsSparseVAE): Trained model instance.
        omic_names (list, optional): Display names for the omics. Defaults to ["specific1", "specific2", "specific3"].
        dir (str, optional): Path to save the figure. If None, displays it interactively.
        top_k (int): Number of top features to visualize per modality.
    """
    omic_keys = ["specific1", "specific2", "specific3"]
    if omic_names is None:
        omic_names = omic_keys

    fig, axes = plt.subplots(3, 3, figsize=(20, 20))
    
    cmap_bin = ListedColormap(['red', 'white'])  # for binary sparsity
    gene_names_1 = pd.read_csv(os.path.join(sys.path[1],'data',disease,'1_featname.csv'),header=None).iloc[:, 0].tolist()
    gene_names_2 = pd.read_csv(os.path.join(sys.path[1],'data',disease,'2_featname.csv'),header=None).iloc[:, 0].tolist()
    gene_names_3 = pd.read_csv(os.path.join(sys.path[1],'data',disease,'3_featname.csv'),header=None).iloc[:, 0].tolist()
    gene_names = [gene_names_1,gene_names_2,gene_names_3]   
    
    per_omic = []
    for idx, omic_key in enumerate(omic_keys):
        W = test_model.specific_modules[omic_key].get_generator_mask().detach().cpu().numpy()
        fi = np.sum(np.abs(W), axis=1)
        top_idx = np.argsort(fi)[::-1][:top_k]
        top_scores = fi[top_idx]
        top_names = [gene_names[idx][i] for i in top_idx]

        # Heatmap data (clamped by your sparsity threshold)
        W_clamped = W.copy()
        W_clamped[np.abs(W_clamped) <= sparsity_threshold] = 0.0
        W_top = W_clamped[top_idx, :]          # shape: [top_k, latent_dim]
        W_top_T = W_top.T                       # [latent_dim, top_k] for heatmap

        # Per-latent max contributors
        W_abs = np.abs(W)
        max_idx_per_lat = np.argmax(W_abs, axis=0)
        max_val_per_lat = W_abs[max_idx_per_lat, np.arange(W.shape[1])]
        max_name_per_lat = [gene_names[idx][i] for i in max_idx_per_lat]

        per_omic.append(dict(
            key=omic_key,
            display=omic_names[idx],
            W=W,
            top_idx=top_idx,
            top_scores=top_scores,
            top_names=top_names,
            heatmap=W_top_T,
            lat_vals=max_val_per_lat,
            lat_names=max_name_per_lat,
            latent_dim=W.shape[1]
        ))

    V = len(per_omic)

    # ======================================================
    # (1) Top-k barplots (all omics side-by-side)
    # ======================================================
    fig, axes = plt.subplots(1, V, figsize=(6*V, 5), squeeze=False)
    axes = axes[0]
    for j, info in enumerate(per_omic):
        ax = axes[j]
        ax.bar(range(top_k), info['top_scores'], color=OMIC_COLOR_MAP[info['key']])
        ax.set_xticks(range(top_k))
        ax.set_xticklabels(info['top_names'], rotation=30, ha='right', fontsize=12)
        ax.axhline(y=0.5, color='red', linestyle='--', linewidth=1)
        ax.set_title(f"Strengths of top {top_k} {info['display']} features", fontsize=16)
        ax.set_xlabel("")  # no x-labels
        if j == 0: ax.set_ylabel("Aggregated activation strenght", fontsize=14)
    plt.tight_layout()
    if dir:
        plt.savefig(os.path.join(dir, f"FeatureImportance_Top{top_k}_AllOmics.pdf"), dpi=300)
        plt.close(fig)
    else:
        plt.show()

    # ======================================================
    # (2) Heatmaps of |W| (shared scale; colorbar only on rightmost)
    # ======================================================
    # Shared symmetric color scale
    max_abs = max(np.max(np.abs(info['heatmap'])) for info in per_omic)
    vmin, vmax = -max_abs, max_abs

    # Create an extra axis for the colorbar so heatmaps stay the same size
    fig, axes = plt.subplots(
        1, V + 1, figsize=(6*V, 5), squeeze=False,
        gridspec_kw={'width_ratios': [1]*V + [0.05]}  # last column = cbar axis
    )
    axes = axes[0]
    heat_axes = axes[:-1]    # actual heatmap axes
    cbar_ax   = axes[-1]     # colorbar axis

    mappable_for_cbar = None
    for j, info in enumerate(per_omic):
        ax = heat_axes[j]
        hm = sns.heatmap(
            info['heatmap'],
            cmap="coolwarm",
            center=0,
            vmin=vmin, vmax=vmax,
            linewidths=0.5,
            ax=ax,
            cbar=False  # no per-axis colorbar; we'll add one shared below
        )
        if j == V - 1:
            # keep a reference to the QuadMesh for the shared colorbar
            mappable_for_cbar = hm.collections[0]

        ax.set_xticks(range(top_k))
        ax.set_xticklabels(info['top_names'], rotation=30, ha='right', fontsize=12)
        ax.set_title(f"Heatmap of $W_{{{info['display']}}}$", fontsize=16)
        ax.set_xlabel("")  # no x-labels
        ax.set_ylabel("Latent factors" if j == 0 else "", fontsize=14)

    # One shared colorbar that doesn't change subplot widths
    if mappable_for_cbar is not None:
        fig.colorbar(mappable_for_cbar, cax=cbar_ax)

    plt.tight_layout()
    if dir:
        plt.savefig(os.path.join(dir, "FeatureImportance_Heatmap_AllOmics.pdf"), dpi=300)
        plt.close(fig)
    else:
        plt.show()
    
    

    # ======================================================
    # (3) Per-latent max-contribution bars (all omics side-by-side)
    # ======================================================
    fig, axes = plt.subplots(1, V, figsize=(6*V, 7), squeeze=False)
    axes = axes[0]
    global_max = max(np.max(info['lat_vals']) for info in per_omic)
    for j, info in enumerate(per_omic):
        ax = axes[j]
        y = np.arange(info['latent_dim'])  # latent dims on y-axis
        ax.barh(y, info['lat_vals'], color=OMIC_COLOR_MAP[info['key']])  # horizontal bars

        # Label latent dims on y-axis
        ax.set_yticks(y)
        ax.set_yticklabels(y,fontsize=11)
        ax.invert_yaxis()  # optional: so latent dim 0 is at top
        ax.set_xlim(0, global_max * 1.23)  # add 10% headroom
        
        # Annotate gene names at the end of each bar
        for i, (val, name) in enumerate(zip(info['lat_vals'], info['lat_names'])):
            ax.text(val + 0.02, i, name, va='center', ha='left', fontsize=10)

        # Labeling
        ax.set_ylabel("Latent factors", fontsize=14 if j == 0 else 0)
        if j != 0:
            ax.set_ylabel("")
        ax.set_xlabel("Max absolute activation strengths", fontsize=14)
        ax.set_title(f"Top feature per latent dimension in {info['display']}",
                    fontsize=16)

    plt.tight_layout()
    plt.subplots_adjust(top=0.88)

    if dir:
        plt.savefig(os.path.join(dir, "FeatureImportance_LatentContrib_AllOmics.pdf"),
                    dpi=300, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()