import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker
import matplotlib.gridspec as gridspec
import os

# ── output directory ──────────────────────────────────────────────────────────
os.makedirs("validation_plots", exist_ok=True)

# ── load data ─────────────────────────────────────────────────────────────────
standalone_df = pd.read_csv("standalone_clusters_soa.csv")
truth_df      = pd.read_csv(
    "/home/momedmoh/data/newdata/output/Chronotestset/truth_clusterprop_soa.csv"
)
standalone_df = standalone_df.rename(columns={"modType": "moduleType"})

# ── exact-key match ───────────────────────────────────────────────────────────
keys = ["event", "detId", "x", "y", "z", "width", "isSeed", "mip", "moduleType"]
merged = pd.merge(standalone_df, truth_df, on=keys, how="inner",
                  suffixes=("_standalone", "_truth"))

print(f"standalone rows : {len(standalone_df)}")
print(f"truth rows      : {len(truth_df)}")
print(f"matched rows    : {len(merged)}")

# ── style constants (matching reference image) ────────────────────────────────
COLOR_CMSSW      = "magenta"  # magenta line  — CMSSW Unpacker
COLOR_STANDALONE = "green"    # green fill    — Stand-alone Unpacker
ALPHA            = 0.75
RATIO_COLOR      = "#2020CC"  # dark blue triangles

# ── variables to plot ─────────────────────────────────────────────────────────
plot_vars = {
    "x"          : {"bins": 50,                        "label": "x [strip / pixel units]"},
    "y"          : {"bins": 50,                        "label": "y [strip / pixel units]"},
    "z"          : {"bins": 50,                        "label": "z [strip / pixel units]"},
    "width"      : {"bins": np.arange(0.5, 15.5, 1.0), "label": "cluster size"},
    "mip"        : {"bins": 40,                        "label": "MIP charge"},
    "isSeed"     : {"bins": 2,                         "label": "isSeed flag"},
    "moduleType" : {"bins": 4,                         "label": "module type"},
    "detId"      : {"bins": 60,                        "label": "detId"},
}

# ── helper ────────────────────────────────────────────────────────────────────
def ratio_with_err(num_vals, den_vals, bins):
    h_num, edges = np.histogram(num_vals, bins=bins)
    h_den, _     = np.histogram(den_vals, bins=edges)
    ratio     = np.full_like(h_num, np.nan, dtype=float)
    ratio_err = np.full_like(h_num, np.nan, dtype=float)
    mask = h_den > 0
    ratio[mask]     = h_num[mask] / h_den[mask]
    ratio_err[mask] = ratio[mask] * np.sqrt(
        1.0 / h_num[mask].clip(1) + 1.0 / h_den[mask])
    centres = 0.5 * (edges[:-1] + edges[1:])
    return centres, ratio, ratio_err, edges

# ── plot loop ─────────────────────────────────────────────────────────────────
for var, cfg in plot_vars.items():
    sa_vals    = standalone_df[var].dropna()
    truth_vals = truth_df[var].dropna()

    bins = cfg["bins"] if isinstance(cfg["bins"], np.ndarray) else np.linspace(
        min(sa_vals.min(), truth_vals.min()),
        max(sa_vals.max(), truth_vals.max()),
        cfg["bins"] + 1
    )

    fig = plt.figure(figsize=(9, 7), facecolor="white")
    gs  = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.08)
    ax_main  = fig.add_subplot(gs[0])
    ax_ratio = fig.add_subplot(gs[1], sharex=ax_main)

    # ── main: filled overlapping bars ────────────
    ax_main.hist(truth_vals, bins=bins,
                 color=COLOR_CMSSW, alpha=ALPHA,
                 label="CMSSW Unpacker", edgecolor=COLOR_CMSSW,
                 histtype="step", linewidth=1.8, zorder=2)
    ax_main.hist(sa_vals, bins=bins,
                 color=COLOR_STANDALONE, alpha=ALPHA,
                 label="Stand-alone Unpacker", edgecolor="none", zorder=3)

    ax_main.set_ylabel("Clusters", fontsize=13)
    ax_main.legend(fontsize=11, frameon=True, framealpha=0.9, loc="upper right")
    ax_main.tick_params(labelbottom=False, direction="out", which="both")
    ax_main.tick_params(axis="y", labelsize=11)
    ax_main.set_axisbelow(True)
    ax_main.grid(color="lightgrey", linestyle="--", linewidth=0.8, zorder=0)
    ax_main.set_facecolor("white")
    for sp in ax_main.spines.values():
        sp.set_linewidth(0.8); sp.set_color("#444444")

    # ── ratio: triangle markers ───────────────────
    centres, ratio, ratio_err, _ = ratio_with_err(sa_vals, truth_vals, bins)
    mask = np.isfinite(ratio)
    ax_ratio.plot(centres[mask], ratio[mask],
                  marker="^", markersize=5,
                  color=RATIO_COLOR, linewidth=0, zorder=3)
    ax_ratio.axhline(1.0, color="black", linewidth=1.0, linestyle="--", zorder=2)

    ax_ratio.set_ylim(0.8, 1.2)
    ax_ratio.set_yticks([0.8, 0.9, 1.0, 1.1, 1.2])
    ax_ratio.set_yticklabels(["0.8", "0.9", "1.0", "1.1", "1.2"], fontsize=9)
    ax_ratio.set_ylabel("Ratio", fontsize=12)
    ax_ratio.set_xlabel(cfg["label"], fontsize=13)
    ax_ratio.tick_params(direction="out", which="both")
    ax_ratio.tick_params(axis="x", labelsize=11)
    ax_ratio.set_axisbelow(True)
    ax_ratio.grid(axis="y", color="lightgrey", linestyle="--", linewidth=0.8, zorder=0)
    ax_ratio.set_facecolor("white")
    for sp in ax_ratio.spines.values():
        sp.set_linewidth(0.8); sp.set_color("#444444")

    ax_main.yaxis.set_label_coords(-0.07, 0.5)
    ax_ratio.yaxis.set_label_coords(-0.07, 0.5)

    out_path = f"validation_plots/{var}_validation.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {out_path}")

# ── summary table ─────────────────────────────────────────────────────────────
n_sa = len(standalone_df); n_truth = len(truth_df); n_match = len(merged)
fig_s, ax_s = plt.subplots(figsize=(5, 2.5), facecolor="white")
ax_s.axis("off")
tbl = ax_s.table(
    cellText=[
        ["Standalone rows",        f"{n_sa}"],
        ["Truth rows",             f"{n_truth}"],
        ["Matched rows",           f"{n_match}"],
        ["Match fraction (SA)",    f"{n_match/n_sa*100:.1f} %"],
        ["Match fraction (Truth)", f"{n_match/n_truth*100:.1f} %"],
    ],
    colLabels=["Metric", "Value"], cellLoc="center", loc="center"
)
tbl.auto_set_font_size(False); tbl.set_fontsize(10); tbl.scale(1.4, 1.6)
fig_s.suptitle("Matching summary", fontsize=12, y=0.98)
fig_s.savefig("validation_plots/matching_summary.png", dpi=150, bbox_inches="tight")
plt.close(fig_s)
print("saved: validation_plots/matching_summary.png")
print("All done.")