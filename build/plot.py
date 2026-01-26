import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -------- configuration --------
csv_file = "validation_results.csv"
output_dir = "validation_plots"
os.makedirs(output_dir, exist_ok=True)

# -------- load data --------
df = pd.read_csv(csv_file)

# Ensure required columns exist in the CSV file
required_columns = [
    "expected_x", "unpacked_x",
    "expected_y", "unpacked_y",
    "expected_width", "unpacked_width"
]
missing_columns = [col for col in required_columns if col not in df.columns]
if missing_columns:
    raise ValueError(f"Missing required columns in the CSV file: {', '.join(missing_columns)}")

# -------- plotting configuration --------
plots = [
    ("x", "expected_x", "unpacked_x"),
    ("y", "expected_y", "unpacked_y"),
    ("width", "expected_width", "unpacked_width"),
]

for name, expected_col, unpacked_col in plots:
    fig, (ax_top, ax_bottom) = plt.subplots(
        2, 1, figsize=(8, 6),
        gridspec_kw={"height_ratios": [3, 1]},
        sharex=True
    )

    expected = df[expected_col].values.copy()
    unpacked = df[unpacked_col].values.copy()

    # -------- width special handling (integer-aligned bins for BOTH) --------
    if name == "width":
        # Combine both datasets to find full range
        all_widths = np.concatenate([expected, unpacked])
        min_val = int(all_widths.min())
        max_val = int(all_widths.max())

        bins = np.arange(min_val, max_val + 2, 1)
        centers = bins[:-1]
        widths_arr = np.ones(len(centers))

        counts_orig, _ = np.histogram(unpacked, bins=bins)
        counts_redigi, _ = np.histogram(expected, bins=bins)

        # Plot Stand-alone as filled bars
        ax_top.bar(
            centers,
            counts_orig,
            width=widths_arr,
            align="edge",
            color="blue",
            alpha=0.6,
            label="Stand-alone Unpacker"
        )

        # Plot CMSSW as histogram outline (unfilled)
        ax_top.hist(
            expected,
            bins=bins,
            #histtype='step',
            color="brown",
            #linewidth=2.5,
            alpha=0.6,
            label="CMSSW Unpacker"
        )

        # Set x-axis limit for width plot
        ax_top.set_xlim(0, 9)

    else:
        # For x and y: use combined range from both datasets
        all_values = np.concatenate([expected, unpacked])
        value_range = (all_values.min(), all_values.max())
        
        n_bins = 20
        counts_orig, edges = np.histogram(unpacked, bins=n_bins, range=value_range)
        counts_redigi, _ = np.histogram(expected, bins=edges)

        centers = edges[:-1]
        widths_arr = edges[1:] - edges[:-1]

        # Plot Stand-alone as filled bars
        ax_top.bar(
            centers,
            counts_orig,
            width=widths_arr,
            align="edge",
            color="blue",
            alpha=0.6,
            label="Stand-alone Unpacker"
        )

        # Plot CMSSW as histogram outline (unfilled)
        ax_top.hist(
            expected,
            bins=edges,
            #histtype='step',
            color="brown",
            alpha=0.6,
            label="CMSSW Unpacker"
        )

    # -------- top panel cosmetics --------
    ax_top.set_ylabel("Clusters", fontsize=12)
    ax_top.legend(fontsize=10)
    ax_top.grid(True, linestyle="--", alpha=0.5)
    ax_top.set_title(f"Validation Plot: {name.capitalize()}", fontsize=14, fontweight="bold")

    # -------- ratio panel --------
    ratio = np.ones_like(counts_orig, dtype=float)
    mask = counts_orig > 0
    ratio[mask] = counts_redigi[mask] / counts_orig[mask]

    ax_bottom.plot(
        centers,
        ratio,
        "^",
        color="blue",
        markersize=4,
        label="Ratio"
    )

    ax_bottom.axhline(1.0, color="black", linewidth=1, linestyle="--")
    ax_bottom.set_ylim(0.8, 1.2)
    ax_bottom.set_ylabel("Ratio", fontsize=12)
    ax_bottom.set_xlabel(name.capitalize(), fontsize=12)
    ax_bottom.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{name}_validation.png"), dpi=300)
    plt.close()

print(f"Validation plots saved to {output_dir}/")