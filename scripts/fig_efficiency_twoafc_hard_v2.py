import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# =========================
# Style (IJCAI-like, your palette)
# =========================
COL_S1 = "#C4D5EB"   # blue
COL_S5 = "#FDE5D4"   # orange/peach
COL_GRID = "#E8E8E8" # light gray
COL_TEXT = "#2f2f2f"
COL_AXIS = "#333333"

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 16,
    "axes.titlesize": 18,
    "axes.labelsize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
    "figure.titlesize": 20,
})

def _darken(hex_color, factor=0.72):
    """Darken a hex color for line strokes while keeping your palette as facecolor."""
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16) / 255.0
    g = int(hex_color[2:4], 16) / 255.0
    b = int(hex_color[4:6], 16) / 255.0
    r, g, b = r * factor, g * factor, b * factor
    return "#{:02x}{:02x}{:02x}".format(int(r*255), int(g*255), int(b*255))

LINE_S1 = _darken(COL_S1, 0.70)
LINE_S5 = _darken(COL_S5, 0.70)

# =========================
# Data Loading (replaces TODO placeholder)
# =========================
# Automatically load data from efficiency_summary.csv
csv_path = "/mnt/work/repos/TextAlign-mindeye2/results/tables/efficiency_summary.csv"

if not os.path.exists(csv_path):
    print(f"Warning: CSV not found at {csv_path}. Using placeholder data values for demonstration.")
    # Fallback to user provided placeholders if CSV missing
    baseline_s1_mean = np.array([0.858, 0.905, 0.985])
    baseline_s1_err  = np.array([0.010, 0.008, 0.004])
    baseline_s5_mean = np.array([0.870, 0.927, 0.990])
    baseline_s5_err  = np.array([0.009, 0.006, 0.003])
    textalign_s1_mean = np.array([0.903, 0.932, 0.985])
    textalign_s1_err  = np.array([0.006, 0.006, 0.003])
    textalign_s5_mean = np.array([0.907, 0.939, 0.989])
    textalign_s5_err  = np.array([0.006, 0.005, 0.003])
else:
    df = pd.read_csv(csv_path)
    # The columns in efficiency_summary.csv are:
    # subj,model,setting,seed,N,ccd_acc1,ccd_ci_lo,ccd_ci_hi,twoafc_hard,twoafc_ci_lo,twoafc_ci_hi,rsa_rho,rsa_ci_lo,rsa_ci_hi
    # We want twoafc_hard and its error.
    
    # Helper to clean model names and settings
    df['model'] = df['model'].astype(str).str.lower()
    df['setting'] = df['setting'].astype(str).str.lower()
    
    # Define budget order
    budget_map = {"1sess": 0, "1h": 0, "1": 0, 
                  "2sess": 1, "2h": 1, "2": 1,
                  "40sess": 2, "full": 2, "40": 2}
    
    df['budget_idx'] = df['setting'].map(budget_map)
    
    def extract_data(model_name, subj_id):
        # Filter
        d = df[(df['model'] == model_name) & (df['subj'] == subj_id)].copy()
        if d.empty:
            return np.array([np.nan, np.nan, np.nan]), np.array([0,0,0])
            
        # Sort by budget
        d = d.sort_values('budget_idx')
        
        # Select metric
        mean = d['twoafc_hard'].values
        hi = d['twoafc_ci_hi'].values
        lo = d['twoafc_ci_lo'].values
        
        # Calculate approximate symmetric error for plotting, or use per-point
        # Error bar format: usually symmetric or deviations. 
        # Matplotlib yerr takes (2, N) for asymmetric.
        err_low = mean - lo
        err_high = hi - mean
        err = np.vstack([err_low, err_high])
        
        # If any missing points in 0,1,2 sequence?
        # Reindex to ensure 3 points
        expected_indices = [0, 1, 2]
        final_mean = []
        final_err_lo = []
        final_err_hi = []
        
        # Create a dict for lookup
        lookup_mean = dict(zip(d['budget_idx'], mean))
        lookup_lo = dict(zip(d['budget_idx'], err_low))
        lookup_hi = dict(zip(d['budget_idx'], err_high))
        
        for idx in expected_indices:
            if idx in lookup_mean:
                final_mean.append(lookup_mean[idx])
                final_err_lo.append(lookup_lo[idx])
                final_err_hi.append(lookup_hi[idx])
            else:
                final_mean.append(np.nan)
                final_err_lo.append(0)
                final_err_hi.append(0)
                
        return np.array(final_mean), np.vstack([final_err_lo, final_err_hi])

    baseline_s1_mean, baseline_s1_err = extract_data("baseline", 1)
    baseline_s5_mean, baseline_s5_err = extract_data("baseline", 5)
    textalign_s1_mean, textalign_s1_err = extract_data("textalign_llm", 1)
    textalign_s5_mean, textalign_s5_err = extract_data("textalign_llm", 5)

# x positions (3 budgets)
x = np.arange(3)
x_labels = ["1 sess", "2 sess", "40 sess"]  # (was 1h/2h/full)

# =========================
# Plot
# =========================
fig, axes = plt.subplots(
    1, 2,
    figsize=(10.8, 4.2),
    sharey=True,
    gridspec_kw={"wspace": 0.08}
)

def plot_panel(ax, title, s1_mean, s1_err, s5_mean, s5_err):
    ax.set_title(title, pad=8, fontweight="normal", color=COL_TEXT)

    # grid
    ax.grid(True, axis="y", color=COL_GRID, linewidth=1.0)
    ax.grid(True, axis="x", color=COL_GRID, linewidth=0.8, alpha=0.8)

    # clean spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(COL_AXIS)
    ax.spines["bottom"].set_color(COL_AXIS)
    ax.tick_params(colors=COL_TEXT)

    # Lines + markers (face uses your palette, stroke uses darkened tone)
    lw = 2.6
    ms = 7.5
    cap = 3.5

    # Filter NaNs for plotting lines (otherwise line breaks)
    # Mask valid data points
    valid_s1 = ~np.isnan(s1_mean)
    valid_s5 = ~np.isnan(s5_mean)

    h1 = ax.errorbar(
        x[valid_s1], s1_mean[valid_s1], 
        yerr=s1_err[:, valid_s1] if s1_err.ndim > 1 else s1_err[valid_s1],
        fmt="o-", lw=lw, ms=ms,
        color=LINE_S1,
        markerfacecolor=COL_S1, markeredgecolor=COL_AXIS, markeredgewidth=1.0,
        ecolor=LINE_S1, elinewidth=1.6, capsize=cap, capthick=1.6,
        label="S1"
    )

    h2 = ax.errorbar(
        x[valid_s5], s5_mean[valid_s5], 
        yerr=s5_err[:, valid_s5] if s5_err.ndim > 1 else s5_err[valid_s5],
        fmt="s-", lw=lw, ms=ms,
        color=LINE_S5,
        markerfacecolor=COL_S5, markeredgecolor=COL_AXIS, markeredgewidth=1.0,
        ecolor=LINE_S5, elinewidth=1.6, capsize=cap, capthick=1.6,
        label="S5"
    )

    ax.set_xticks(x, x_labels)
    ax.set_xlabel("Data (sessions)", color=COL_TEXT)
    return h1, h2

h1, h2 = plot_panel(
    axes[0], "baseline",
    baseline_s1_mean, baseline_s1_err,
    baseline_s5_mean, baseline_s5_err
)
plot_panel(
    axes[1], "textalign_llm",
    textalign_s1_mean, textalign_s1_err,
    textalign_s5_mean, textalign_s5_err
)

axes[0].set_ylabel("Hard-2AFC", color=COL_TEXT)

# y range: keep consistent and readable (don’t change meaning)
axes[0].set_ylim(0.83, 1.00)
axes[0].set_yticks([0.85, 0.90, 0.95, 1.00])

# Suptitle (slightly smaller than main paper fig, but still clear)
fig.suptitle("Efficiency curve (shared982) - TwoAFC Hard", y=1.05, fontsize=16, fontweight="normal", color=COL_TEXT)

# One shared legend at bottom
fig.legend(
    handles=[h1, h2],
    labels=["S1", "S5"],
    loc="lower center",
    ncol=2,
    frameon=False,
    # Adjust bbox_to_anchor to change the legend position (x, y)
    bbox_to_anchor=(0.5, -0.09)
)

# Tight layout with extra bottom room for legend
plt.tight_layout(rect=[0, 0.06, 1, 0.98])

# Save (do NOT overwrite original)
out_dir = "/mnt/work/repos/TextAlign-mindeye2/figures"
os.makedirs(out_dir, exist_ok=True)
out_png = os.path.join(out_dir, "Fig_efficiency_twoafc_hard_v2.png")
out_pdf = os.path.join(out_dir, "Fig_efficiency_twoafc_hard_v2.pdf")
plt.savefig(out_png, dpi=300, bbox_inches="tight")
plt.savefig(out_pdf, bbox_inches="tight")
plt.close()
print("Saved:", out_png)
print("Saved:", out_pdf)
