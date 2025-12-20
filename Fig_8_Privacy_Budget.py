import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from pathlib import Path

# Load data
csv_conv = Path('convergence.csv')
csv_budget = Path('privacy_budget.csv')

# Load and merge
if csv_conv.exists() and csv_budget.exists():
    df_conv = pd.read_csv(csv_conv)
    df_budget = pd.read_csv(csv_budget)
    if 'epsilon' in df_conv.columns:
        df_conv = df_conv.drop(columns=['epsilon'])
    df = pd.merge(df_budget, df_conv, on='episode', how='inner')
else:
    raise FileNotFoundError("Required files not found.")

# --- STYLE FUNCTION (Exactly as requested) ---
def style():
    plt.rcParams.update({
        "font.family":"serif",
        "axes.labelsize":22,
        "axes.edgecolor":"black",
        "xtick.labelsize":18,
        "xtick.color":"black",
        "xtick.major.width":1.5,
        "xtick.major.size":6,
        "ytick.labelsize":18,
        "ytick.color":"black",
        "ytick.major.width":1.5,
        "ytick.major.size":6,
        "lines.linewidth":2.5,
        "lines.markersize":6,
        "legend.fontsize":18,
        "legend.frameon":True,
        "legend.framealpha":1.0,
        "legend.edgecolor":"black",
    })

def figax(): return plt.subplots(figsize=(10,7), dpi=600)

def colours(n): 
    import numpy as np
    return plt.cm.viridis(np.linspace(0.2,0.8,max(1,n)))

# Apply style
style()

# Data prep
x = df['episode']
y_budget = df['epsilon']
y_sensing = df['mean_sensing_db']

# Plotting
fig1, ax1 = figax()

# Left Axis: Privacy Budget (using first color from palette)
# Added ms=4 to match your snippet's downsampling/marker style
ax1.plot(x, y_budget, color=colours(2)[0], marker="o", label=r"Privacy Budget ($\epsilon$)", lw=2.5, ms=4)

# Threshold Line: Horizontal at y=5
# Using Red to distinguish from the viridis palette
ax1.axhline(y=5, color='red', linestyle='-.', linewidth=2, label=r'Privacy Threshold ($\epsilon=5$)')

ax1.set_xlabel("Training Episode")
ax1.set_ylabel(r"Privacy Budget $\epsilon$")
ax1.grid(True, ls="--", alpha=0.3)
ax1.set_xlim(x.min(), x.max())

# Right Axis: Mean Sensing Metric
ax1b = ax1.twinx()
# Using second color from palette
ax1b.plot(x, y_sensing, color=colours(2)[1], marker="s", label="Mean Sensing Metric (dB)", lw=2.5, ms=4)
ax1b.set_ylabel("Mean Sensing Metric (dB)")

# Combine legends
h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax1b.get_legend_handles_labels()
# 'lower right' is used in your snippet, but might overlap here. 
# 'center right' is often safer for dual-axis plots with rising trends.
ax1.legend(h1+h2, l1+l2, loc="center right") 

# Adjust margins (account for dual axis labels)
# Save using the house style margins
plt.tight_layout()
plt.subplots_adjust(left=0.075, right=0.9, top=0.975, bottom=0.125) 

plt.savefig('privacy_sensing_Fig_8.png', dpi=600)
plt.show()