import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path
import numpy as np

# =======================  HOUSE STYLE & UTILITIES  =======================

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
        "legend.fontsize":18, # Kept at 16 to match your previous request
        "legend.frameon":True,
        "legend.framealpha":1.0,
        "legend.edgecolor":"black",
    })

def finalize(fig, left=0.15, right=0.85, top=0.90, bottom=0.15,
             savepath=None, dpi=600, show=False):
    plt.subplots_adjust(left=left, right=right, top=top, bottom=bottom)
    if savepath:
        fig.savefig(savepath, dpi=dpi)
    if show:
        plt.show()

# =======================  MAIN LOGIC  =======================

style()

# Load data
csv_path = Path('convergence.csv')
lat_path = Path('latency_traces.csv')

# 1. Load Convergence Data
if csv_path.exists():
    df = pd.read_csv(csv_path)
else:
    raise FileNotFoundError("Please provide convergence.csv")

# 2. Load and Process Latency Data
if lat_path.exists():
    df_lat = pd.read_csv(lat_path)
    # Group by episode to get mean latency per episode
    lat_per_episode = df_lat.groupby('episode')['T_tot_ms'].mean().reset_index()
    lat_per_episode.rename(columns={'T_tot_ms': 'mean_latency'}, inplace=True)
    
    # Merge with main dataframe
    df = pd.merge(df, lat_per_episode, on='episode', how='left')
else:
    print("Warning: latency_traces.csv not found. Latency will not be plotted.")
    df['mean_latency'] = 0 

# Compute 20-episode moving averages
df['MA20_return'] = df['mean_return'].rolling(window=20, min_periods=1).mean()
df['MA20_latency'] = df['mean_latency'].rolling(window=20, min_periods=1).mean()

# Filter Data up to Episode 250
df = df[df['episode'] <= 250]

x = df['episode']
y_ret = df['mean_return']
y_ret_ma = df['MA20_return']
y_lat_ma = df['MA20_latency']

# Plotting
fig, ax1 = plt.subplots(figsize=(10, 7), dpi=600)

# --- Left Axis: Mean Return ---
# Raw mean return
p1, = ax1.plot(
    x, y_ret,
    linestyle='-',
    marker='o',
    markevery=10,
    color='tab:blue',
    alpha=0.4,
    label='Mean Return (Raw)'
)

# 20-Episode MA Return
p2, = ax1.plot(
    x, y_ret_ma,
    linestyle='-',
    color='red',
    linewidth=2.5,
    marker='s',
    markevery=15,
    label='Mean Return (MA20)'
)

ax1.set_xlabel('Training Episode')
ax1.set_ylabel(r'Mean return $\mathbb{E}[R_t]$')
ax1.grid(True, linestyle="--", alpha=0.3)
ax1.set_xlim([x.iloc[0], x.iloc[-1]])

# --- Right Axis: Latency ---
ax2 = ax1.twinx()

# 20-Episode MA Latency (Blue)
p3, = ax2.plot(
    x, y_lat_ma,
    linestyle='--',
    color='blue', 
    linewidth=2.5,
    label='Latency (MA20)'
)

ax2.set_ylabel('Avg. Latency (ms)', color='blue')
ax2.tick_params(axis='y', labelcolor='blue', colors='blue')
ax2.spines['right'].set_color('blue')
ax2.spines['right'].set_linewidth(2)

# Combine legends
lines = [p1, p2, p3]
labels = [l.get_label() for l in lines]
leg = ax1.legend(lines, labels, loc='center right', handlelength=1.6)
leg.get_frame().set_linewidth(1.5)
leg.get_frame().set_edgecolor("black")

# Save using the house style margins
plt.tight_layout()
plt.subplots_adjust(left=0.125, right=0.9, top=0.975, bottom=0.15) 
plt.savefig('convergence_Fig_7.png', dpi=600)
plt.show()