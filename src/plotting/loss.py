# %%
import re

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# %%
def parse_log_file(file_path):
    """Parse log file and extract validation loss and training time data."""
    val_losses = []
    train_times = []

    with open(file_path, 'r') as f:
        for line in f:
            if 'val_loss' in line:
                # Extract values using regex
                val_loss_match = re.search(r'val_loss:([\d.]+)', line)
                train_time_match = re.search(r'train_time:(\d+)ms', line)

                if val_loss_match and train_time_match:
                    val_losses.append(float(val_loss_match.group(1)))
                    # Convert milliseconds to minutes
                    train_times.append(float(train_time_match.group(1)) / (1000 * 60))

    return pd.DataFrame({'train_time_minutes': train_times, 'val_loss': val_losses})


# %%
baseline_log_file = '../../logs/4c627c0d-029c-4f8a-bd18-40f99b43b22e.txt'
data_baseline = parse_log_file(baseline_log_file)
arch_tweaks_log_file = '../../logs/14fcdb07-443d-4d1c-b307-061bc4bd2cd6.txt'
data_arch_tweaks = parse_log_file(arch_tweaks_log_file)
muon_log_file = '../../logs/59951c17-fbe5-4577-a1bc-6dc0c1802d2e.txt'
data_muon = parse_log_file(muon_log_file)
dataloading_tweaks_log_file = '../../logs/08047f73-cb01-4f47-a901-de901b2a6b6e.txt'
data_dataloading_tweaks = parse_log_file(dataloading_tweaks_log_file)

# %%
sns.set(style='whitegrid')
plt.figure(figsize=(10, 6), facecolor='none')
ax = sns.lineplot(x='train_time_minutes', y='val_loss', data=data_baseline, linewidth=2, label='#1 Baseline')
ax = sns.lineplot(x='train_time_minutes', y='val_loss', data=data_arch_tweaks, linewidth=2, label='#2.1 Architecture Tweaks')
ax = sns.lineplot(x='train_time_minutes', y='val_loss', data=data_muon, linewidth=2, label='#2.2 Muon Optimizer')
ax = sns.lineplot(x='train_time_minutes', y='val_loss', data=data_dataloading_tweaks, linewidth=2, label='#2.3 Dataloading Tweaks')
ax.set_xlabel('Wallclock Time (2xRTX4090-minutes)')
ax.set_ylabel('FineWeb Val Loss (bits)')
ax.set_ylim(3.25, 4.0)
ax.set_xlim(0, ax.get_xlim()[1])
ax.axhline(3.28, color='r', linestyle='--', label='Speedrun Target')
plt.gca().set_facecolor('none')
plt.legend()
plt.savefig('plots/2p3_loss_plot.png', facecolor=plt.gca().get_facecolor(), bbox_inches='tight', dpi=300)
plt.show()

# %%
