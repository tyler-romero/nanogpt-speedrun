# %%
import glob

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


# %%
def _load_data_shard(filename):
    with open(filename, 'rb') as f:
        # first read the header, which is 256 int32 integers (4 bytes each)
        header = np.frombuffer(f.read(256 * 4), dtype=np.int32)
        assert header[0] == 20240520, 'magic number mismatch in the data .bin file'
        assert header[1] == 1, 'unsupported version'
        ntok = header[2]  # number of tokens (claimed)
        # the rest of it are tokens, stored as uint16
        tokens = np.frombuffer(f.read(), dtype=np.uint16)
    assert len(tokens) == ntok, 'number of tokens read does not match header?'
    return tokens


# %%
def analyze_token_distribution(filename_pattern):
    """Analyze the distribution of tokens in the dataset"""
    files = sorted(glob.glob(filename_pattern))
    assert len(files) > 0, f'No files found matching pattern: {filename_pattern}'

    token_counts = {}

    for file in tqdm(files):
        print(f'Processing {file}...')
        tokens = _load_data_shard(file)

        # Count occurrences of each token
        unique, counts = np.unique(tokens, return_counts=True)
        for token, count in zip(unique, counts):
            token_counts[token] = token_counts.get(token, 0) + count

    # Convert to sorted list of (token, count) tuples
    token_distribution = sorted(token_counts.items(), key=lambda x: x[1], reverse=True)

    # Print the number of unique tokens identified
    unique_token_count = len(token_counts)
    print(f'Number of unique tokens identified: {unique_token_count}')

    return token_distribution, token_counts


# %%
# Set the path to your data files
data_path = '../data/fineweb10B/fineweb_train_*.bin'

# Analyze token distribution
token_distribution, token_counts = analyze_token_distribution(data_path)

# %%
# Plot histogram of top tokens
plt.figure(figsize=(12, 6))
top_n = 500  # Show top 500 tokens
tokens = [t[0] for t in token_distribution[:top_n]]
total_tokens = sum(token_counts.values())
percentages = [t[1] / total_tokens * 100 for t in token_distribution[:top_n]]
cumulative_percentages = np.cumsum(percentages)

# Create a figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Left plot: Grouped bars for better visualization
group_size = 50  # Group tokens into buckets of 50
n_groups = top_n // group_size
grouped_percentages = []
for i in range(n_groups):
    start_idx = i * group_size
    end_idx = min((i + 1) * group_size, top_n)
    grouped_percentages.append(sum(percentages[start_idx:end_idx]))

# Plot grouped bars
ax1.bar(np.arange(n_groups), grouped_percentages, width=0.8)
ax1.set_title(f'Distribution of Top {top_n} Tokens (Grouped by {group_size})')
ax1.set_xlabel(f'Token Groups (each group = {group_size} tokens)')
ax1.set_ylabel('Percentage of Total Dataset (%)')
ax1.set_xticks(np.arange(n_groups))
ax1.set_xticklabels([f'{i * group_size + 1}-{min((i + 1) * group_size, top_n)}' for i in range(n_groups)])
plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')

# Right plot: Cumulative distribution
ax2.plot(np.arange(top_n), cumulative_percentages, color='b', linewidth=2)
ax2.set_title('Cumulative Token Distribution')
ax2.set_xlabel('Number of Top Tokens')
ax2.set_ylabel('Cumulative Percentage (%)')
ax2.grid(True, linestyle='--', alpha=0.7)
ax2.set_ylim(0, 100)

# Add annotations for key percentages
key_percentages = [50, 75, 90, 95]
for pct in key_percentages:
    # Find the index where cumulative percentage exceeds the target
    idx = np.searchsorted(cumulative_percentages, pct)
    if idx < len(cumulative_percentages):
        ax2.axhline(y=pct, color='r', linestyle='--', alpha=0.5)
        ax2.axvline(x=idx, color='r', linestyle='--', alpha=0.5)
        ax2.annotate(f'{pct}% at {idx} tokens', xy=(idx, pct), xytext=(idx + 20, pct + 5), arrowprops=dict(arrowstyle='->'))

plt.tight_layout()
plt.savefig('plots/token_distribution.png', bbox_inches='tight', dpi=300)
plt.show()

# %%
# Print summary statistics
total_tokens = sum(token_counts.values())
unique_tokens = len(token_counts)
print(f'Total tokens: {total_tokens}')
print(f'Unique tokens: {unique_tokens}')
print('Top 10 most common tokens:')
for i, (token, count) in enumerate(token_distribution[:10]):
    print(f'  {i + 1}. Token {token}: {count} occurrences ({count / total_tokens * 100:.2f}%)')

# %%
# Plot token frequency distribution (log scale)
plt.figure(figsize=(12, 6))
max_tok = 50020
token_ranks = np.arange(1, min(max_tok, len(token_distribution)) + 1)
token_freqs = [t[1] for t in token_distribution[:max_tok]]

# Plot actual token frequencies
plt.loglog(token_ranks, token_freqs, 'k.', markersize=2, label='Observed frequencies')

# Fit Zipf's law (power law) to the data
# Zipf's law: f(r) = C / ((r + b)^a) where r is rank, f is frequency, a, b, and C are parameters
from scipy.optimize import curve_fit


def zipf_law(r, C, a, b):
    return C / ((r + b) ** a)


# Use non-zero ranks to avoid log(0) issues
valid_indices = np.array(token_freqs) > 0
valid_ranks = token_ranks[valid_indices]
valid_freqs = np.array(token_freqs)[valid_indices]

# Initial parameter guesses - important for better fitting
initial_params = [valid_freqs[0], 1.0, 2.7]  # C ≈ first frequency, a ≈ 1, b ≈ 2.7

# Fit the curve with bounds to keep parameters reasonable
params, _ = curve_fit(zipf_law, valid_ranks, valid_freqs, p0=initial_params, bounds=([0, 0.01, 0], [np.inf, 10, 20]))
C, a, b = params

# Generate the fitted line
fitted_curve = zipf_law(token_ranks, C, a, b)

# Plot the fitted line
plt.loglog(token_ranks, fitted_curve, 'r-', linewidth=1.5, label=f"Zipf's law fit: f(r) = {C:.1e} / ((r + {b:.2f})^{a:.3f})")
plt.title("Token Frequency Distribution with Zipf's Law Fit (log-log scale)")
plt.xlabel('Token Rank')
plt.ylabel('Frequency')
plt.grid(True, which='major', ls='-', alpha=0.5)
plt.grid(False, which='minor')  # Turn off minor grid lines
plt.legend()
plt.tight_layout()
plt.savefig('plots/token_frequency_loglog.png', bbox_inches='tight', dpi=300)
plt.show()

# %%
