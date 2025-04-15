# %%
import glob

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
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
def analyze_sequence_lengths(filename_pattern, eos_token=50256):
    """Analyze sequence lengths in the dataset by counting tokens between EOS markers"""
    files = sorted(glob.glob(filename_pattern))
    assert len(files) > 0, f'No files found matching pattern: {filename_pattern}'

    sequence_lengths = []

    for file in tqdm(files):
        print(f'Processing {file}...')
        tokens = _load_data_shard(file)

        # Find positions of EOS tokens
        eos_positions = np.where(tokens == eos_token)[0]

        # Calculate sequence lengths
        if len(eos_positions) > 0:
            # First sequence starts at position 0
            seq_lens = np.diff(np.concatenate([[0], eos_positions]))
            sequence_lengths.extend(seq_lens.tolist())

            # Add the last sequence if it doesn't end with EOS
            if eos_positions[-1] < len(tokens) - 1:
                sequence_lengths.append(len(tokens) - eos_positions[-1] - 1)

    return sequence_lengths


# %%
# Set the path to your data files
data_path = '../data/fineweb10B/fineweb_train_*.bin'

# Analyze sequence lengths
sequence_lengths = analyze_sequence_lengths(data_path)

# %%
# Print summary statistics
print(f'Total sequences: {len(sequence_lengths)}')
print(f'Mean sequence length: {np.mean(sequence_lengths):.2f}')
print(f'Median sequence length: {np.median(sequence_lengths):.2f}')
print(f'Min sequence length: {np.min(sequence_lengths)}')
print(f'Max sequence length: {np.max(sequence_lengths)}')
print(f'95th percentile: {np.percentile(sequence_lengths, 95):.2f}')
print(f'99th percentile: {np.percentile(sequence_lengths, 99):.2f}')

# %%
# Plot histogram of sequence lengths
sns.set(style='whitegrid')
plt.figure(figsize=(10, 6), facecolor='none')
sns.histplot(data=sequence_lengths, bins=np.arange(0, 10000, 200), kde=True)
plt.title('Distribution of Sequence Lengths (first 500M tokens in training set)')
plt.xlabel('Sequence Length (tokens)')
plt.ylabel('Frequency')
plt.xlim(0, 10000)  # Limit x-axis to 10k

# Add label for sequences around 10k
bucket_9800_10000 = sum((length >= 9800) & (length < 10000) for length in sequence_lengths)
# Create a transparent label with an arrow pointing to the bucket
plt.annotate(
    f'n={bucket_9800_10000}',
    xy=(9900, plt.gca().get_ylim()[1] * 0.01),  # Arrow tip position (middle of bucket)
    xytext=(9500, plt.gca().get_ylim()[1] * 0.07),  # Text position
    bbox=dict(facecolor='none', edgecolor='gray', alpha=0.8),  # Transparent box
    ha='center',
    arrowprops=dict(arrowstyle='->', color='gray', alpha=0.8),
)  # Add arrow

plt.gca().set_facecolor('none')
plt.savefig('plots/sequence_length_histogram.png', facecolor=plt.gca().get_facecolor(), bbox_inches='tight', dpi=300)
plt.show()

# %%
# Plot CDF of sequence lengths
sns.set(style='whitegrid')
plt.figure(figsize=(10, 6), facecolor='none')
sorted_lengths = np.sort(sequence_lengths)
cumulative = np.arange(1, len(sorted_lengths) + 1) / len(sorted_lengths)
plt.plot(sorted_lengths, cumulative, linewidth=2)
plt.title('Cumulative Distribution of Sequence Lengths')
plt.xlabel('Sequence Length (tokens)')
plt.ylabel('Cumulative Probability')
plt.xlim(0, 10000)  # Limit x-axis to 10k
plt.gca().set_facecolor('none')
plt.savefig('plots/sequence_length_cdf.png', facecolor=plt.gca().get_facecolor(), bbox_inches='tight', dpi=300)
plt.show()

# %%
