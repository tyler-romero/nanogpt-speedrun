# %%
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


# %%
def plot_attention_mask(mask, title):
    """Plot an attention mask with the given title."""
    sns.set(style='whitegrid')
    plt.figure(figsize=(10, 8))

    # Create heatmap with linewidths parameter to show cell boundaries
    ax = sns.heatmap(
        mask,
        cmap='YlGnBu_r',
        square=True,
        cbar=False,
        annot=False,
        vmin=0,
        vmax=1,
        linewidths=0.1,
        linecolor='black',  # Set line color
    )

    ax.set_title(title, fontsize=16, pad=20)

    # Add token labels on axes
    num_tokens = mask.shape[0]
    token_labels = [f'K{i}' for i in range(num_tokens)]
    query_labels = [f'Q{i}' for i in range(num_tokens)]

    ax.set_xticks(np.arange(num_tokens) + 0.5)
    ax.set_yticks(np.arange(num_tokens) + 0.5)
    ax.set_xticklabels(token_labels)
    ax.set_yticklabels(query_labels, rotation=0)  # Set rotation to 0 to make labels horizontal

    ax.set_xlabel('Key Tokens', fontsize=12, labelpad=10)
    ax.set_ylabel('Query Tokens', fontsize=12, labelpad=10)

    plt.tight_layout()
    plt.gca().set_facecolor('none')
    plt.show()


# plot_attention_mask(mask, 'Causal Attention Mask')


# %%
def create_causal_attention_mask(seq_len=8):
    """Create a causal attention mask where each token can attend to itself and previous tokens."""
    return np.tril(np.ones((seq_len, seq_len)))


def create_document_attention_mask(tokens_per_doc=[3, 2, 3]):
    """Create a document attention mask where tokens can only attend to tokens in the same document."""
    total_tokens = sum(tokens_per_doc)
    mask = np.zeros((total_tokens, total_tokens))

    start_idx = 0
    for doc_len in tokens_per_doc:
        end_idx = start_idx + doc_len
        mask[start_idx:end_idx, start_idx:end_idx] = 1
        start_idx = end_idx

    return mask


def create_sliding_window_attention_mask(seq_len=8, window_size=3):
    """Create a sliding window attention mask where each token can attend to the previous tokens in the window."""
    mask = np.zeros((seq_len, seq_len))
    for i in range(seq_len):
        # Each token can attend to itself and window_size-1 tokens before it
        start_idx = max(0, i - window_size + 1)
        mask[i, start_idx : i + 1] = 1
    return mask


# %%
seq_len = 16

# %%
mask = create_causal_attention_mask(seq_len=seq_len)
plot_attention_mask(mask, 'Causal Attention Mask')

# %%
mask = create_document_attention_mask([7, 4, 5])
plot_attention_mask(mask, 'Document Attention Mask (3 documents)')


# %%
def and_masks(*masks):
    result = masks[0].copy()
    for mask in masks[1:]:
        result = result * mask  # Element-wise multiplication for AND
    return (result > 0).astype(np.bool)


# %%
causal_mask = create_causal_attention_mask(seq_len=seq_len)
document_mask = create_document_attention_mask([7, 4, 5])
combined_mask = and_masks(causal_mask, document_mask)

plot_attention_mask(combined_mask, 'Causal + Document Attention Mask')

# %%
sliding_mask = create_sliding_window_attention_mask(seq_len=seq_len, window_size=5)
all_mask = and_masks(causal_mask, document_mask, sliding_mask)
plot_attention_mask(all_mask, 'Causal + Document + Sliding Window Attention Mask (3 documents)')

# %%


def plot_multiple_attention_masks(masks, titles, figsize=(18, 6)):
    """Plot multiple attention masks side by side with the given titles."""
    sns.set(style='whitegrid')
    n_masks = len(masks)
    fig, axes = plt.subplots(1, n_masks, figsize=figsize)

    if n_masks == 1:
        axes = [axes]

    for i, (mask, title, ax) in enumerate(zip(masks, titles, axes)):
        # Create heatmap with linewidths parameter to show cell boundaries
        sns.heatmap(mask, cmap=sns.color_palette(['#1d4f60', '#c4e6c3']), square=True, cbar=False, annot=False, vmin=0, vmax=1, linewidths=0.05, linecolor='black', ax=ax)

        ax.set_title(title, fontsize=16, pad=10)

        # Add token labels on axes
        num_tokens = mask.shape[0]
        token_labels = [f'K{i}' for i in range(num_tokens)]
        query_labels = [f'Q{i}' for i in range(num_tokens)]

        ax.set_xticks(np.arange(num_tokens) + 0.5)
        ax.set_yticks(np.arange(num_tokens) + 0.5)
        ax.set_xticklabels(token_labels, fontsize=10)
        ax.set_yticklabels(query_labels, rotation=0, fontsize=10)

        ax.set_xlabel('Key Tokens', labelpad=5)
        ax.set_ylabel('Query Tokens', labelpad=5)

        if i == 0:
            # Add legend to indicate masked vs. visible tokens
            from matplotlib.patches import Patch

            legend_elements = [Patch(facecolor='#1d4f60', label='Masked (Cannot Attend)'), Patch(facecolor='#c4e6c3', label='Visible (Can Attend)')]
            ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.15), frameon=True, ncol=2, fontsize=9)

    plt.tight_layout()
    return fig


# Example usage:
seq_len = 16
causal_mask = create_causal_attention_mask(seq_len=seq_len)
document_mask = create_document_attention_mask([7, 4, 5])
sliding_mask = create_sliding_window_attention_mask(seq_len=seq_len, window_size=5)

masks = [causal_mask, document_mask, sliding_mask]
titles = ['Causal Attention', 'Document Attention (3 documents)', 'Sliding Window Attention (window = 5)']

fig = plot_multiple_attention_masks(masks, titles)
plt.gca().set_facecolor('none')
fig.set_facecolor('none')
# Increase DPI for higher resolution and use lossless format
plt.savefig('plots/attention_masks.png', facecolor=plt.gca().get_facecolor(), bbox_inches='tight', dpi=600, format='png', transparent=True)
# Alternative: save as vector format for perfect scaling
plt.savefig('plots/attention_masks.svg', facecolor=plt.gca().get_facecolor(), bbox_inches='tight', format='svg', transparent=True)
plt.show()

# %%
# Create combined attention mask with all three types
combined_all = and_masks(causal_mask, sliding_mask)
combined_all = and_masks(combined_all, document_mask)

# Plot the combined mask
fig_combined = plot_multiple_attention_masks([combined_all], ['Causal + Sliding Window + Document'])
plt.gca().set_facecolor('none')
fig_combined.set_facecolor('none')
plt.savefig('plots/combined_attention_masks.png', facecolor=plt.gca().get_facecolor(), bbox_inches='tight', dpi=300)
plt.show()

# %%
