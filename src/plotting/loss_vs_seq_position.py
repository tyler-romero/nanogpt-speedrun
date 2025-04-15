# %%
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from datasets import load_dataset
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# %%
# Set up the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load model and tokenizer
model_name = 'gpt2'  # You can change this to other GPT2 variants like "gpt2-medium", "gpt2-large", etc.
model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
model.eval()  # Set model to evaluation mode
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Set padding token to be the EOS token
tokenizer.pad_token = tokenizer.eos_token

# %%
# Sample texts of different lengths to evaluate
# Load FineWeb dataset
# Load a small sample from FineWeb
dataset = load_dataset('HuggingFaceFW/fineweb', split='train', streaming=True)
texts = []

# Take a few examples with different lengths
count = 0
for example in dataset:
    texts.append(example['text'][:10000])
    count += 1
    if count == 100:
        break


# %%
# Function to calculate loss at each position
def calculate_position_loss(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=1024).to(device)
    input_ids = inputs['input_ids']
    position_losses = []
    with torch.no_grad():
        outputs = model(input_ids=input_ids, labels=input_ids)
        shift_logits = outputs.logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        losses = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        position_losses = losses.view(1, -1).squeeze().cpu().numpy()

    return position_losses


# %%
# Calculate losses for each text
all_losses = []
max_length = 0

for text in texts:
    losses = calculate_position_loss(text)
    all_losses.append(losses)
    max_length = max(max_length, len(losses))


# %%
# Pad shorter sequences to max_length
padded_losses = []
for losses in all_losses:
    padded = np.pad(losses, (0, max_length - len(losses)), 'constant', constant_values=np.nan)
    padded_losses.append(padded)

# Calculate mean, ignoring NaN values
mean_losses = np.nanmean(padded_losses, axis=0)
positions = np.arange(1, max_length + 1)

# Plot loss vs sequence position
sns.set(style='whitegrid')
plt.figure(figsize=(10, 6), facecolor='none')
plt.plot(positions, mean_losses, linewidth=2)
plt.title('GPT-2 Average Loss vs. Sequence Position')
plt.xlabel('Sequence Position (tokens)')
plt.ylabel('Average Loss (in nats)')
plt.xlim(0, max_length)
plt.gca().set_facecolor('none')
plt.savefig('plots/avg_loss_vs_seq_position.png', facecolor=plt.gca().get_facecolor(), bbox_inches='tight', dpi=300)
plt.show()

# %%
# Plot smoothed version for better trend visibility
sns.set(style='whitegrid')
plt.figure(figsize=(10, 6), facecolor='none')
# Apply rolling average for smoothing
window_size = min(50, len(mean_losses) // 10)  # Adaptive window size
smoothed_losses = np.convolve(mean_losses, np.ones(window_size) / window_size, mode='valid')
smoothed_positions = positions[window_size - 1 :]
plt.plot(smoothed_positions, smoothed_losses, linewidth=2)
plt.title('Smoothed GPT-2 Average Loss vs. Sequence Position')
plt.xlabel('Sequence Position (tokens)')
plt.ylabel('Average Loss (in nats)')
plt.xlim(0, max_length)
plt.gca().set_facecolor('none')
plt.savefig('plots/smoothed_avg_loss_vs_seq_position.png', facecolor=plt.gca().get_facecolor(), bbox_inches='tight', dpi=300)
plt.show()

# %%
