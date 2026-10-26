from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
REPORTS.mkdir(exist_ok=True)

np.random.seed(42)

def softmax(values):
    values = values - np.max(values, axis=-1, keepdims=True)
    exp_values = np.exp(values)
    return exp_values / np.sum(exp_values, axis=-1, keepdims=True)

# Part 1: Single-head self-attention
X = np.random.rand(4, 4)
WQ = np.random.rand(4, 4)
WK = np.random.rand(4, 4)
WV = np.random.rand(4, 4)

Q = X @ WQ
K = X @ WK
V = X @ WV

scores = Q @ K.T / np.sqrt(K.shape[-1])
attention = softmax(scores)
attention_output = attention @ V

print("Attention matrix:")
print(np.round(attention, 3))
print("Row sums:", np.round(attention.sum(axis=1), 3))

with open(REPORTS / "task3_attention_matrix.csv", "w",
          newline="", encoding="utf-8-sig") as file:
    writer = csv.writer(file)
    writer.writerows(attention)

# Part 2: Two attention heads and output projection
def make_head(data, seed):
    rng = np.random.default_rng(seed)
    d = data.shape[1]
    q = data @ rng.random((d, d))
    k = data @ rng.random((d, d))
    v = data @ rng.random((d, d))
    weights = softmax(q @ k.T / np.sqrt(d))
    return weights @ v

head1 = make_head(X, 1)
head2 = make_head(X, 2)

combined = np.concatenate((head1, head2), axis=1)
W_output = np.random.rand(combined.shape[1], 4)
multi_head_output = combined @ W_output

print("Multi-head output shape:", multi_head_output.shape)

# Part 3: Sinusoidal positional encoding
def positional_encoding(length, dimensions):
    positions = np.arange(length)[:, None]
    dims = np.arange(dimensions)[None, :]
    angles = positions / np.power(
        10000, (2 * (dims // 2)) / dimensions
    )

    result = np.zeros((length, dimensions))
    result[:, 0::2] = np.sin(angles[:, 0::2])
    result[:, 1::2] = np.cos(angles[:, 1::2])
    return result

encoding = positional_encoding(10, 8)

plt.figure()
plt.imshow(encoding, aspect="auto")
plt.xlabel("Dimension")
plt.ylabel("Position")
plt.title("Sinusoidal Positional Encoding")
plt.colorbar()
plt.tight_layout()
plt.savefig(REPORTS / "task3_positional_encoding.png")
plt.close()

# Part 4: Hugging Face pipelines
# The first run downloads models and requires internet access.
try:
    from transformers import pipeline

    fill_mask = pipeline(
        "fill-mask",
        model="google-bert/bert-base-uncased"
    )
    print("\nBERT fill-mask:")
    print(fill_mask("Python is a [MASK] language.")[0])

    text_generator = pipeline(
        "text-generation",
        model="openai-community/gpt2"
    )
    print("\nGPT-2 text generation:")
    print(text_generator(
        "Artificial intelligence is",
        max_new_tokens=25,
        do_sample=False,
        pad_token_id=50256
    )[0]["generated_text"])

except Exception as error:
    print("\nHugging Face pipeline did not run.")
    print("Check the installation, internet, and error:")
    print(error)

# Save a short model comparison for the report.
comparison = [
    ["Encoder", "BERT", "Good for understanding and fill-mask tasks."],
    ["Decoder", "GPT-2", "Good for generating the next tokens."],
    ["RNN", "Recurrent network",
     "Processes sequences step by step; less parallelizable."]
]

with open(REPORTS / "task3_model_comparison.csv", "w",
          newline="", encoding="utf-8-sig") as file:
    writer = csv.writer(file)
    writer.writerow(["Type", "Example", "Difference"])
    writer.writerows(comparison)

print("\nTASK 3 COMPLETED")
print("Check the reports folder.")