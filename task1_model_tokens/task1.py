from pathlib import Path
import csv
import json
import tiktoken

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
REPORTS = ROOT / "reports"
DATA.mkdir(exist_ok=True)
REPORTS.mkdir(exist_ok=True)

# Five text types required by the assessment
texts = [
    ("Sentence", "Large language models can generate text."),
    ("Paragraph", "Machine learning learns patterns from data. "
     "Models use these patterns to make predictions. "
     "The quality of predictions depends on the data."),
    ("Page", ("Artificial intelligence is used in many industries. "
     "It supports automation, data analysis, and customer service. ") * 40),
    ("JSON", json.dumps({
        "name": "Ravi",
        "skills": ["Python", "SQL", "Machine Learning"],
        "experience": 2
    })),
    ("Tamil", "வணக்கம்! நீங்கள் எப்படி இருக்கிறீர்கள்?")
]

encoding = tiktoken.get_encoding("cl100k_base")
token_rows = []

for name, text in texts:
    tokens = encoding.encode(text)
    words = len(text.split())
    token_count = len(tokens)

    token_rows.append({
        "text_type": name,
        "words": words,
        "tokens": token_count,
        "words_per_token": round(words / token_count, 3)
        if token_count else 0,
        "fits_8k": token_count <= 8000,
        "fits_128k": token_count <= 128000
    })

with open(REPORTS / "task1_token_results.csv", "w",
          newline="", encoding="utf-8-sig") as file:
    writer = csv.DictWriter(file, fieldnames=token_rows[0].keys())
    writer.writeheader()
    writer.writerows(token_rows)

# Prices are examples in USD per 1 million tokens.
# Replace them with the current prices for your chosen model.
def estimate_cost(input_tokens, output_tokens, price_in, price_out):
    input_cost = input_tokens / 1_000_000 * price_in
    output_cost = output_tokens / 1_000_000 * price_out
    return round(input_cost + output_cost, 4)

chats_per_day = 1000
days_per_month = 30
monthly_input = chats_per_day * 1500 * days_per_month
monthly_output = chats_per_day * 400 * days_per_month

price_in = 0.15
price_out = 0.60

cost = estimate_cost(
    monthly_input, monthly_output, price_in, price_out
)

cost_report = {
    "monthly_input_tokens": monthly_input,
    "monthly_output_tokens": monthly_output,
    "example_input_price_per_million": price_in,
    "example_output_price_per_million": price_out,
    "example_monthly_cost_usd": cost,
    "note": "Example prices only. Verify official current pricing."
}

with open(REPORTS / "task1_cost_estimate.json", "w",
          encoding="utf-8") as file:
    json.dump(cost_report, file, indent=4)

# Basic family comparison. Verify context sizes against official
# documentation for the exact model you choose.
comparison = [
    ["GPT", "OpenAI", "Closed weights", "Varies by model",
     "General-purpose assistant and tools"],
    ["Claude", "Anthropic", "Closed weights", "Varies by model",
     "Writing, analysis, and coding"],
    ["Llama", "Meta", "Open-weight models", "Varies by model",
     "Self-hosting and customization"],
    ["Gemini", "Google", "Closed weights", "Varies by model",
     "Multimodal tasks and long-context use"]
]

with open(REPORTS / "task1_model_comparison.csv", "w",
          newline="", encoding="utf-8-sig") as file:
    writer = csv.writer(file)
    writer.writerow([
        "Family", "Owner", "Weight access",
        "Context window", "Strengths"
    ])
    writer.writerows(comparison)

print("TASK 1 COMPLETED")
print("Token results:", REPORTS / "task1_token_results.csv")
print("Cost estimate:", REPORTS / "task1_cost_estimate.json")
print("Model comparison:", REPORTS / "task1_model_comparison.csv")
print(json.dumps(cost_report, indent=4))