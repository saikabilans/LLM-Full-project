from pathlib import Path
import sys
import csv
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from openrouter_client import client, MODEL

REPORTS = ROOT / "reports"
REPORTS.mkdir(exist_ok=True)

experiments = [
    ("Summarization",
     "Summarize in one sentence: AI can analyze data, recognize "
     "patterns, generate text, and assist with repetitive tasks."),
    ("Classification",
     "Classify this ticket as Billing, Technical, or Account: "
     "I was charged twice for my subscription."),
    ("Extraction",
     "Extract name and skills as JSON from this text: "
     "Name: Saikabilan. Skills: Python and SQL."),
    ("Translation",
     "Translate into Tamil: Good morning. How can I help you?"),
    ("Code generation",
     "Write a Python function that returns the sum of two numbers."),
    ("Recent information",
     "Who won the most recent major international cricket match today? "
     "If you cannot verify current information, say so."),
    ("Hallucination without safeguard",
     "Give the journal and DOI for the fictional paper "
     "'Quantum Mango Networks for Reliable AI' by Dr Alex Example."),
    ("Hallucination with safeguard",
     "Give the journal and DOI for the fictional paper "
     "'Quantum Mango Networks for Reliable AI' by Dr Alex Example. "
     "If you cannot verify that it exists, say you do not know."),
    ("Middle fact",
     "What is the secret code in this text? " +
     ("The weather is sunny. General information follows. " * 30) +
     "The secret code is ORANGE-4729. " +
     ("More general information follows. " * 30)),
    ("Arithmetic",
     "A shop has 20 apples, sells 7, and receives 5 more. "
     "How many apples are left?"),
    ("Logic",
     "All roses are flowers. Some flowers fade quickly. "
     "Does that prove that some roses fade quickly? Explain briefly.")
]

results = []

for name, prompt in experiments:
    print("\nRunning:", name)
    start = time.perf_counter()

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Be accurate. Do not invent facts. "
                        "State uncertainty when appropriate."
                    )
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )

        answer = response.choices[0].message.content or ""
        usage = response.usage
        tokens = usage.total_tokens if usage else "Not provided"
        status = "success"

    except Exception as error:
        answer = str(error)
        tokens = "Not provided"
        status = "error"

    elapsed = round(time.perf_counter() - start, 2)

    results.append({
        "test": name,
        "prompt": prompt,
        "answer": answer,
        "tokens": tokens,
        "seconds": elapsed,
        "status": status
    })

    print(answer)
    print("Tokens:", tokens, "| Seconds:", elapsed)

with open(REPORTS / "task2_results.csv", "w",
          newline="", encoding="utf-8-sig") as file:
    writer = csv.DictWriter(file, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)

safeguards = [
    "Validate model output before using it.",
    "Check important facts against trusted sources.",
    "Use structured output and schema validation.",
    "Protect private data and API credentials.",
    "Require human review for high-impact decisions."
]

with open(REPORTS / "task2_safeguards.txt", "w",
          encoding="utf-8") as file:
    file.write("\n".join(safeguards))

print("\nTASK 2 COMPLETED")
print("Check reports/task2_results.csv")
print("Check reports/task2_safeguards.txt")