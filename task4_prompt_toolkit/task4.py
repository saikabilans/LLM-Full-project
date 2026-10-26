from pathlib import Path
import sys
import csv
import json
import re
from pydantic import BaseModel, ValidationError

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from openrouter_client import client, MODEL

DATA = ROOT / "data"
PROMPTS = ROOT / "prompts"
REPORTS = ROOT / "reports"

for folder in (DATA, PROMPTS, REPORTS):
    folder.mkdir(exist_ok=True)

# Create 15 labeled tickets if the CSV does not exist.
tickets_path = DATA / "tickets.csv"

tickets = [
    ("Charged twice for subscription", "Billing", "High"),
    ("Need a refund for my order", "Billing", "High"),
    ("Invoice amount looks incorrect", "Billing", "Medium"),
    ("Payment failed but money was deducted", "Billing", "High"),
    ("Please explain my monthly bill", "Billing", "Low"),
    ("App crashes when I open settings", "Technical", "High"),
    ("Download button is not working", "Technical", "Medium"),
    ("Website displays a server error", "Technical", "High"),
    ("Page loads very slowly", "Technical", "Medium"),
    ("Screen turns blank after login", "Technical", "High"),
    ("I forgot my password", "Account", "Medium"),
    ("Please change my account email", "Account", "Low"),
    ("My account has been locked", "Account", "High"),
    ("I cannot sign in to my profile", "Account", "High"),
    ("How can I update my username", "Account", "Low")
]

if not tickets_path.exists():
    with open(tickets_path, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(["ticket", "category", "priority"])
        writer.writerows(tickets)

with open(tickets_path, newline="", encoding="utf-8-sig") as file:
    ticket_rows = list(csv.DictReader(file))

# Three versions of the classification prompt.
prompt_versions = {
    "v1": (
        "Classify the support ticket as Billing, Technical, or Account. "
        "Return one category only. Ticket: {ticket}"
    ),
    "v2": (
        "Role: customer support classifier. Task: choose one category. "
        "Categories: Billing, Technical, Account. "
        "Billing covers charges and refunds; Technical covers bugs; "
        "Account covers login and profile issues. "
        "Return only the category. Ticket: {ticket}"
    ),
    "v3": (
        "Classify the ticket as Billing, Technical, or Account. "
        "Examples: 'Need a refund' -> Billing; "
        "'App crashes' -> Technical; 'Forgot password' -> Account. "
        "Return only the category. Ticket: {ticket}"
    )
}

for version, prompt in prompt_versions.items():
    (PROMPTS / f"classification_{version}.txt").write_text(
        prompt, encoding="utf-8"
    )

def call_model(prompt, temperature=0, top_p=1):
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        top_p=top_p
    )
    return response.choices[0].message.content or ""

# Classification accuracy for each prompt version.
classification_results = []
accuracy_results = []

for version, template in prompt_versions.items():
    correct = 0

    for row in ticket_rows:
        try:
            answer = call_model(
                template.format(ticket=row["ticket"]),
                temperature=0
            ).strip()

            match = re.search(
                r"\b(Billing|Technical|Account)\b",
                answer,
                re.IGNORECASE
            )
            predicted = match.group(1).title() if match else "Invalid"
            is_correct = predicted == row["category"]
            correct += int(is_correct)

        except Exception as error:
            predicted = "API_ERROR"
            is_correct = False
            print("Classification error:", error)

        classification_results.append({
            "version": version,
            "ticket": row["ticket"],
            "expected": row["category"],
            "predicted": predicted,
            "correct": is_correct
        })

    accuracy = correct / len(ticket_rows) if ticket_rows else 0
    accuracy_results.append({
        "version": version,
        "correct": correct,
        "total": len(ticket_rows),
        "accuracy": round(accuracy, 3)
    })

with open(REPORTS / "task4_classification_results.csv", "w",
          newline="", encoding="utf-8-sig") as file:
    writer = csv.DictWriter(
        file,
        fieldnames=[
            "version", "ticket", "expected", "predicted", "correct"
        ]
    )
    writer.writeheader()
    writer.writerows(classification_results)

with open(REPORTS / "task4_accuracy.csv", "w",
          newline="", encoding="utf-8-sig") as file:
    writer = csv.DictWriter(file, fieldnames=accuracy_results[0].keys())
    writer.writeheader()
    writer.writerows(accuracy_results)

# Five sample resumes.
resumes_path = DATA / "resumes.json"

sample_resumes =[
    {
        "name": "resume_01",
        "format": "pdf",
        "text": """Name: Saikabilan
Skills: Python, SQL, Pandas, Machine Learning
Experience: 2 years as a Data Analyst
Education: B.Sc Computer Science
Strengths: Problem solving, teamwork"""
    },
    {
        "name": "resume_02",
        "format": "txt",
        "text": """Name: Priya 
Skills: Python, Java, SQL, FastAPI
Experience: 3 years as a Software Developer
Education: B.E Computer Science
Strengths: Communication, debugging"""
    },
    {
        "name": "resume_03",
        "format": "pdf",
        "text": """Name: Karthik 
Skills: Excel, SQL, Power BI
Experience: 1 year as a Data Analyst
Education: B.Com
Strengths: Analytical thinking"""
    },
    {
        "name": "resume_04",
        "format": "txt",
        "text": """Name: Ken S
Skills: Python, TensorFlow, Deep Learning
Experience: 4 years as an ML Engineer
Education: M.Sc Data Science
Strengths: Research, model development"""
    },
    {
        "name": "resume_05",
        "format": "pdf",
        "text": """Name: Rahul 
Skills: HTML, CSS, JavaScript, React
Experience: 2 years as a Web Developer
Education: B.Tech IT
Strengths: Creativity, teamwork"""
    },
    {
        "name": "resume_06",
        "format": "txt",
        "text": """Name: Sai R
Skills: Python, NLP, Scikit-learn
Experience: 3 years in Machine Learning
Education: M.Sc Computer Science
Strengths: Problem solving, communication"""
    },
    {
        "name": "resume_07",
        "format": "pdf",
        "text": """Name: Sanjay 
Skills: Java, Spring Boot, MySQL
Experience: 5 years as a Backend Developer
Education: B.E Information Technology
Strengths: System design, debugging"""
    },
    {
        "name": "resume_08",
        "format": "txt",
        "text": """Name: Anitha P
Skills: Python, Statistics, Tableau
Experience: 2 years as a Business Analyst
Education: MBA
Strengths: Reporting, analytical thinking"""
    }
]


if not resumes_path.exists():
    with open(resumes_path, "w", encoding="utf-8") as file:
        json.dump(sample_resumes, file, indent=4)

with open(resumes_path, encoding="utf-8") as file:
    resumes = json.load(file)

class ResumeInfo(BaseModel):
    name: str
    skills: list[str]
    years_experience: float

resume_results = []
invalid_count = 0

for resume in resumes:
    prompt = (
        "Extract resume details. Return only a JSON object with "
        "name (string), skills (array of strings), and "
        "years_experience (number). If missing, use an empty name, "
        "empty skills list, and 0 years. Resume: " + resume["text"]
    )

    try:
        answer = call_model(prompt, temperature=0)
        match = re.search(r"\{.*\}", answer, re.DOTALL)

        if not match:
            raise ValueError("No JSON object found")

        data = json.loads(match.group())
        validated = ResumeInfo.model_validate(data)

        resume_results.append({
            "resume": resume["name"],
            "valid": True,
            "extracted": validated.model_dump(),
            "error": ""
        })

    except Exception as error:
        invalid_count += 1
        resume_results.append({
            "resume": resume["name"],
            "valid": False,
            "extracted": None,
            "error": str(error)
        })

with open(REPORTS / "task4_resume_results.json", "w",
          encoding="utf-8") as file:
    json.dump(resume_results, file, indent=4)

# Temperature and top-p creative prompt experiment.
creative_results = []
creative_prompt = "Write a two-sentence story about a robot that learns to garden."

for temperature in [0, 0.7, 1.2]:
    for run in range(1, 6):
        try:
            answer = call_model(
                creative_prompt, temperature=temperature, top_p=0.95
            )
            status = "success"
        except Exception as error:
            answer = str(error)
            status = "error"

        creative_results.append({
            "experiment": "temperature",
            "setting": temperature,
            "run": run,
            "answer": answer,
            "status": status
        })

for top_p in [0.3, 0.95]:
    try:
        answer = call_model(
            creative_prompt, temperature=0.7, top_p=top_p
        )
        status = "success"
    except Exception as error:
        answer = str(error)
        status = "error"

    creative_results.append({
        "experiment": "top_p",
        "setting": top_p,
        "run": 1,
        "answer": answer,
        "status": status
    })

with open(REPORTS / "task4_creativity_results.csv", "w",
          newline="", encoding="utf-8-sig") as file:
    writer = csv.DictWriter(
        file,
        fieldnames=[
            "experiment", "setting", "run", "answer", "status"
        ]
    )
    writer.writeheader()
    writer.writerows(creative_results)

# Five word problems, with and without a step-by-step request.
problems = [
    "A shop has 12 apples and sells 5. How many remain?",
    "There are 4 boxes with 6 pens each. How many pens?",
    "A bus has 20 people, 7 leave and 3 enter. How many now?",
    "A book costs 10 dollars. What do 3 books cost?",
    "Sam has 15 coins and gives 4 to a friend. How many remain?"
]

reasoning_results = []

for problem in problems:
    for style in ["direct", "step_by_step"]:
        prompt = problem
        if style == "step_by_step":
            prompt += " Explain the calculation step by step."

        try:
            answer = call_model(prompt, temperature=0)
            status = "success"
        except Exception as error:
            answer = str(error)
            status = "error"

        reasoning_results.append({
            "problem": problem,
            "style": style,
            "answer": answer,
            "status": status
        })

with open(REPORTS / "task4_reasoning_results.csv", "w",
          newline="", encoding="utf-8-sig") as file:
    writer = csv.DictWriter(
        file, fieldnames=reasoning_results[0].keys()
    )
    writer.writeheader()
    writer.writerows(reasoning_results)

print("TASK 4 COMPLETED")
print("Classification accuracy:", accuracy_results)
print("Invalid resume outputs:", invalid_count)
print("Check reports/task4_*.csv and reports/task4_resume_results.json") 
