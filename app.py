import os
from flask import Flask, render_template, request, jsonify
from groq import Groq
import torch
from transformers import pipeline

app = Flask(__name__)

# ============================================
# Load BART Classifier
# ============================================
print("Loading classifier...")
classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli",
    device=0 if torch.cuda.is_available() else -1
)
print("✅ Classifier loaded")

# ============================================
# Groq Setup
# ============================================
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
groq_client = Groq(api_key=GROQ_API_KEY)

# ============================================
# Labels
# ============================================
SCIENCE_LABEL = "science question about physics chemistry biology astronomy earth science"
NON_SCIENCE_LABEL = "non science question like math coding history sports general knowledge"

# ============================================
# Check Science
# ============================================
def is_science_question(question):
    try:
        result = classifier(
            question,
            candidate_labels=[SCIENCE_LABEL, NON_SCIENCE_LABEL]
        )

        label = result["labels"][0]
        score = result["scores"][0]

        if label != SCIENCE_LABEL or score < 0.6:
            return False

    except:
        return False

    return True

# ============================================
# Detect Format
# ============================================
def detect_format(message):
    m = message.lower()
    if any(w in m for w in ["bullet", "points", "list"]):
        return "bullet"
    elif any(w in m for w in ["short", "brief"]):
        return "short"
    elif any(w in m for w in ["detail", "detailed"]):
        return "detailed"
    elif any(w in m for w in ["numbered", "steps"]):
        return "numbered"
    elif any(w in m for w in ["paragraph", "para"]):
        return "paragraph"
    return "bullet"

# ============================================
# Generate Answer
# ============================================
def generate_answer(question, fmt):

    if fmt == "bullet":
        instruction = "Answer ONLY in bullet points. Each line must start with '•'. No paragraphs."
    elif fmt == "paragraph":
        instruction = "Answer in a single clean paragraph only. No bullet points."
    elif fmt == "numbered":
        instruction = "Answer ONLY as numbered steps (1. 2. 3.)."
    elif fmt == "short":
        instruction = "Answer in 1-2 short sentences only."
    elif fmt == "detailed":
        instruction = "Give a detailed explanation."
    else:
        instruction = "Answer normally."

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": f"You are a Science Tutor Bot. {instruction}"},
            {"role": "user", "content": question}
        ],
        max_tokens=500,
        temperature=0.1
    )

    return response.choices[0].message.content.strip()

# ============================================
# Routes
# ============================================
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_message = data.get("message", "").strip()

    if not user_message:
        return jsonify({"response": "Please ask a science question!"})

    # Detect format
    fmt = detect_format(user_message)

    # Clean text
    clean = user_message.lower()

    for p in ["bullet", "points", "list", "short", "brief",
              "detailed", "paragraph", "numbered",
              "in bullet points", "in paragraph", "in short", "in detail"]:
        clean = clean.replace(p, "").strip()

    # Check science
    if not is_science_question(clean):
        return jsonify({"response": "This chatbot only answers science-related questions."})

    # Generate answer
    answer = generate_answer(clean, fmt)

    # ===== FORMAT FIX =====
    if fmt == "bullet":
        lines = answer.split("\n")
        answer = "\n".join([f"• {line.lstrip('-•0123456789. ').strip()}" for line in lines if line.strip()])

    elif fmt == "numbered":
        lines = answer.split("\n")
        answer = "\n".join([f"{i+1}. {line.lstrip('-•0123456789. ').strip()}" for i, line in enumerate(lines) if line.strip()])

    elif fmt == "paragraph":
        answer = " ".join(answer.split())

    return jsonify({"response": answer})


# ============================================
# Run
# ============================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
