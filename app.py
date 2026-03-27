import os
from flask import Flask, render_template, request, jsonify
from groq import Groq

app = Flask(__name__)

# ============================================
# Groq Setup
# ============================================
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
groq_client = Groq(api_key=GROQ_API_KEY)

# ============================================
# Detect Format
# ============================================
def detect_format(message):
    m = message.lower()
    if any(w in m for w in ['numbered', 'steps']):
        return 'numbered'
    elif any(w in m for w in ['paragraph', 'para']):
        return 'paragraph'
    elif any(w in m for w in ['bullet', 'points', 'list']):
        return 'bullet'
    return 'bullet'


# ============================================
# Detect Style
# ============================================
def detect_style(message):
    m = message.lower()
    if any(w in m for w in ['detail', 'detailed']):
        return 'detailed'
    elif any(w in m for w in ['short', 'brief']):
        return 'short'
    return 'normal'


# ============================================
# Convert Bullets to Numbers (Backup Fix)
# ============================================
def convert_bullets_to_numbers(text):
    lines = text.split("\n")
    new_lines = []
    count = 1

    for line in lines:
        stripped = line.strip()

        if stripped.startswith("•") or stripped.startswith("-") or stripped.startswith("*"):
            content = stripped[1:].strip()
            new_lines.append(f"{count}. {content}")
            count += 1
        else:
            new_lines.append(line)

    return "\n".join(new_lines)


# ============================================
# Check Science (Groq only)
# ============================================
def is_science_question(question):
    try:
        check = groq_client.chat.completions.create(
            model='llama-3.3-70b-versatile',
            messages=[
                {
                    'role': 'system',
                    'content': 'Reply ONLY with YES or NO. Is this a science question? Science includes physics, chemistry, biology, astronomy, earth science.'
                },
                {
                    'role': 'user',
                    'content': question
                }
            ],
            max_tokens=5,
            temperature=0
        )

        verdict = check.choices[0].message.content.strip().upper()
        return 'YES' in verdict

    except Exception as e:
        print("Classification Error:", e)
        return False


# ============================================
# Generate Answer
# ============================================
def generate_answer(question, fmt, style):

    # Format instruction
    if fmt == 'bullet':
        format_instruction = "Answer ONLY in bullet points. Each line must start with '•'."
    elif fmt == 'paragraph':
        format_instruction = "Answer ONLY in a single clean paragraph. Do NOT use bullet points or numbering."
    elif fmt == 'numbered':
        format_instruction = "Answer ONLY in numbered points like 1. 2. 3. Do NOT use bullet points."
    else:
        format_instruction = "Answer clearly."

    # Style instruction
    if style == 'short':
        style_instruction = "Keep the answer short and concise."
    elif style == 'detailed':
        style_instruction = "Give a detailed explanation."
    else:
        style_instruction = "Answer normally."

    response = groq_client.chat.completions.create(
        model='llama-3.3-70b-versatile',
        messages=[
            {
                'role': 'system',
                'content': f'You are a Science Tutor Bot. {format_instruction} {style_instruction}'
            },
            {
                'role': 'user',
                'content': question
            }
        ],
        max_tokens=700,
        temperature=0.1
    )

    answer = response.choices[0].message.content.strip()

    # Backup fix: convert bullets to numbers if needed
    if fmt == 'numbered':
        answer = convert_bullets_to_numbers(answer)

    return answer


# ============================================
# Routes
# ============================================
@app.route('/')
def home():
    return render_template('index.html')


@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_message = data.get('message', '').strip()

    if not user_message:
        return jsonify({'response': 'Please ask a science question!'})

    # Detect format and style
    fmt = detect_format(user_message)
    style = detect_style(user_message)

    # Clean message
    clean = user_message.lower()

    for p in ['bullet', 'points', 'list', 'short', 'brief',
              'detailed', 'detail', 'paragraph', 'numbered', 'steps',
              'in bullet points', 'in paragraph', 'in short', 'in detail', 'in numbered form']:
        clean = clean.replace(p, '').strip()

    # Check if science
    if not is_science_question(clean):
        return jsonify({'response': 'This chatbot only answers science-related questions.'})

    # Generate answer
    answer = generate_answer(clean, fmt, style)

    return jsonify({'response': answer})


# ============================================
# Run
# ============================================
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
