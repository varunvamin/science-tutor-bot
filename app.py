import os
from flask import Flask, render_template, request, jsonify
from groq import Groq

app = Flask(__name__)

# ============================================
# Setup Groq Client
# ============================================
GROQ_API_KEY = os.environ.get('GROQ_API_KEY')
groq_client = Groq(api_key=GROQ_API_KEY)

# ============================================
# Science Classification using Groq
# ============================================
def is_science_question(question):
    try:
        check = groq_client.chat.completions.create(
            model='llama-3.3-70b-versatile',
            messages=[
                {
                    'role': 'system',
                    'content': '''You are a strict classifier.
Answer ONLY: SCIENCE or NOT_SCIENCE.

SCIENCE: physics, chemistry, biology, astronomy, earth science
NOT_SCIENCE: math, coding, history, politics, sports, etc'''
                },
                {'role': 'user', 'content': question}
            ],
            max_tokens=5,
            temperature=0,
        )

        verdict = check.choices[0].message.content.strip().upper()
        return verdict == 'SCIENCE'

    except Exception as e:
        print("Classification Error:", e)
        return False

# ============================================
# Generate Answer
# ============================================
def generate_answer(question, output_format='bullet'):
    if output_format == 'bullet':
        instruction = "Answer ONLY in bullet points. Each line must start with '•'. No paragraphs. No markdown."
    elif output_format == 'paragraph':
        instruction = "Answer in a single clean paragraph. Do NOT use bullet points or lists."
    elif output_format == 'numbered':
        instruction = "Answer ONLY as a numbered list (1. 2. 3.). No paragraphs."
    elif output_format == 'short':
        instruction = "Answer in 1-2 short sentences only."
    elif output_format == 'detailed':
        instruction = "Give a detailed explanation in multiple paragraphs."
    else:
        instruction = "Answer in a clean paragraph."

    response = groq_client.chat.completions.create(
        model='llama-3.3-70b-versatile',
        messages=[
            {
                'role': 'system',
                'content': f'You are a strict Science Tutor Bot. {instruction}'
            },
            {
                'role': 'user',
                'content': question
            }
        ],
        max_tokens=500,
        temperature=0.1,
    )

    return response.choices[0].message.content.strip()

# ============================================
# Detect Format
# ============================================
def detect_format(message):
    m = message.lower()
    if any(w in m for w in ['bullet', 'points', 'list']):
        return 'bullet'
    elif any(w in m for w in ['short', 'brief']):
        return 'short'
    elif any(w in m for w in ['detail', 'detailed']):
        return 'detailed'
    elif any(w in m for w in ['numbered', 'steps']):
        return 'numbered'
    elif any(w in m for w in ['paragraph', 'para']):
        return 'paragraph'
    return 'bullet'

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

    # Detect format
    fmt = detect_format(user_message)

    # Clean message
    clean = user_message.lower()

    for p in ['bullet', 'points', 'list', 'short', 'brief',
              'detailed', 'paragraph', 'numbered']:
        clean = clean.replace(p, '')

    for p in ['in bullet points', 'in paragraph', 'in numbered list',
              'in short', 'in detail', 'as bullet points']:
        clean = clean.replace(p, '').strip()

    # Check science
    if not is_science_question(clean):
        return jsonify({'response': 'This chatbot only answers science-related questions.'})

    # Generate answer
    answer = generate_answer(clean, fmt)

# ===== FORMAT ENFORCEMENT =====
if fmt == 'bullet':
    lines = answer.split('\n')
    answer = '\n'.join([f"• {line.lstrip('-•0123456789. ').strip()}" for line in lines if line.strip()])

elif fmt == 'numbered':
    lines = answer.split('\n')
    answer = '\n'.join([f"{i+1}. {line.lstrip('-•0123456789. ').strip()}" for i, line in enumerate(lines) if line.strip()])

elif fmt == 'paragraph':
    answer = ' '.join(answer.split())  # remove line breaks

    return jsonify({'response': answer})

# ============================================
# Run App
# ============================================
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
