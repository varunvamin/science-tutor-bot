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
    formats = {
    'bullet': 'Answer in simple bullet points using "-" only. No bold, no markdown.',
    'paragraph': 'Answer in a clean paragraph without any markdown.',
    'numbered': 'Answer as numbered list using 1. 2. 3. format. No markdown.',
    'short': 'Give a very short 1-2 sentence answer. No formatting.',
    'detailed': 'Give a detailed explanation in plain text. No markdown or symbols.'
}
    instruction = formats.get(output_format, formats['bullet'])

    response = groq_client.chat.completions.create(
        model='llama-3.3-70b-versatile',
        messages=[
            {
                'role': 'system',
                'content': f'You are a Science Tutor Bot. {instruction}'
            },
            {'role': 'user', 'content': question}
        ],
        max_tokens=500,
        temperature=0.3,
    )

    return response.choices[0].message.content

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

    return jsonify({'response': answer})

# ============================================
# Run App
# ============================================
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
