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
def generate_answer(question, fmt):

    if fmt == 'bullet':
        instruction = "Answer ONLY in bullet points. Each line must start with '•'."
    elif fmt == 'paragraph':
        instruction = "Answer in a single clean paragraph."
    elif fmt == 'numbered':
        instruction = "Answer ONLY as numbered steps (1. 2. 3.)."
    elif fmt == 'short':
        instruction = "Answer in 1-2 short sentences."
    elif fmt == 'detailed':
        instruction = "Give a detailed explanation."
    else:
        instruction = "Answer normally."

    response = groq_client.chat.completions.create(
        model='llama-3.3-70b-versatile',
        messages=[
            {'role': 'system', 'content': f'You are a Science Tutor Bot. {instruction}'},
            {'role': 'user', 'content': question}
        ],
        max_tokens=500,
        temperature=0.1
    )

    return response.choices[0].message.content.strip()

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
              'detailed', 'paragraph', 'numbered',
              'in bullet points', 'in paragraph', 'in short', 'in detail']:
        clean = clean.replace(p, '').strip()

    # Check if science
    if not is_science_question(clean):
        return jsonify({'response': 'This chatbot only answers science-related questions.'})

    # Generate answer
    answer = generate_answer(clean, fmt)

    return jsonify({'response': answer})


# ============================================
# Run
# ============================================
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)






