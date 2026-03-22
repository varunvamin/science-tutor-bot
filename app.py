import os
from flask import Flask, render_template, request, jsonify
from groq import Groq
 
app = Flask(__name__)
 
# ============================================
# Setup Groq Client
# Reads API key from environment variable
# ============================================
GROQ_API_KEY = os.environ.get('GROQ_API_KEY')  # Read from environment!
groq_client = Groq(api_key=GROQ_API_KEY)
 
# ============================================
# Science Classification Labels
# ============================================
SCIENCE_LABEL = 'a question about natural science including physics chemistry biology astronomy earth science or environmental science'
NON_SCIENCE_LABEL = 'a question about anything other than natural science such as mathematics history politics sports entertainment cooking geography literature religion finance technology computer science or general knowledge'
 
def is_science_question(question):
    try:
        check = groq_client.chat.completions.create(
            model='llama-3.3-70b-versatile',
            messages=[
                {
                    'role': 'system',
                    'content': 'Answer ONLY YES or NO. Is this a science question?'
                },
                {'role': 'user', 'content': question}
            ],
            max_tokens=3,
            temperature=0,
        )
        verdict = check.choices[0].message.content.strip().upper()
        return 'YES' in verdict
    except Exception as e:
        print(f'Groq Error: {e}')
        return False
 
    # Layer 2: Groq double check
    try:
        check = groq_client.chat.completions.create(
            model='llama-3.3-70b-versatile',
            messages=[
                {
                    'role': 'system',
                    'content': '''You are a strict science topic classifier.
Answer with ONLY one word: SCIENCE or NOT_SCIENCE.
SCIENCE topics: physics, chemistry, biology, astronomy, earth science, environmental science.
NOT_SCIENCE topics: mathematics, coding, history, politics, sports, entertainment, cooking, geography, literature, religion, finance, general knowledge.
Be strict - mathematics and coding are NOT science!'''
                },
                {'role': 'user', 'content': question}
            ],
            max_tokens=5,
            temperature=0,
        )
        verdict = check.choices[0].message.content.strip().upper()
        print(f'Groq Verdict: {verdict}')
        return 'NOT_SCIENCE' not in verdict
    except Exception as e:
        print(f'Groq Error: {e}')
        return False
 
def generate_answer(question, output_format='bullet'):
    """Generate science answer using Groq Llama 3"""
    formats = {
        'bullet':    'Answer in bullet points only. Each point starting with •',
        'paragraph': 'Answer in a clear paragraph.',
        'numbered':  'Answer as a numbered list.',
        'short':     'Give a very short one or two sentence answer.',
        'detailed':  'Give a very detailed and thorough answer.',
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
 
def detect_format(message):
    """Detect desired output format from user message"""
    m = message.lower()
    if any(w in m for w in ['bullet', 'points', 'list']): return 'bullet'
    elif any(w in m for w in ['short', 'brief', 'quick']): return 'short'
    elif any(w in m for w in ['detail', 'detailed', 'thorough']): return 'detailed'
    elif any(w in m for w in ['numbered', 'steps', 'number']): return 'numbered'
    elif any(w in m for w in ['paragraph', 'para']): return 'paragraph'
    return 'bullet'
 
# ============================================
# Flask Routes
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
 
    # Clean format words
    clean = user_message
    for p in ['in bullet points', 'in paragraph', 'in numbered list',
              'in short', 'in detail', 'as bullet points']:
        clean = clean.replace(p, '').strip()
 
    # Check if science
    if not is_science_question(clean):
        return jsonify({'response': 'This chatbot only answers science-related questions.'})
 
    # Generate answer
    answer = generate_answer(clean, fmt)
    return jsonify({'response': answer})
 
if __name__ == '__main__':
    # Read port from environment — required for Railway!
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
