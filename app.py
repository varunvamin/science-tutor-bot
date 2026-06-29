from flask import Flask, render_template, request, jsonify
from services.llm_service import LLMService
from services.classifier_service import ClassifierService

app = Flask(__name__)

# Initialize services
llm_service = LLMService()
classifier_service = ClassifierService()

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

    # Clean format words for cleaner classification
    clean = user_message
    for p in ['in bullet points', 'in paragraph', 'in numbered list',
              'in short', 'in detail', 'as bullet points']:
        clean = clean.replace(p, '').strip()

    # Two-layer science classification check
    # Layer 1: BART (if available)
    if not classifier_service.is_science_question(clean):
        return jsonify({'response': 'This chatbot only answers science-related questions.'})

    # Layer 2: Groq double check
    if not llm_service.verify_science_topic(clean):
        return jsonify({'response': 'This chatbot only answers science-related questions.'})

    # Generate answer
    answer = llm_service.generate_answer(clean, fmt)
    return jsonify({'response': answer})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
