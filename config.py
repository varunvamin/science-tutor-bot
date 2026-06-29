import os

class Config:
    # API Keys
    GROQ_API_KEY = os.environ.get('GROQ_API_KEY', 'your_api_key_here')
    
    # Model Names
    BART_MODEL_NAME = 'facebook/bart-large-mnli'
    GROQ_MODEL_NAME = 'llama-3.3-70b-versatile'
    
    # Classification Labels
    SCIENCE_LABEL = 'a question about natural science including physics chemistry biology astronomy earth science or environmental science'
    NON_SCIENCE_LABEL = 'a question about anything other than natural science such as mathematics history politics sports entertainment cooking geography literature religion finance technology computer science or general knowledge'
    
    # Prompts
    GROQ_CLASSIFIER_SYSTEM_PROMPT = '''You are a strict science topic classifier.
Answer with ONLY one word: SCIENCE or NOT_SCIENCE.
SCIENCE topics: physics, chemistry, biology, astronomy, earth science, environmental science.
NOT_SCIENCE topics: mathematics, coding, history, politics, sports, entertainment, cooking, geography, literature, religion, finance, general knowledge.
Be strict - mathematics and coding are NOT science!'''
    
    # Formats
    FORMATS = {
        'bullet':    'Answer in bullet points only. Each point starting with •',
        'paragraph': 'Answer in a clear paragraph.',
        'numbered':  'Answer as a numbered list.',
        'short':     'Give a very short one or two sentence answer.',
        'detailed':  'Give a very detailed and thorough answer.',
    }
