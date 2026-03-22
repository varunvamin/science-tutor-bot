# 🔬 Science Tutor Bot - Web Application

## Setup Instructions

### Step 1: Install dependencies
```
pip install -r requirements.txt
```

### Step 2: Add your Groq API key
Open `app.py` and replace:
```python
GROQ_API_KEY = 'gsk_YOUR_KEY_HERE'
```
With your actual Groq API key from console.groq.com

### Step 3: Run locally
```
python app.py
```
Open browser and go to: http://localhost:5000

### Step 4: Deploy online (Render.com - Free)
1. Push code to GitHub
2. Go to render.com
3. Click New → Web Service
4. Connect your GitHub repo
5. Set Start Command: `python app.py`
6. Click Deploy!

## How it works
- BART zero-shot classification → detects if question is science
- Groq double check → second verification
- Llama 3.3 70B → generates the answer
