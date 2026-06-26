# 🔬 Science Tutor Bot - Web Application

## Setup Instructions

### Step 1: Install dependencies
```
pip install -r requirements-local.txt
```

### Step 2: Set your Groq API key
Set your Groq API key as an environment variable before running the app. 
On Windows (PowerShell):
```powershell
$env:GROQ_API_KEY="your_api_key_here"
```
On Mac/Linux:
```bash
export GROQ_API_KEY="your_api_key_here"
```

### Step 3: Run locally
```
python app.py
```
Open browser and go to: http://localhost:5000

### Step 4: Deploy online (Vercel - Free)
1. Push your code to GitHub.
2. Go to [Vercel.com](https://vercel.com/) and click **Add New → Project**.
3. Import your `science-tutor-bot` GitHub repository.
4. In the configuration, open **Environment Variables** and add `GROQ_API_KEY` with your API key.
5. Click **Deploy!**

*Note: The app is configured to use the heavy PyTorch/BART model when run locally, but automatically switches to a lightweight, Groq-only serverless mode when deployed to Vercel to bypass the 250MB size limits.*

## How it works
- BART zero-shot classification → detects if question is science
- Groq double check → second verification
- Llama 3.3 70B → generates the answer
