# 🔬 Science Tutor Bot - Web Application

<p align="left">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white" />
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" />
  <img src="https://img.shields.io/badge/pytest-0A9EDC?style=for-the-badge&logo=pytest&logoColor=white" />
</p>

An intelligent web application that uses Llama 3.3 and BART zero-shot classification to accurately identify and answer science-related questions, while politely rejecting off-topic queries. 

Recently updated with a **modular architecture**, comprehensive **error logging**, and a highly polished **glassmorphic/airy UI**.

## ✨ Features
- **Two-Layer Topic Classification:** Uses HuggingFace BART locally and Groq LLM as a fallback to strictly ensure only science topics are discussed.
- **Rich formatting:** Request answers in bullets, paragraphs, numbered lists, or short/detailed formats.
- **Beautiful UI:** A clean, airy design with a soft gradient background, subtle shadows, and micro-animations.
- **Copy to Clipboard & Clear Chat:** Built-in UI tools to easily copy bot answers or reset the conversation.
- **Modular Codebase:** Cleanly separated concerns (`app.py`, `config.py`, `services/`, `static/`).

## ⚙️ Setup Instructions

### Step 1: Install dependencies
```bash
pip install -r requirements-local.txt
# (For running tests, pip install -r requirements.txt is sufficient)
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
```bash
python app.py
```
Open browser and go to: http://localhost:5000

### Step 4: Run Tests
```bash
pytest tests/
```

### Step 5: Deploy online (Vercel - Free)
1. Push your code to GitHub.
2. Go to [Vercel.com](https://vercel.com/) and click **Add New → Project**.
3. Import your `science-tutor-bot` GitHub repository.
4. In the configuration, open **Environment Variables** and add `GROQ_API_KEY` with your API key.
5. Click **Deploy!**

*Note: The app is configured to use the heavy PyTorch/BART model when run locally, but automatically switches to a lightweight, Groq-only serverless mode when deployed to Vercel to bypass the 250MB size limits.*

## 🧠 Architecture
- **`app.py`**: Flask router and entry point.
- **`config.py`**: Centralized configuration and prompts.
- **`services/llm_service.py`**: Groq API integration.
- **`services/classifier_service.py`**: BART classification logic.
- **`static/`**: Separated CSS styling and frontend JavaScript logic.

## 📝 License
This project is open-source and available under the [MIT License](LICENSE).