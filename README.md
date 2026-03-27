# 🔬 Science Tutor Bot

An AI-powered web application that answers **only science-related questions** in a clean, user-friendly interface.  
It uses the **Groq API** with **Llama 3.3 70B Versatile** for fast and intelligent responses.

---

## 📌 Overview

Science Tutor Bot is designed as a **domain-specific educational assistant**.  
Unlike a general chatbot, it is restricted to **science topics only**, such as:

- Physics
- Chemistry
- Biology
- Astronomy
- Earth Science

The application also supports multiple response formats and styles, making it useful for quick learning as well as detailed explanations.

---

## ✨ Features

- ✅ Answers only **science-related questions**
- ✅ Fast AI-generated responses using **Groq API**
- ✅ Supports multiple output formats:
  - Bullet Points
  - Numbered Points
  - Paragraph
- ✅ Supports multiple response styles:
  - Short Answer
  - Detailed Explanation
- ✅ Rejects non-science questions
- ✅ Runs locally and can also be deployed online

---

## 🛠️ Technologies Used

### Frontend
- HTML
- CSS
- JavaScript

### Backend
- Python
- Flask

### AI / API
- Groq API
- Llama 3.3 70B Versatile

### Deployment
- Render (optional)

---

## ⚙️ How It Works

### 1. User Input
The user enters a question in the chatbot interface.

### 2. Science Validation
The bot first checks whether the question is related to science using the **Groq API**.

### 3. Format & Style Detection
The system detects the required:
- **Format** → bullet / numbered / paragraph
- **Style** → short / detailed

### 4. AI Response Generation
If the question is valid, the bot sends it to **Llama 3.3 70B Versatile** through Groq to generate the answer.

### 5. Final Output
The response is displayed to the user in the selected format and style.

---

## 🚀 Setup Instructions

### Step 1: Install Dependencies
Open the project folder in terminal / command prompt and run:

```bash
pip install -r requirements.txt
