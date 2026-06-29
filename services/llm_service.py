from groq import Groq
import logging
from config import Config

logger = logging.getLogger(__name__)

class LLMService:
    """
    Service class for interacting with the Groq Large Language Model API.
    Handles the generation of tutor responses and secondary classification checks.
    """
    def __init__(self):
        """Initializes the LLM service with the Groq API key from configuration."""
        self.client = Groq(api_key=Config.GROQ_API_KEY) if Config.GROQ_API_KEY != 'your_api_key_here' else None

    def generate_answer(self, chat_history, output_format='bullet'):
        """
        Generates a science-focused answer using the Groq API.
        
        Args:
            chat_history (str or list): The user's question or full conversation history.
            output_format (str): The requested formatting style (e.g., 'bullet', 'paragraph').
            
        Returns:
            str: The generated response or an error message.
        """
        if not self.client:
            return "Error: Groq API key is missing or invalid."
            
        instruction = Config.FORMATS.get(output_format, Config.FORMATS['bullet'])
        
        system_msg = {
            'role': 'system',
            'content': f'You are a Science Tutor Bot. {instruction}'
        }
        
        if isinstance(chat_history, str):
            messages = [system_msg, {'role': 'user', 'content': chat_history}]
        else:
            # We only want to send the last 5 messages to avoid blowing up the context window
            messages = [system_msg] + chat_history[-5:]

        try:
            response = self.client.chat.completions.create(
                model=Config.GROQ_MODEL_NAME,
                messages=messages,
                max_tokens=500,
                temperature=0.3,
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f'Groq Error generating answer: {e}')
            return "Sorry, I encountered an error generating an answer."

    def verify_science_topic(self, question):
        """
        Acts as a secondary check to verify if a topic is science-related.
        Used as a fallback when the local BART classifier is unsure or unavailable.
        
        Args:
            question (str): The user's question to verify.
            
        Returns:
            bool: True if it's a science topic, False otherwise.
        """
        if not self.client:
            return False
            
        try:
            check = self.client.chat.completions.create(
                model=Config.GROQ_MODEL_NAME,
                messages=[
                    {
                        'role': 'system',
                        'content': Config.GROQ_CLASSIFIER_SYSTEM_PROMPT
                    },
                    {'role': 'user', 'content': question}
                ],
                max_tokens=5,
                temperature=0,
            )
            verdict = check.choices[0].message.content.strip().upper()
            logger.info(f'Groq Verdict: {verdict}')
            return 'NOT_SCIENCE' not in verdict
        except Exception as e:
            logger.error(f'Groq Error verifying topic: {e}')
            return False
