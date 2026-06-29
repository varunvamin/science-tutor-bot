import logging
from config import Config

logger = logging.getLogger(__name__)

class ClassifierService:
    """
    Service class for zero-shot text classification using a local HuggingFace pipeline.
    Used to quickly filter out non-science questions before calling the external LLM API.
    """
    def __init__(self):
        """Initializes the BART zero-shot classifier if the transformers library and model are available."""
        self.has_local_model = False
        self.classifier = None
        
        try:
            from transformers import pipeline
            import torch
            self.has_local_model = True
            
            logger.info('Loading BART classifier...')
            self.classifier = pipeline(
                'zero-shot-classification',
                model=Config.BART_MODEL_NAME,
                device=0 if torch.cuda.is_available() else -1
            )
            logger.info('✅ Classifier loaded!')
        except ImportError:
            logger.warning('Running in lightweight mode. BART classifier disabled.')
        except Exception as e:
            logger.error(f'Failed to load BART classifier: {e}')
            self.has_local_model = False

    def is_science_question(self, question):
        """
        Uses the local BART zero-shot classifier to determine if a question is science-related.
        
        Args:
            question (str): The user's question.
            
        Returns:
            bool: True if the question is likely science-related or if the local model is unavailable.
                  False if the model is highly confident it is not science.
        """
        if self.has_local_model and self.classifier is not None:
            try:
                result = self.classifier(
                    question,
                    candidate_labels=[Config.SCIENCE_LABEL, Config.NON_SCIENCE_LABEL]
                )
                
                # Extract score for SCIENCE_LABEL
                science_score = result['scores'][0] if result['labels'][0] == Config.SCIENCE_LABEL else result['scores'][1]
                logger.info(f'BART Science Score: {science_score:.2f}')
                
                # If non-science is top label or confidence is low, reject
                if result['labels'][0] != Config.SCIENCE_LABEL or result['scores'][0] <= 0.6:
                    return False
                return True
            except Exception as e:
                logger.error(f'BART Error: {e}')
                return False
        else:
            logger.info('Skipping BART classification (lightweight mode)')
            # Return True so the second layer (Groq) can handle it
            return True
