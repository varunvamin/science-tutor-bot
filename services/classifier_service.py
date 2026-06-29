import logging
from config import Config

logger = logging.getLogger(__name__)

class ClassifierService:
    def __init__(self):
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
        """Use BART zero-shot classification as a first layer check"""
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
