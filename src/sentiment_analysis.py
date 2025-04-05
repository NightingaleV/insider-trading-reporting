#%%

import logging
from config.base import HUGGING_FACE_API_KEY

#%%

class SentimentAnalysis:
    """Sentiment Analysis using FinBERT model from Hugging Face
    
    !!! note
        This class uses the FinBERT model from Hugging Face for sentiment analysis.
        The model is a transformer-based model fine-tuned for financial sentiment analysis.
        
    !!! warning
        Local model requires additional dependencies and may not be as performant as the remote model.
    
    Methods:
        - analyze(text: str): Analyzes the sentiment of the given text.
    """
    MODEL_NAME = 'yiyanghkust/finbert-tone'

    def __init__(self, local_model=False):
        self.local_model= local_model
        self.model = self._get_local_model() if local_model else self._get_remote_model()
        # self.model = self._get_model()
        
    def _get_remote_model(self):
        from huggingface_hub import InferenceClient
        logging.info("Setting up Inference Client for Sentiment Analysis")
        client = InferenceClient(
            model=self.MODEL_NAME,
            api_key=HUGGING_FACE_API_KEY,
        )
        logging.info("Inference Client Ready for Sentiment Analysis")
        return client
    
    def _get_local_model(self):
        from transformers import Tokenizer, Transformer, pipeline
        logging.info("Get Model for Sentiment Analysis")
        from transformers import BertTokenizer, BertForSequenceClassification
        finbert = BertForSequenceClassification.from_pretrained(self.MODEL_NAME,num_labels=2)
        tokenizer = BertTokenizer.from_pretrained(self.MODEL_NAME)
        logging.info("Model Loaded")
        model = pipeline('sentiment-analysis', model=finbert, tokenizer=tokenizer)
        return model
    
    def _get_inference(self, text: list[str]):
        if self.local_model:
            results = self.model(text)
            label_map = {
                'Positive': 1.0,
                'Neutral': 0.5,
                'Negative': 0.0
            }
            formatted_results = []
            for result in results:
                # Convert score from [-1, 1] to [0, 1] range
                normalized_score = (result['score'] + 1) / 2
                formatted_results.append({
                    'label': label_map.get(result['label'], 0.5),  # default to neutral if unknown
                    'score': normalized_score
                })
        else:
            results = self.model.text_classification(text=text)
            formatted_results = [{'label': result['label'], 'score': result['score']} for result in results]
        return formatted_results
    
    def analyze(self, text: list[str]) -> list:
        if isinstance(text, str):
            text = [text]
        try:
            return self._get_inference(text)
        except Exception as e:
            logging.error(f"Error in sentiment analysis: {e}")
            return []
        
#%% 
if __name__ == "__main__":
    MOCK_NEWS = [
        {
            "title": "Company X reports record profits",
            "content": "Company X has reported record profits for the last quarter, exceeding analysts' expectations."
        },
        {
            "title": "Company Y faces lawsuit over product safety",
            "content": "Company Y is facing a lawsuit over allegations of product safety violations."
        },
        {
            "title": "Company Z announces merger with Company A",
            "content": "Company Z has announced a merger with Company A, creating a new industry leader."
        }
    ]

    MOCK_NEWS_LIST = [news['title'] + ": " + news['content'] for news in MOCK_NEWS]
    print(MOCK_NEWS_LIST)
    # Example usage
    sentiment_analyzer = SentimentAnalysis()
    results = sentiment_analyzer.analyze(MOCK_NEWS_LIST)
    print(results)
    scores = [result['score'] for result in results]
    print(scores)