"""
Text Feature Extraction

Extracts emotion-relevant features from text:
- Sentiment analysis
- Sentence embeddings (SentenceTransformer, etc.)
- Basic linguistic features

Optional dependency: transformers, sentence-transformers
"""

import numpy as np
from typing import Optional, List
import logging


class TextFeatureExtractor:
    """
    Extract emotion-relevant features from text
    
    Requires: transformers and/or sentence-transformers (optional)
    Falls back to simple bag-of-words features
    """
    
    def __init__(self,
                 model_name: str = "all-MiniLM-L6-v2",
                 use_sentiment: bool = True,
                 embedding_dim: int = 768):
        """
        Args:
            model_name: SentenceTransformer model name
            use_sentiment: Extract sentiment features
            embedding_dim: Expected embedding dimension
        """
        self.embedding_dim = embedding_dim
        self.use_sentiment = use_sentiment
        
        # Try to load sentence transformer
        self.sentence_encoder = None
        try:
            from sentence_transformers import SentenceTransformer
            self.sentence_encoder = SentenceTransformer(model_name)
            logging.info(f"Loaded SentenceTransformer: {model_name}")
        except ImportError:
            logging.warning("sentence-transformers not available")
        except Exception as e:
            logging.warning(f"Could not load SentenceTransformer: {e}")
        
        # Try to load sentiment analyzer
        self.sentiment_analyzer = None
        if use_sentiment:
            try:
                from transformers import pipeline
                self.sentiment_analyzer = pipeline(
                    "sentiment-analysis",
                    model="distilbert-base-uncased-finetuned-sst-2-english"
                )
                logging.info("Loaded sentiment analyzer")
            except ImportError:
                logging.warning("transformers not available for sentiment")
            except Exception as e:
                logging.warning(f"Could not load sentiment analyzer: {e}")
    
    def extract_features(self, text: str) -> np.ndarray:
        """
        Extract comprehensive text features
        
        Args:
            text: Input text string
        
        Returns:
            Feature vector (768 dimensions default)
        """
        features = []
        
        # 1. Sentence embedding
        embedding = self.extract_embedding(text)
        features.append(embedding)
        
        # 2. Sentiment features
        if self.use_sentiment:
            sentiment_features = self.extract_sentiment(text)
            features.append(sentiment_features)
        
        # Concatenate and pad/trim to expected dimension
        all_features = np.concatenate(features)
        
        if len(all_features) < self.embedding_dim:
            all_features = np.pad(all_features, (0, self.embedding_dim - len(all_features)))
        elif len(all_features) > self.embedding_dim:
            all_features = all_features[:self.embedding_dim]
        
        return all_features.astype(np.float32)
    
    def extract_embedding(self, text: str) -> np.ndarray:
        """
        Extract sentence embedding
        
        Returns:
            Embedding vector
        """
        if self.sentence_encoder is not None:
            try:
                embedding = self.sentence_encoder.encode(text, convert_to_numpy=True)
                return embedding
            except Exception as e:
                logging.warning(f"Error extracting embedding: {e}")
        
        # Fallback: simple bag-of-words features
        return self._extract_bow_features(text)
    
    def _extract_bow_features(self, text: str) -> np.ndarray:
        """Simple bag-of-words features as fallback"""
        # Very simple features: word count, character count, etc.
        words = text.lower().split()
        
        features = [
            len(words),  # Word count
            len(text),   # Character count
            len(set(words)) / (len(words) + 1),  # Unique word ratio
            np.mean([len(w) for w in words]) if words else 0,  # Avg word length
        ]
        
        # Pad to reasonable size
        features = np.array(features)
        features = np.pad(features, (0, max(0, 384 - len(features))))
        
        return features[:384]  # Return first 384 dimensions
    
    def extract_sentiment(self, text: str) -> np.ndarray:
        """
        Extract sentiment features
        
        Returns:
            Sentiment feature vector (3 dimensions: positive, negative, neutral scores)
        """
        if self.sentiment_analyzer is not None:
            try:
                result = self.sentiment_analyzer(text[:512])[0]  # Limit text length
                
                label = result['label'].lower()
                score = result['score']
                
                if 'positive' in label:
                    return np.array([score, 1 - score, 0.0])
                elif 'negative' in label:
                    return np.array([1 - score, score, 0.0])
                else:
                    return np.array([0.0, 0.0, 1.0])
                    
            except Exception as e:
                logging.warning(f"Error extracting sentiment: {e}")
        
        # Fallback: neutral sentiment
        return np.array([0.5, 0.5, 0.0])


def extract_text_features_batch(text_batch: List[str],
                                **kwargs) -> np.ndarray:
    """
    Convenience function to extract features from a batch of texts
    
    Args:
        text_batch: List of text strings
        **kwargs: Additional arguments for TextFeatureExtractor
    
    Returns:
        Feature matrix (batch, 768)
    """
    extractor = TextFeatureExtractor(**kwargs)
    
    feature_list = []
    
    for text in text_batch:
        features = extractor.extract_features(text)
        feature_list.append(features)
    
    return np.stack(feature_list)
