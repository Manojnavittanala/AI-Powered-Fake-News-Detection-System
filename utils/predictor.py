import os
import pickle
import logging
import numpy as np
from utils.preprocessing import clean_text

LABEL_MAP = {0: 'Fake', 1: 'Real'}
# The checked-in sample has only five articles per class, so calibrated scores
# are compressed near 50%. Abstain only on a near coin-flip by default; larger
# deployments can set UNCERTAINTY_THRESHOLD=70 in the environment.
UNCERTAINTY_THRESHOLD = float(os.environ.get('UNCERTAINTY_THRESHOLD', '65'))
logger = logging.getLogger(__name__)


def load_model(model_path='model.pkl', vectorizer_path='vectorizer.pkl'):
    """Load the trained model and vectorizer."""
    if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
        raise FileNotFoundError('Model or vectorizer file not found. Please run train_model.py first.')

    with open(model_path, 'rb') as model_file:
        saved = pickle.load(model_file)

    with open(vectorizer_path, 'rb') as vec_file:
        vectorizer = pickle.load(vec_file)

    if isinstance(saved, dict):
        model = saved.get('model') or saved.get('pac_model')
        if model is not None:
            model._fraud_model_name = saved.get('model_name', model.__class__.__name__)
    else:
        model = saved
    metrics = saved.get('metrics', {}) if isinstance(saved, dict) else {}
    if isinstance(saved, dict):
        metrics['model_name'] = saved.get('model_name', 'unknown')
    return model, vectorizer, metrics


def _sigmoid(score):
    return 1 / (1 + np.exp(-score))


def get_top_keywords(text, model, vectorizer, top_n=10):
    cleaned = clean_text(text)
    vector = vectorizer.transform([cleaned])
    feature_names = vectorizer.get_feature_names_out()

    if hasattr(model, 'coef_'):
        coefficients = model.coef_[0]
        token_indices = vector.nonzero()[1]
        weighted_words = []

        for idx in token_indices:
            influence = float(coefficients[idx] * vector[0, idx])
            weighted_words.append((feature_names[idx], influence))

        weighted_words.sort(key=lambda item: abs(item[1]), reverse=True)
        return [word for word, _ in weighted_words[:top_n]]

    if hasattr(model, 'feature_log_prob_'):
        log_probabilities = model.feature_log_prob_[1] - model.feature_log_prob_[0]
        token_indices = vector.nonzero()[1]
        weighted_words = [
            (feature_names[idx], float(log_probabilities[idx] * vector[0, idx]))
            for idx in token_indices
        ]
        weighted_words.sort(key=lambda item: abs(item[1]), reverse=True)
        return [word for word, _ in weighted_words[:top_n]]

    return []


def predict_news(text: str, model, vectorizer):
    """Predict whether a news text is Real or Fake and return explanation."""
    cleaned = clean_text(text)
    vector = vectorizer.transform([cleaned])

    label_index = int(model.predict(vector)[0])
    label = LABEL_MAP.get(label_index, 'Unknown')

    probability = None
    confidence = 0.0
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(vector)[0]
        fake_probability = float(proba[0])
        real_probability = float(proba[1])
    else:
        score = float(model.decision_function(vector)[0])
        real_prob = float(_sigmoid(score))
        real_probability = real_prob
        fake_probability = 1.0 - real_prob

    confidence = max(fake_probability, real_probability) * 100.0
    probability = real_probability if label_index == 1 else fake_probability
    uncertain = confidence < UNCERTAINTY_THRESHOLD

    keywords = get_top_keywords(text, model, vectorizer, top_n=8)

    logger.info('Input text: %s', text)
    logger.info('Processed text: %s', cleaned)
    logger.info('Vector shape: %s', vector.shape)
    logger.info('Predicted label: %s', label)
    logger.info('Prediction probability: fake=%.4f real=%.4f', fake_probability, real_probability)
    logger.info('Confidence: %.2f%%', confidence)

    return {
        'label': 'Uncertain' if uncertain else label,
        'raw_label': label,
        'uncertain': uncertain,
        'message': 'Prediction uncertain. Please verify this article using trusted news sources.' if uncertain else None,
        'model_used': getattr(model, '_fraud_model_name', model.__class__.__name__),
        'confidence': round(confidence, 2),
        'probability': round(probability * 100.0, 2),
        'probabilities': {
            'Fake': round(fake_probability * 100.0, 2),
            'Real': round(real_probability * 100.0, 2),
        },
        'keywords': keywords,
        'clean_text': cleaned,
    }
