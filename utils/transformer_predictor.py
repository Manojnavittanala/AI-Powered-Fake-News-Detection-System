"""Optional Hugging Face transformer detector.

This adapter is deliberately lazy: classical TF-IDF inference remains the
zero-install production default. Install transformers and torch, then provide
a fine-tuned sequence-classification model directory to use this backend.
"""


def load_transformer(model_name_or_path):
    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as error:
        raise RuntimeError(
            'Transformer backend is optional. Install transformers and torch first.'
        ) from error

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path)
    return tokenizer, model


def predict_transformer(text, tokenizer, model):
    try:
        import torch
    except ImportError as error:
        raise RuntimeError('Transformer backend requires torch.') from error

    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
    with torch.no_grad():
        logits = model(**inputs).logits
    probabilities = torch.softmax(logits, dim=-1)[0].tolist()
    label_index = int(torch.argmax(logits, dim=-1)[0])
    labels = {0: 'Fake', 1: 'Real'}
    confidence = max(probabilities) * 100
    return {
        'label': labels.get(label_index, 'Unknown') if confidence >= 70 else 'Uncertain',
        'raw_label': labels.get(label_index, 'Unknown'),
        'confidence': round(confidence, 2),
        'probability': round(probabilities[label_index] * 100, 2),
        'probabilities': {
            'Fake': round(probabilities[0] * 100, 2),
            'Real': round(probabilities[1] * 100, 2),
        },
        'model_used': 'transformer',
        'uncertain': confidence < 70,
        'message': 'Prediction uncertain. Please verify this article using trusted news sources.' if confidence < 70 else None,
        'keywords': [],
    }
