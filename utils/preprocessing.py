import html
import re
from functools import lru_cache

from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

stopwords = set(ENGLISH_STOP_WORDS)
MIN_ARTICLE_LENGTH = 100
lemmatizer = WordNetLemmatizer()

CONTRACTIONS = {
    "can't": 'cannot', "won't": 'will not', "n't": ' not',
    "'re": ' are', "'ve": ' have', "'ll": ' will',
    "'d": ' would', "'m": ' am', "'s": ' is',
}


def _expand_contractions(text: str) -> str:
    for contraction, expansion in CONTRACTIONS.items():
        text = re.sub(re.escape(contraction), expansion, text, flags=re.IGNORECASE)
    return text


@lru_cache(maxsize=8192)
def _lemmatize(token: str) -> str:
    try:
        return lemmatizer.lemmatize(token)
    except LookupError:
        # WordNet is optional at runtime; the normalized token remains stable
        # when the local NLTK corpus is unavailable.
        return token


def clean_text(text: str) -> str:
    """Normalize and clean a news text string."""
    if not isinstance(text, str):
        return ''

    text = html.unescape(text).lower()
    text = _expand_contractions(text)
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    text = re.sub(r'\b[\w.+-]+@[\w-]+\.[\w.-]+\b', ' ', text)
    text = text.encode('ascii', 'ignore').decode('ascii')
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r"\s+", ' ', text).strip()

    tokens = [word for word in text.split() if word not in stopwords and len(word) > 2]
    return ' '.join(_lemmatize(token) for token in tokens)


def prepare_corpus(text_series):
    """Apply cleaning to a pandas Series of text values."""
    return text_series.fillna('').apply(clean_text)


def validate_news_text(text: str):
    """Return a user-facing validation error, or None for usable article text."""
    if not isinstance(text, str) or not text.strip():
        return 'Please provide a news article.'

    normalized = text.strip()
    if len(normalized) < MIN_ARTICLE_LENGTH:
        return 'Please provide at least 100 characters of article text.'
    if not re.search(r'[A-Za-z]', normalized):
        return 'Please provide an article containing words, not numbers only.'

    words = re.findall(r'[A-Za-z]+', normalized.lower())
    if len(set(words)) < 4 or len(set(normalized.lower())) < 12:
        return 'The text does not look like a news article. Please provide more varied text.'
    if re.search(r'(?:qwerty|asdfgh|zxcvbn|(.)\1{5,})', normalized.lower()):
        return 'The text does not look like a news article.'
    if re.search(r'(<script|</?html|function\s*\(|select\s+.+\s+from|lorem ipsum)', normalized.lower()):
        return 'Please provide a news article, not code or placeholder text.'

    return None
