import pytest
from utils.preprocessing import clean_text


def test_clean_text_removes_stopwords_and_punctuation():
    text = 'Breaking News: This is a Fake! story with numbers 123.'
    cleaned = clean_text(text)
    assert 'breaking' in cleaned
    assert 'news' in cleaned
    assert 'fake' in cleaned
    assert 'story' in cleaned
    assert '123' not in cleaned
    assert 'this' not in cleaned
    assert 'is' not in cleaned


def test_clean_text_returns_empty_for_non_string():
    assert clean_text(None) == ''
    assert clean_text(12345) == ''
