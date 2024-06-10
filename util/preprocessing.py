import pandas as pd
from typing import Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix


def preProcessSentimentData(train: pd.DataFrame, test: pd.DataFrame, **kwargs) -> Tuple:
    xlabel = kwargs.get('xlabel', 'content')
    ylabel = kwargs.get('ylabel', 'sentiment')
    max_features = kwargs.get('max_features', 5000)
    assert isinstance(train, pd.DataFrame) and isinstance(test, pd.DataFrame)
    train.fillna('', inplace=True)
    X_train = train[xlabel]
    y_train = train[ylabel]
    X_test = test[xlabel]
    y_test = test[ylabel]
    vectorizer = TfidfVectorizer(stop_words='english', max_features=max_features)
    X_train_diff : csr_matrix = vectorizer.fit_transform(X_train)
    X_test_diff : csr_matrix = vectorizer.transform(X_test)
    return X_train_diff, X_test_diff, y_train, y_test


