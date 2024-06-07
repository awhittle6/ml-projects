import pandas as pd
from typing import Tuple
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from typing import Dict, Any



def preProcessSentimentData(train: pd.DataFrame, test: pd.DataFrame, **kwargs) -> Tuple:
    xlabel = kwargs.get('xlabel', 'content')
    ylabel = kwargs.get('ylabel', 'sentiment')
    max_features = kwargs.get('max_features', 5000)
    assert isinstance(train, pd.DataFrame) and isinstance(test, pd.DataFrame)
    X_train = train[xlabel]
    y_train = train[ylabel]
    X_test = test[xlabel]
    y_test = test[ylabel]
    vectorizer = TfidfVectorizer(stop_words='english', max_features=max_features)
    X_train_diff = vectorizer.fit_transform(X_train)
    X_test_diff = vectorizer.transform(X_test)
    return X_train_diff, X_test_diff, y_train, y_test


