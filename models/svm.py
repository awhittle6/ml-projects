from sklearn.svm import SVC
from sklearn.metrics import classification_report
import pandas as pd


class SentimentAnalysisSVM:
    def __init__(self, X_train : pd.Series, X_test: pd.Series, y_train: pd.Series, y_test: pd.Series):
        print(type(X_train))
        # assert isinstance(X_train, pd.Series)
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def train(self, c=1):
        self.svm = SVC(kernel='linear', C=c)
        self.svm.fit(self.X_train, self.y_train)

    def predict(self):
        self.y_pred = self.svm.predict(self.X_test)

    def evaluate(self):
        return classification_report(self.y_test, self.y_pred)