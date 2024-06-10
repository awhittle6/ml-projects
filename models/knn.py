from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
import pandas as pd

class KNN:
    def __init__(self, X_train : pd.Series, X_test: pd.Series, y_train: pd.Series, y_test: pd.Series):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
    
    def train(self, n_neighbors=5):
        self.knn = KNeighborsClassifier(n_neighbors=n_neighbors)
        self.knn.fit(self.X_train, self.y_train)

    def predict(self):
        self.y_pred = self.knn.predict(self.X_test)
    
    def evaluate(self):
        return classification_report(self.y_test, self.y_pred)