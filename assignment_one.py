from util.initialize import *
from util.preprocessing import *
from pathlib import Path
from models.svm import SentimentAnalysisSVM
from models.knn import KNN
from models.neural import NNAgent



def main():
    # Import dataset
    base_path  = Path.cwd()
    data_path = base_path / 'data'
    sentiment_train = data_path / 'sentiment_testing.csv'
    sentiment_test = data_path / 'sentiment_testing.csv'
    train_df = import_dataset(sentiment_train, headers=['id', 'entity', 'sentiment', 'content'])
    test_df = import_dataset(sentiment_test, headers=['id', 'entity', 'sentiment', 'content'])
    X_train, X_test, y_train, y_test = preProcessSentimentData(train_df, test_df)
    svm = SentimentAnalysisSVM(X_train, X_test, y_train, y_test)
    svm.train()
    svm.predict()
    knn = KNN(X_train, X_test, y_train, y_test)
    knn.train()
    knn.predict()
    nna = NNAgent(X_train, X_test, y_train, y_test, X_train.shape[1])
    print(svm.evaluate())
    print(knn.evaluate())





if __name__ == '__main__':
    main()
    # import_dataset(P'sentiment_testing.csv', headers=['id', 'entity', 'sentiment', 'content'])