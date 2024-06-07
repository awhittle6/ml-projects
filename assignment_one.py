from util.initialize import import_dataset
from pathlib import Path




def main():
    # Import dataset
    base_path  = Path.cwd()
    data_path = base_path / 'data'
    sentiment_train = data_path / 'sentiment_testing.csv'
    sentiment_test = data_path / 'sentiment_testing.csv'
    train_df = import_dataset(sentiment_train, headers=['id', 'entity', 'sentiment', 'content'])
    test_df = import_dataset(sentiment_test, headers=['id', 'entity', 'sentiment', 'content'])
    print(type(train_df['content']))





if __name__ == '__main__':
    main()
    # import_dataset(P'sentiment_testing.csv', headers=['id', 'entity', 'sentiment', 'content'])