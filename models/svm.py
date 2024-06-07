import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# Importing the dataset
def import_dataset():
    global train, test
    train = pd.read_csv('sentiment_training.csv').headers = []
    test = pd.read_csv('sentiment_testing.csv')

if __name__ == '__main__':
    import_dataset()
    print(train)