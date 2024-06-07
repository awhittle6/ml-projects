import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple, Dict, Union


# Importing the dataset
def import_dataset(csv_file: str, headers: List[str]= None) -> pd.DataFrame:
    df = pd.read_csv(csv_file, names=headers)
    assert isinstance(df, pd.DataFrame)
    return df

# if __name__ == '__main__':
#     import_dataset('sentiment_testing.csv', headers=['id', 'entity', 'sentiment', 'content'])
#     print(train.head())