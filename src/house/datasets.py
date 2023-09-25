from pandas import DataFrame
from sklearn.model_selection import train_test_split


class Datasets:

    # noinspection PyPep8Naming
    def __init__(self, house_data: DataFrame):
        X, y = house_data.drop('Price', axis=1), house_data['Price']
        _sets: tuple[DataFrame, DataFrame, DataFrame, DataFrame] \
            = train_test_split(X, y, test_size=0.3, random_state=1, stratify=X.Zip_loc.array)
        self.X_train, self.X_test, self.y_train, self.y_test = _sets
        self.X_train_encoded = DataFrame()
        self.X_test_encoded = DataFrame()
