from typing import Callable

from pandas import DataFrame
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import DecisionTreeClassifier

from house.datasets import Datasets
from house.provide_data import load_house_data, save_predicted
from house.encode import target_encode, one_hot_encode, ordinal_encode


def classify():
    data = load_house_data()
    fit_and_print_macro_f1(data, one_hot_encode, 'OneHot')
    fit_and_print_macro_f1(data, ordinal_encode, 'Ordinal')
    fit_and_print_macro_f1(data, target_encode, 'Target')


def fit_and_print_macro_f1(data: Datasets, encode_func: Callable[[Datasets], Datasets], encoder_type: str):
    data = encode_func(data)
    classifier = DecisionTreeClassifier(criterion='entropy', max_features=3, splitter='best',
                                        max_depth=6, min_samples_split=4, random_state=3)
    classifier.fit(data.X_train_encoded, data.y_train)
    y_test_predicted = classifier.predict(data.X_test_encoded)
    macro_f1 = extract_macro_f1(data, y_test_predicted)
    print(f'{encoder_type}Encoder:{macro_f1}')


def extract_macro_f1(data, y_test_predicted):
    df = DataFrame({'True': data.y_test, 'Predicted': y_test_predicted})
    report_lines = classification_report(df['True'], df['Predicted']).splitlines()
    macro_line = list(filter(lambda s: 'macro' in s, report_lines))[0]
    macro_f1 = macro_line.split()[-2]
    return macro_f1


if __name__ == '__main__':
    classify()
