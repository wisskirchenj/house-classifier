import pandas as pd

from house.provide_data import load_house_data


def classify():
    house_data = load_house_data()
    print(house_data.head())


if __name__ == '__main__':
    classify()