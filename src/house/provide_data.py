import os
import requests
import sys
import pandas as pd
from numpy import ndarray

from house.datasets import Datasets

DATA_PATH = '../Data'
FILE_NAME = 'house_class.csv'
CSV_PATH = DATA_PATH + '/' + FILE_NAME


def download_data_if_needed():
    if not os.path.exists(DATA_PATH):
        os.mkdir(DATA_PATH)
    if FILE_NAME not in os.listdir(DATA_PATH):
        sys.stderr.write("[INFO] Dataset is loading.\n")
        url = "https://www.dropbox.com/s/7vjkrlggmvr5bc1/house_class.csv?dl=1"
        response = requests.get(url, allow_redirects=True)
        open(CSV_PATH, 'wb').write(response.content)
        sys.stderr.write("[INFO] Loaded.\n")


def load_house_data() -> Datasets:
    download_data_if_needed()
    return Datasets(pd.read_csv(CSV_PATH))


def save_predicted(y_predicted: ndarray, filename: str):
    pd.DataFrame(y_predicted).to_csv(DATA_PATH + '/' + filename, index=False, header=False)
