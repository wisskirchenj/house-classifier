from house.provide_data import load_house_data


def classify():
    data = load_house_data()
    print(data.X_train['Zip_loc'].value_counts().to_dict())


if __name__ == '__main__':
    classify()
