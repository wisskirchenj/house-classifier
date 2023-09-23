from house.provide_data import load_house_data


def classify():
    house_data = load_house_data()
    print(house_data.shape[0])
    print(house_data.shape[1])
    print(house_data.isna().any().any())
    print(house_data['Room'].max())
    print(house_data['Area'].mean().round(1))
    print(house_data['Zip_loc'].nunique())


if __name__ == '__main__':
    classify()
