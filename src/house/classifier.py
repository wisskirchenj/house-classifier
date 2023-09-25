from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from house.provide_data import load_house_data, save_predicted
from house.encode import target_encode


def classify():
    data = load_house_data()
    data = target_encode(data)
    classifier = DecisionTreeClassifier(criterion='entropy', max_features=3, splitter='best',
                                        max_depth=6, min_samples_split=4, random_state=3)
    classifier.fit(data.X_train_encoded, data.y_train)
    y_test_predicted = classifier.predict(data.X_test_encoded)
    save_predicted(y_test_predicted, 'price_predicted_target.csv')
    print(accuracy_score(data.y_test, y_test_predicted).__round__(4))


if __name__ == '__main__':
    classify()
