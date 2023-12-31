# IDEA EDU Course

Project implemented in the track 'Machine Learning Algorithms from Scratch' of hyperskill.org JetBrains Academy. 
Purpose of doing this project is to work with Pandas and get to know some more Machine Learning.

## Technology / External Libraries

- Python 3.11.5,
- with packages: pandas, scikit-learn, requests, unittest, unittest.mock, parameterized

## Repository Contents

The sources of main project tasks (6 stages) and pytest testing.

## Installation

Install the wheel file using pip after clone/fork:

    > pip install dist/house-classifier-3.11.5-py3-none-any.whl

## Program description

House classifier is a Python-based app, that implements a basic workflow of training a machine learning algorithm,
from importing the data to evaluating the model’s performance. It makes use of different data encoders.

Welcome to a real estate company located in Amsterdam. Your supervisor needs an ML model that predicts the price category
of a house based on various parameters. The key problem is, that there is too much categorical data. In this project,
we work with the input data and apply ready-to-use machine learning (decision tree) algorithms to the data.

Have fun!

## Project progress

Project was completed on 25.09.23

23.09.23 Project started & Stage 1 completed - Load CSV-data using requests. Extract required information from the data
with Pandas DataFrame methods. Use unittest.mock.patch to mock input() in pytest.

24.09.23 Stage 2 completed - use `sklearn test_train_split` to split data into train and test data sets. Provide the
Datasets class to store these sets.

25.09.23 Stage 3 completed - use `OneHotEncoder` to encode categorical data. Use `DecisionTreeClassifier` to train a
model on the train data set. Use `accuracy_score` to evaluate the model on the test data set.

25.09.23 Stage 4 completed - use `OrdinalEncoder` to encode categorical data this time - yields the best score of the 3.
Use `DecisionTreeClassifier` and `accuracy_score` as before.
    
25.09.23 Stage 5 completed - use `TargetEncoder` to encode categorical data this time.
Use `DecisionTreeClassifier` and `accuracy_score` as before.

25.09.23 Final Stage 6 completed - use `sklearn.metrics classification_report` to get a detailed report on the
model's performance. Compare the macro F1-score of the 3 encoding variant models.