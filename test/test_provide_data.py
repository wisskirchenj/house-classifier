import unittest
from unittest.mock import patch
from io import StringIO

import house.provide_data
import house.classifier


class ProvideDataTest(unittest.TestCase):

    @patch('builtins.open')
    @patch('requests.get')
    @patch('os.listdir')
    @patch('os.mkdir')
    @patch('sys.stderr', new_callable=StringIO)
    def test_no_data_dir_gets_and_loads(self, mock_stderr, mock_mkdir, mock_listdir, mock_get, mock_open):
        house.provide_data.DATA_PATH = "not_there"
        mock_listdir.return_value = tuple()
        house.provide_data.download_data_if_needed()
        # directory does not exist, so mkdir must create it = mock_mkdir is called once
        mock_mkdir.assert_called_once()
        mock_listdir.assert_called_once()
        mock_get.assert_called()
        mock_open.assert_called()
        lines = mock_stderr.getvalue().splitlines()
        self.assertEqual(2, len(lines))
        self.assertEqual('[INFO] Dataset is loading.', lines[0])
        self.assertEqual('[INFO] Loaded.', lines[1])

    # noinspection PyUnusedLocal
    @patch('requests.get')
    @patch('os.listdir')
    @patch('os.mkdir')
    def test_data_exists_no_load(self, mock_mkdir, mock_listdir, mock_get):
        house.provide_data.FILE_NAME = "house_class_test.csv"
        mock_listdir.return_value = ('house_class_test.csv', )
        house.provide_data.download_data_if_needed()
        mock_listdir.assert_called_once()
        mock_get.assert_not_called()

    def test_read_csv_loads_data_correctly(self):
        house.provide_data.DATA_PATH = "test/Data"
        house.provide_data.FILE_NAME = "house_class_test.csv"
        house.provide_data.CSV_PATH = "test/Data/house_class_test.csv"
        data = house.provide_data.load_house_data()
        self.assertTupleEqual((639, 6), data.X_train.shape)
        self.assertTupleEqual((275, 6), data.X_test.shape)
        self.assertTupleEqual((639, ), data.y_train.shape)
        self.assertTupleEqual((275, ), data.y_test.shape)

    @patch('sys.stdout', new_callable=StringIO)
    def test_classify_on_test_data_works(self, mock_stdout):
        house.provide_data.DATA_PATH = "test/Data"
        house.provide_data.FILE_NAME = "house_class_test.csv"
        house.provide_data.CSV_PATH = "test/Data/house_class_test.csv"
        house.classifier.classify()
        lines = mock_stdout.getvalue().splitlines()
        self.assertEqual(3, len(lines))
        self.assertEqual(lines[0], 'OneHotEncoder:0.64')
        self.assertEqual(lines[1], 'OrdinalEncoder:0.86')
        self.assertEqual(lines[2], 'TargetEncoder:0.75')
