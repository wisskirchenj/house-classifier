import unittest
from unittest.mock import patch
from io import StringIO
from pandas.testing import assert_series_equal
import pandas

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
        self.assertTupleEqual((5, 7), data.shape)
        assert_series_equal(data['Price'], pandas.Series([0, 0, 1, 0, 1]), check_names=False)
        self.assertEqual(4.850476, data.at[1, 'Lon'])

    @patch('sys.stdout', new_callable=StringIO)
    def test_classify_on_test_data_works(self, mock_stdout):
        house.provide_data.DATA_PATH = "test/Data"
        house.provide_data.FILE_NAME = "house_class_test.csv"
        house.provide_data.CSV_PATH = "test/Data/house_class_test.csv"
        data = house.classifier.classify()
        lines = mock_stdout.getvalue().splitlines()
        self.assertEqual('5', lines[0])
        self.assertEqual('7', lines[1])
        self.assertEqual('False', lines[2])
        self.assertEqual('6', lines[3])
        self.assertEqual('99.8', lines[4])
        self.assertEqual('5', lines[5])
