"""
The mnist_train.csv file contains the 60,000 training examples and labels. 
The mnist_test.csv contains 10,000 test examples and labels. 
Each row consists of 785 values: the first value is the label (a number from 0 to 9) and the remaining 784 values are the pixel values (a number from 0 to 255).

Source:
https://www.kaggle.com/datasets/oddrationale/mnist-in-csv
"""

from typing import List
from numpy import ndarray, loadtxt
from DigitData import DigitData

trainFilePath = "dataset/mnist_train.csv"
testFilePath = "dataset/mnist_test.csv"


class DatasetLoader:

    file_path: str
    matrix_data: ndarray
    num_data: int

    def __init__(self, file_path: str) -> None:
        """
        Initialize the DatasetLoader for a dataset saved as <file_path>
        """
        self.file_path = file_path
        self.loadCSVfile()
        self.num_data = len(self.matrix_data)

    def loadCSVfile(self) -> None:
        """
        Load data from the CSV dataset and save it as a 2-D NumPy array.
        """
        data = loadtxt(open(self.file_path, "rb"), delimiter=",", skiprows=1)
        self.matrix_data = data

    def _CreateDigitData(self, row: int) -> DigitData:
        """
        Return a DigitData object from row <row> of <self.matrix_data>
        """
        data = self.matrix_data[row]
        label = data[0]
        pixels = data[1:]

        return DigitData(int(label), pixels)

    def CreateListofData(self) -> List[DigitData]:
        """
        Return a list of DigitData objects from the entire dataset
        """
        data_set = [self._CreateDigitData(i) for i in range(testDataLoader.num_data)]
        return data_set


trainDataLoader = DatasetLoader(trainFilePath)
testDataLoader = DatasetLoader(testFilePath)


if __name__ == "__main__":
    pass
