"""
Class to represent a single MNIST handwritten digit data.
"""
from dataclasses import dataclass
from numpy import ndarray, zeros, reshape


@dataclass
class DigitData:

    label: ndarray         # (10, 1) NumPy array
    pixel_values: ndarray  # 28x28 pixels represented as a (784,1) NumPy array

    def __init__(self, label: int, pvs: ndarray) -> None:
        """
        Initialize a DigitData object with its <label> and its flattended (1,n) NumPy array <pvs>.
        """
        label_array = zeros((10, 1))
        label_array[label] = 1.0

        self.label = label_array
        self.pixel_values = DigitData._reformat_array(pvs)

    @staticmethod
    def _reformat_array(array: ndarray) -> ndarray:
        """
        Reformat a Numpy array to (n, 1) shape
        """
        return reshape(array, (len(array), 1))

    def getLabel(self) -> ndarray:
        return self.label

    def getPixelValues(self) -> ndarray:
        return self.pixel_values


if __name__ == "__main__":
    pass
