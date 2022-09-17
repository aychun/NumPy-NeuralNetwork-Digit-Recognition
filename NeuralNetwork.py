"""
Class to implement machine laerning neural network with only NumPy. 
"""


import numpy as np
from typing import List, Optional, Tuple
from DigitData import DigitData
from pathlib import Path


class NeuralNetwork:

    weights: List[np.ndarray]
    biases: List[np.ndarray]
    layer_sizes: List[int]
    layer_num = int
    input_shape: np.ndarray
    accuracy: float

    def __init__(self, *layer_size_args: int) -> None:
        """
        Initialize the NeuralNetwork with the number of neurons in each layer
        given in <layer_size_args>
        """

        input_size = layer_size_args[0]
        self.input_shape = (input_size, 1)
        self.layer_sizes = list(layer_size_args)
        self.layer_num = len(layer_size_args)
        self.accuracy = 0.0

        self.weights = []
        self.biases = []

        for layer_size in layer_size_args[1:]:
            weight_shape = (layer_size, input_size)
            w = np.random.randn(weight_shape[0], weight_shape[1])
            b = np.random.randn(layer_size, 1)

            input_size = layer_size

            self.weights.append(w)
            self.biases.append(b)

    def forward_prop(self, a: np.ndarray) -> np.ndarray:
        """
        Forward propagation of the neural netwrok given the input <a>
        """

        if np.shape(a) != self.input_shape:
            raise ValueError("Size (shape) of the input does not match with the NN")

        for w, b in zip(self.weights, self.biases):
            z = w @ a + b
            a = NeuralNetwork.sigmoid(z)

        return a

    def stochastic_gradient_descent(
        self,
        epochs: int,
        batch_size: int,
        learning_rate: float,
        training_set: List[DigitData],
        validation_set: Optional[List[DigitData]] = None,
        save_network: bool = False,
    ) -> None:
        """
        Train the neural network with the stochastic gradient descent method.
        The algorithm was adopted and modified from Michael Nielsen's free online textbook
        "Neural Networks and Deep Learning" (http://neuralnetworksanddeeplearning.com/index.html)
        and it was re-written for the purpose of personal study.
        """

        for i in range(epochs):
            np.random.shuffle(training_set)

            batches = []

            for j in range(0, len(training_set), batch_size):
                batches.append(training_set[j : j + batch_size])

            for batch in batches:
                self.train_network(batch, learning_rate)

            print(f"training the network ... {i+1}/{epochs} ({int((i+1)/epochs*100)}%)")
            if validation_set:
                print(
                    f"Accuracy: {self.evaluate_testset(validation_set)} / {len(validation_set)}"
                )

        if save_network:
            self.save_trained_network()

    def train_network(
        self,
        training_data: List[DigitData],
        learning_rate: float,
        print_process: bool = False,
        print_cost: bool = False,
    ) -> None:
        """
        Train the neural network with the <training_data> and learning rate <learning_rate>.
        """

        dw_sum = [np.zeros(w.shape, dtype=np.float64) for w in self.weights]
        db_sum = [np.zeros(b.shape, dtype=np.float64) for b in self.biases]

        count = 0
        for data in training_data:
            db, dw = self.get_gradients(data)

            db_sum = [nb + dnb for nb, dnb in zip(db_sum, db)]
            dw_sum = [nw + dnw for nw, dnw in zip(dw_sum, dw)]

            count += 1
            if print_process:
                print(f"Used data: {count} / {len(training_data)}")

            if print_cost:
                self._print_quad_cost(data)

        lr = learning_rate / len(training_data)

        self.weights = [w - lr * dw for w, dw in zip(self.weights, dw_sum)]
        self.biases = [b - lr * db for b, db in zip(self.biases, db_sum)]

    def get_gradients(
        self, DigitData: DigitData
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Return the gradient of the quadratic cost function respect to the weights and biases given
        a sigle input from <DigitData>
        """

        zs = []
        input = DigitData.getPixelValues()
        activations = [input]
        y = DigitData.getLabel()

        for w, b in zip(self.weights, self.biases):
            z = w @ input + b
            zs.append(z)
            input = NeuralNetwork.sigmoid(z)
            activations.append(input)

        out = input

        # Last layer
        delta = NeuralNetwork.quadratic_cost_derivative(
            y, out
        ) * NeuralNetwork.sigmoid_prime(zs[-1])

        dw = [np.zeros(w.shape, dtype=np.float64) for w in self.weights]
        db = [np.zeros(b.shape, dtype=np.float64) for b in self.biases]

        db[-1] = delta
        dw[-1] = delta @ activations[-2].T

        for i in range(2, self.layer_num):
            z = zs[-i]
            delta = (self.weights[-i + 1].T @ delta) * NeuralNetwork.sigmoid_prime(z)

            db[-i] = delta
            dw[-i] = delta @ activations[-i - 1].T

        return db, dw

    def evaluate_testset(self, test_set: List[DigitData]) -> int:
        """
        Evaluate the network with a test set (validation set) <test_set>.
        Update <self.accuracy> and return the number of inputs that were correctly classified.
        """

        count = 0

        for data in test_set:
            if self.evaluate(data):
                count += 1

        self.accuracy = (round(count / len(test_set), 4)) * 100

        return count

    def evaluate(self, test_data: DigitData) -> bool:
        """
        Evaluate the network with a single test data <test_data>
        and return true if it was correctly classified
        """

        x = test_data.getPixelValues()

        return np.argmax(self.forward_prop(x)) == np.argmax(test_data.getLabel())

    @staticmethod
    def sigmoid(z: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-z))

    @staticmethod
    def sigmoid_prime(z: np.ndarray) -> np.ndarray:
        s = NeuralNetwork.sigmoid(z)
        return s * (1 - s)

    def _print_quad_cost(self, data: DigitData) -> None:
        """
        Used for testing and debugging

        Print the value of quadratic cost function given a signle Digitdata <data>
        """
        prediction = self.forward_prop(data.getPixelValues())

        y = data.getLabel()

        print("Quadratic Cost function: ")
        print(NeuralNetwork._quadratic_cost([prediction], [y]))

    @staticmethod
    def _quadratic_cost(
        target_list: List[np.ndarray], output_list: List[np.ndarray]
    ) -> float:
        """
        Used for testing and debugging

        Return the quadratic cost given a list of desired outputs <target_list>
        and the actual outputs from <output_list>
        """

        sum = 0

        for y, a in zip(target_list, output_list):
            sum += np.linalg.norm(y - a) ** 2

        return 1 / (2 * len(target_list)) * sum

    @staticmethod
    def quadratic_cost_derivative(target: np.ndarray, output: np.ndarray) -> np.ndarray:
        """
        Return the derivative of the quadratic cost function.
        """
        return output - target

    def save_trained_network(self) -> None:
        """
        Save the weights and biases of this network as a .npz file.
        """

        path = Path(
            f"npz_files/{self.layer_sizes}_ACCURACY{round(self.accuracy, 4)}.npz"
        )

        np.savez(
            path, layers=self.layer_sizes, weights=self.weights, biases=self.biases
        )

    def load_npz_file(self, filepath: str) -> None:
        """
        Load saved weights and biases from a .npz file.
        """

        data = np.load(Path(filepath), allow_pickle=True)
        self.weights = data["weights"]
        self.biases = data["biases"]

        self.layer_sizes = data["layers"]
        self.layer_num = len(self.layer_sizes)
        self.input_shape = (self.layer_sizes[0], 1)

    @staticmethod
    def softmax(Z: np.ndarray) -> np.ndarray:
        return np.exp(Z) / np.sum(np.exp(Z))

    def __str__(self) -> str:

        return "Weights: \n" + str(self.weights) + "\nBiases: \n" + str(self.biases)


if __name__ == "__main__":
    pass
