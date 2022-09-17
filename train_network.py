"""
Train NueralNetwork with the MNIST dataset. The size of the neural network and 
the hyperparameters used in learning were somewhat arbitrarily decided by trial and error.
"""

from DatasetLoader import trainDataLoader, testDataLoader
from NeuralNetwork import NeuralNetwork

if __name__ == "__main__":

    train_set = trainDataLoader.CreateListofData()
    validation_set = testDataLoader.CreateListofData()

    nn = NeuralNetwork(784, 30, 20, 10)

    print("Before Training:")
    print(
        f"{nn.evaluate_testset(validation_set)} out of {len(validation_set)} data were classified correctly"
    )

    nn.stochastic_gradient_descent(300, 15, 0.06, train_set, validation_set)
    nn.save_trained_network()

    print("After Training")
    print(  
        f"{nn.evaluate_testset(validation_set)} out of {len(validation_set)} data were classified correctly"
    )
