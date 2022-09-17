# NumPy-NeuralNetwork-Digit-Recognition

Personal machine learning project to implement a neural network with only NumPy and
an interactive GUI with Tkinter for testing and fun.

The project is focused on MNIST dataset recognition, yet the inner layers of the program
could be reused for other machine learning classification models.
The outer layers (mostly GUI classes) will require some modifications to be adapted.

## File Descriptions

`NeuralNetwork.py`\
Trainable neural network class for classification models using the stochastic gradient descent algorithm for learning. Weights and biases are stored as NumPy arrays.

`DigitData.py`\
Class to represent a single MNIST datum. Its pixel values and label are represented as (n, 1) NumPy arrays.

`DatasetLoader.py`\
Class to load the dataset. Read and stores the MNIST handwritten digit database from .csv files. Creates DigitData objects to be used for the NeuralNetwork class.

`train_network.py`\
Python script to train and save the NeuralNetwork.

`DrawingBoard.py` and `gui.py`\
Classes and functions to implement the interactive GUI.

`main.py`\
Run to start the program.

## Training the network

Running `train_network.py` will load the dataset and create lists of DigitData objects

```Python
train_set = trainDataLoader.CreateListofData()
validation_set = testDataLoader.CreateListofData()
```
Then a neural network with randomly initialized weights and biases is instantiated with 4 layers consisting of 784 input neurons, two hidden layers with 30 and 20 nuerons, and output layer of 10 neurons.
```Python
nn = NeuralNetwork(784, 30, 20, 10)
```
Before any training, the network has an accuracy of 7.85%
```Python
Before Training:
785 out of 10000 data were classified correctly
```

Then we trained the network over 300 iterations with the batch size of 15 and learning rate of 0.06. (The hyperparameters were determined by trial and error)
```Python
nn.stochastic_gradient_descent(300, 15, 0.06, train_set, validation_set)
```
```Python
training the network ... 1/20 (5%)
Accuracy: 2564 / 10000
training the network ... 2/20 (10%)
Accuracy: 3947 / 10000
.
.
.
training the network ... 299/300 (99%)
Accuracy: 8521 / 10000
training the network ... 300/300 (100%)
Accuracy: 8541 / 10000
```

```Python
After Training
8541 out of 10000 data were classified correctly
```

## GUI

Running `main.py` will launch the main application
<p align="center">
  <img src=https://user-images.githubusercontent.com/85460898/190877123-9793ba52-f644-4a09-85ed-060e18dad185.png />
</p>

Draw a number on the white drawing board and press the Read button. The program will update the labels according to its evaluation and the grayscale image of the input will also show on a separate window.
<p align="center">
  <img src=https://user-images.githubusercontent.com/85460898/190878066-243d2068-d8e0-440a-b557-be8f948fa0f5.png >
  <img src=https://user-images.githubusercontent.com/85460898/190878068-1a928bc0-82a8-4261-bd74-338ddbe31404.png >
</p>

### Note

The program records the movement of the mouse and translate its coordinates on the drawing board to 28x28 grayscale input. When the image is drawn too fast it can lead to ill-formatted inputs.

![](https://user-images.githubusercontent.com/85460898/190878755-e0982ef7-8adc-4a70-b1a6-3203d88aacf1.png)            |  ![](https://user-images.githubusercontent.com/85460898/190878756-415785dc-490a-462d-98ea-4bab8c6c99fd.png)
:-------------------------:|:-------------------------:
![](https://user-images.githubusercontent.com/85460898/190878752-8217f75e-abc6-411e-a30d-7948553b51f6.png)  |  ![](https://user-images.githubusercontent.com/85460898/190878754-a842e668-84ed-4edd-a836-6f5fe730ade4.png)


## Result


![](https://user-images.githubusercontent.com/85460898/190879040-c73d596d-56ee-4659-9a3e-3a639993ad4a.png)            |  ![](https://user-images.githubusercontent.com/85460898/190879039-ae389353-8797-4f17-994c-e479d64b8d2c.png)
:-------------------------:|:-------------------------:
![](https://user-images.githubusercontent.com/85460898/190879049-f5423a15-760b-4a2a-9a59-38a58689873f.png)            |  ![](https://user-images.githubusercontent.com/85460898/190879050-35045d2e-835e-4cd6-9c7d-3bf98e23cdf9.png)
![](https://user-images.githubusercontent.com/85460898/190879124-503205cc-c9fe-4d8b-9508-1948c6291e0d.png)            |  ![](https://user-images.githubusercontent.com/85460898/190879126-cfd1dbd7-9c4f-4199-82d9-3d35f6e4fe64.png)
![](https://user-images.githubusercontent.com/85460898/190879054-90473d7b-f21f-4f30-8e14-2f0327022a23.png)            |  ![](https://user-images.githubusercontent.com/85460898/190879053-1c668917-5e71-47dd-ba7b-e20648a07916.png)
![](https://user-images.githubusercontent.com/85460898/190879056-3dc5ae78-516a-4e65-ad22-65e0aadc6754.png)            |  ![](https://user-images.githubusercontent.com/85460898/190879058-bc9c27cc-3f4c-43f9-a434-2e104712ae76.png)
![](https://user-images.githubusercontent.com/85460898/190879061-4b537ae1-e3e5-417a-85ad-4e49eddb5780.png)            |  ![](https://user-images.githubusercontent.com/85460898/190879060-e0d3be87-352e-4c41-94d2-a2d77834b061.png)
![](https://user-images.githubusercontent.com/85460898/190879065-60078274-3a98-4c96-8e9a-3b6180ca6756.png)            |  ![](https://user-images.githubusercontent.com/85460898/190879064-caa45a2d-ceda-4583-85a9-0d20556bb1e8.png)
![](https://user-images.githubusercontent.com/85460898/190879068-32f9af6f-1ac4-4f3a-801d-7c87fa5a9210.png)            |  ![](https://user-images.githubusercontent.com/85460898/190879067-6b2f5250-49a4-4c7b-9669-2d50029f2fff.png)
![](https://user-images.githubusercontent.com/85460898/190879073-bf85a821-97f6-4359-a349-1e0dcb2b7338.png)            |  ![](https://user-images.githubusercontent.com/85460898/190879072-3b39ecb7-d646-4342-88c6-b66661ec1f4b.png)
![](https://user-images.githubusercontent.com/85460898/190879074-47867430-1511-4f1c-a77d-ddbb76cd0ee4.png)            |  ![](https://user-images.githubusercontent.com/85460898/190879075-7d1219b2-e04c-4a59-87bd-2cc983359f92.png)

Drawing slowly on the centre, the network correctly guessed 8/10 of my handwritten digits.


## Sources and References
MNIST dataset https://www.kaggle.com/datasets/oddrationale/mnist-in-csv \
Rescaling Numpy Array
https://stackoverflow.com/questions/8090229/resize-with-averaging-or-rebin-a-numpy-2d-array \
Backpropagation
http://neuralnetworksanddeeplearning.com/index.html \
Drawing on Tk.Canvas https://stackoverflow.com/questions/70403360/drawing-with-mouse-on-tkinter-canvas
