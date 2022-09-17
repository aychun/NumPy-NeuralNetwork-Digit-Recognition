"""
Personal machine learning project to implement a neural network with only NumPy and 
an interactive GUI with Tkinter for testing and fun. 

The project is focused on MNIST dataset recognition, yet the inner layers of the program 
could be reused for other machine learning classification models.
The outer layers (mostly GUI classes) will require some modifications to be adapted. 

To use a differently trained network, run <train_network.py> and load the new .npz file by 
changing <npz_file> variable inside the __main__ block. 
Some other presets are also provided in the "npz_files" directory. 

All of the testings were done only on a MacOS and although pathlib.Path() was used for 
reading saved files, one might need to retype the file path on Windows OS.  

Some of the code was modified and/or extended from online resources. The link to the original
resource is provided in those parts. 

More information on the GitHub repository 
https://github.com/aychun/NumPy-NeuralNetwork-Digit-Recognition

Written by Andrew Yooeun Chun <https://github.com/aychun>
2022/September/14
"""

from gui import App

if __name__ == "__main__":

    npz_file = "npz_files/[784, 30, 20, 10]_ACCURACY88.02.npz"

    app = App()
    app.load_widets()
    app.setup_nn(npz_file)
    app.run()
