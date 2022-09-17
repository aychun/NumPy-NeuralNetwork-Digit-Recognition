import tkinter as tk
from typing import Optional
from DrawingBoard import DrawingBoard
from numpy import ndarray, argmax, zeros
from NeuralNetwork import NeuralNetwork

def setup_app(app_width: int, app_height: int, app_title: str) -> tk.Tk:
    """
    Function to return a tk.Tk object for initializing the GUI with the
    width of the application <app_width> and height <app_height> with the
    title <app_title>.
    """
    root = tk.Tk()

    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()

    app_width = app_width
    app_height = app_height

    app_x = (screen_width - app_width) // 2
    app_y = (screen_height - app_height) // 2

    root_geom = f"{app_width}x{app_height}+{app_x}+{app_y}"

    root.geometry(root_geom)
    root.title(app_title)

    return root


class App:
    """
    Class to implement the actual application with interactive GUI.
    """

    root: tk.Tk
    nn: NeuralNetwork
    db: DrawingBoard

    indicator: tk.Label
    label: tk.Label

    def __init__(
        self,
        app_width: int = 500,
        app_height: int = 550,
        app_title: str = "Digit recognition with only NumPy",
    ) -> None:
        """
        Initialize the class by calling the <setup_app> function and creating a
        NeuralNetwork> object and a DrawingBoard object.
        Note that the arguments used in NeuralNetwork() depends on the saved .npz file.
        """

        self.root = setup_app(app_width, app_height, app_title)
        self.nn = NeuralNetwork(784, 30, 20, 10)
        self.db = DrawingBoard()

    def setup_nn(self, saved_file: Optional[str] = None) -> None:
        """
        Set up the neural network by loading the trained weights and biases from the
        .npz file <saved_file>
        """

        if saved_file:
            self.nn.load_npz_file(saved_file)
        else:
            self.nn.load_npz_file("npz_files/[784, 30, 20, 10]_ACCURACY88.82.npz")

    def read_btn(self) -> None:
        """
        Function to be called when the read button is clicked.
        """

        a = self._reshape_pixels()
        prediction = self.nn.forward_prop(a)

        indicator_str = self._prediction_to_string(prediction)

        self._update_indicator(indicator_str)
        self._update_label(argmax(prediction))

        App.show_gray_img(a)

    def reset_btn(self) -> None:
        """
        Function to be called when the Reset button is clicked.
        """
        self.db.reset_canvas()
        self._update_indicator(self._initial_indicator_string())

    def _initial_indicator_string(self) -> str:
        """
        Return the default string for the self.indicator label.
        """

        indicator_str = self._prediction_to_string(zeros((10, 1)), True)

        return indicator_str

    def load_widets(self):
        """
        Load the widets used in the tk.Tk
        """

        self.db.canvas.pack()

        indicator_str = self._initial_indicator_string()
        self.indicator = tk.Label(self.root, text=indicator_str)
        self.label = tk.Label(self.root, text="Neural Network thinks the writing is")

        self.indicator.pack()
        self.label.pack()

        cmd = lambda: self.reset_btn()
        reset_button = tk.Button(self.root, text="Reset", command=cmd)

        reset_button.pack()

        cmd = lambda: self.read_btn()
        read_button = tk.Button(self.root, text="Read", command=cmd)

        read_button.pack()

    def _rescale_pixels(self) -> ndarray:
        """
        Downsacle the pixel array into 28x28.
        Modified code from
        "https://stackoverflow.com/questions/8090229/resize-with-averaging-or-rebin-a-numpy-2d-array"
        """

        x, y = self.db.size
        if x % 28 != 0 or y % 28 != 0:
            raise ValueError("Invalid size of the DrawingBoard")

        shape = (28, 28)
        array = self.db.pixels_array

        sh = shape[0], array.shape[0] // shape[0], shape[1], array.shape[1] // shape[1]

        # return array.reshape(sh).mean(-1).mean(1)
        return array.reshape(sh).max(-1).max(1)

    def _reshape_pixels(self) -> ndarray:
        """
        Reshape the pixel array into (784, 1) for using as the input.
        """

        a = self._rescale_pixels()

        return a.reshape((784, 1))

    def _update_indicator(self, indicator_str: str) -> None:
        """
        Update the self.indicator Label with <indicator_str>
        """
        self.indicator.config(text=indicator_str)

    def _update_label(self, prediction_num: int) -> None:
        """
        Update the self.label Label with given <prediction_num>
        """
        msg = f"Neural Network thinks the writing is {prediction_num}"
        self.label.config(text=msg)

    def _prediction_to_string(
        self, prediction: ndarray, initialization: bool = False
    ) -> str:
        """
        Return a string for self.indicator to use.
        """

        out = ""

        if initialization:
            for i in range(10):
                s = f"{i}: 0.0% \n"
                out += s

            return out[:-2]

        else:

            prediction = prediction.reshape((10,))
            total = sum(prediction)

            for i in range(10):
                percentage = prediction[i] / total * 100

                s = f"{i}: {round(percentage, 4)}% \n"
                out += s

            return out[:-2]

    def run(self) -> None:
        """
        Run the mainloop for <App>
        """
        self.root.mainloop()

    @staticmethod
    def show_gray_img(pixels: ndarray) -> None:
        """
        Display a 28x28 gray scale image of <pixels> obtained from the DrawingBoard Canvas.

        The import statement is intentionaly called inside this method to make the GUI
        update the results instantly without closing the plot.
        """

        import matplotlib.pyplot as plt

        p = pixels.reshape((28, 28))

        plt.gray()
        plt.imshow(p)
        plt.show()


if __name__ == "__main__":
    # app = App()
    # app.load_widets()
    # app.setup_nn("npz_files/[784, 30, 20, 10]_ACCURACY88.82.npz")
    # app.run()
    pass
