"""
Class to implement a drawing board on a GUI using tk.Canvas. 

The original work of drawing with mouse on Tkinter Canvas is from 
https://stackoverflow.com/questions/70403360/drawing-with-mouse-on-tkinter-canvas

and the code was modified and extended for this project. 
"""

import tkinter as tk
from typing import List, Tuple
from numpy import ndarray, zeros


class DrawingBoard:

    line_id: bool
    line_points: List[int]
    line_pixels: List[Tuple[int, int]]
    size: Tuple[int]
    canvas: tk.Canvas

    def __init__(self, width: int = 224, height: int = 224) -> None:

        self.line_id = None
        self.line_points = []
        self.line_pixels = []

        self.size = (width, height)
        self.canvas = tk.Canvas(width=self.size[0], height=self.size[1], bg="white")

        self.canvas.bind("<Button-1>", self.set_start)
        self.canvas.bind("<B1-Motion>", self.draw_line)
        self.canvas.bind("<ButtonRelease-1>", self.end_line)

        # Used for testing and debugging
        # self.canvas.bind("<Button-2>", self.__print_pixels)

    def set_start(self, event: tk.Event) -> None:

        self.line_points.extend((event.x, event.y))

    def draw_line(self, event: tk.Event) -> None:

        ex, ey = event.x, event.y

        if 0 <= ex < self.size[0] and 0 <= ey < self.size[1]:

            self.line_points.extend((event.x, event.y))
            self.line_pixels.append((event.x, event.y))

            if self.line_id is not None:
                self.canvas.delete(self.line_id)
            self.line_id = self.canvas.create_line(self.line_points, fill="black")

    def end_line(self, event: tk.Event) -> None:

        self.line_points.clear()
        self.line_id = None

    def reset_canvas(self) -> None:
        self.canvas.delete("all")
        self.line_id = None
        self.line_points.clear()
        self.line_pixels.clear()

    def __print_pixels(self, event: tk.Event) -> None:
        """
        Used for testing and debugging the code.
        """
        print(self.line_pixels)
        print(self.pixels_array)

    @property
    def pixels_array(self) -> ndarray:
        """
        Return a NumPy array representing the pixels on the Canvas.
        Points where the mouse moves through are saved as having the pixel value of 255
        and its neighbouring pixels with lesser value for implementing the blurriness of
        actual handwritings.
        """

        array = zeros(self.size)

        for coord in self.line_pixels:
            x, y = coord
            array[y][x] = 255.0
            DrawingBoard._blur_neighbor(array, coord)

        return array

    @staticmethod
    def _blur_neighbor(a: ndarray, coord: Tuple[int, int], radius: int = 5):
        """
        'blur' the neibouring pixels in array <a> within the <radius> from the <coord>.
        Note that this is method is not optimized and needs improvements in both
        efficiency and effectiveness.
        """
        size = a.shape
        x, y = coord

        for r in range(1, radius + 1):

            side_darkness = 255 * (0.7**r)
            diag_darkness = 255 * (0.5**r)

            if x - r > 0 and x + r < size[0]:

                if a[y][x - r] < side_darkness:
                    a[y][x - r] = side_darkness
                if a[y][x + r] < side_darkness:
                    a[y][x + r] = side_darkness

            if y - r > 0 and y + r < size[1]:

                if a[y - r][x] < side_darkness:
                    a[y - r][x] = side_darkness

                if a[y + r][x] < side_darkness:
                    a[y + r][x] = side_darkness

            if y - r > 0 and x + r < size[0]:
                ne = a[y - r][x + r]
                if ne < diag_darkness:
                    ne = diag_darkness

            if y - r > 0 and x - r > 0:
                nw = a[y - r][x - r]
                if nw < diag_darkness:
                    nw = diag_darkness

            if y + r < size[0] and x - r > 0:
                sw = a[y + r][x - r]
                if sw < diag_darkness:
                    sw = diag_darkness

            if y + r < size[0] and x + r < size[0]:
                se = a[y + r][x + r]
                if se < diag_darkness:
                    se = diag_darkness


if __name__ == "__main__":
    pass
