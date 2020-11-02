import numpy as np
import cv2


class Grid:
    """
    Responsible for creating a grid with dimensions height X width.
    The grid is superimposed on the video image, writing down through
    the entered() and exited() methods when a centroid is discovered,
    or lost, during detection and tracking.
    --
    This class is not intended to be subclassed.
    """

    def __init__(self, video_dim=(600, 800), height=8, width=8):
        """
        Args:
            video_dim (:tuple:'int'): video dimensions.
            height (int): The grid height in cells.
            width (int): The grid width in cells.
        """
        # Saving the video dimensions for future cell calculations
        self.__video_dim = video_dim
        # Saving height and width of grid
        self.__height = height
        self.__width = width
        # Creating two grids, one for exits and another for ins
        self.__in_grid = self.__create_grid(self.__height, self.__width)
        self.__out_grid = self.__create_grid(self.__height, self.__width)
        # Initialize a mask to use an AND operation to superimpose the
        # grid on the image, it's only instantiated once a first call
        # to superimpose_grid() is made
        self.__grid_mask = None
        self.total_in = 0
        self.total_out = 0

    def entered(self, centroid_coordinates=(None, None)):
        """
        Annotates in which cell a centroid appears in the image.
        Raises: TypeError, ValueError.
        """
        print("called entered for centroid {}".format(centroid_coordinates))
        self.__check_type(centroid_coordinates)
        # Calculates cell
        row, column = self.__calculate_cell(centroid_coordinates)
        # Registers a new appearance
        self.__in_grid[row, column] += 1
        self.total_in += 1

    def exited(self, centroid_coordinates=(None, None)):
        """
        Annotates in which cell a centroid disappears in the image.
        Raises: TypeError, ValueError.
        """
        print("called exited for centroid {}".format(centroid_coordinates))
        self.__check_type(centroid_coordinates)
        # Calculates cell
        row, column = self.__calculate_cell(centroid_coordinates)
        # Registers where the centroid disappeared
        self.__out_grid[row, column] += 1

    def __calculate_cell(self, coordinates=(None, None)):
        """
        Calculates to which cell the coordinates belong to

        Parameters
        ----------
        coordinates: tuple of ints
            The coordinates of the centroid to be mapped to a cell

        Returns
        -------
        out : int
            cell index
        """
        row = (coordinates[1] * (self.__height - 1)) // self.__video_dim[0]
        column = (coordinates[0] * (self.__width - 1)) // self.__video_dim[1]
        return row, column

    def print_grid_as_array(self):
        print("in grid:")
        print(self.__in_grid)
        print("out grid:")
        print(self.__out_grid)

    def superimpose_grid(self, frame):
        """
        Draws the grid on the frame and returns it back to be shown

        :param frame: frame to superimpose the grids data on
        :return: frame with grid printed on it
        """
        # If the mask hasn't been generated yet, do it
        if self.__grid_mask is not None:
            self.generate_mask()

        # frame = cv2.bitwise_and()
        # TODO

        return frame

    def generate_mask(self):
        # Create a black image with dimensions same as video image
        mask = np.zeros((self.__video_dim[0], self.__video_dim[1]), dtype=int)
        # Draw the grid lines on it
        # TODO
        # Save it to be used when needed
        self.__grid_mask = mask

    @staticmethod
    def __create_grid(height, width):
        return np.zeros((height, width), dtype=int)

    @staticmethod
    def __check_type(coordinates):
        """
        Checks if the coordinates received are valid.
        Raises: TypeError, ValueError.
        """
        if coordinates.any() is None:
            raise TypeError("Coordinate tuple is empty")
        elif not all(isinstance(coord, (int, np.integer)) for coord in coordinates):
            raise ValueError("Coordinates must be integer type")
