from tasks.Task import Task
import utils.grid_utils as grid_utils
import numpy as np
import random

class BasicCountingV3(Task):

    def __init__(self, grid_dim_min=3, grid_dim_max=30, num_px_min=1, num_px_max=10):
        super(BasicCountingV3, self).__init__(grid_dim_min, grid_dim_max)
        self.num_px_min = num_px_min
        self.num_px_max = num_px_max

    def generateInputs(self, k):
        input_grids = []
        for _ in range(k):
            num_px = np.random.choice(np.arange(self.num_px_min, self.num_px_max))
            tmp_grid = self._generateInput(num_px)
            input_grids.append(tmp_grid)

        random.shuffle(input_grids)
        return input_grids

    def _generateInput(self, mpt):
        return grid_utils.generateRandomPixels(max_pixels_total=mpt,
                                               max_pixels_per_color=self.num_px_max,
                                               grid_dim_min=self.grid_dim_min,
                                               grid_dim_max=self.grid_dim_max,
                                               sparsity=0.8)

    def _generateOutput(self, input_grid):
        pixel_count = grid_utils.perColorPixelCount(input_grid)
        grid_px_count = np.reshape(pixel_count, [3, 3])
        return grid_px_count

class BasicCountingV4(Task):

    def __init__(self, grid_dim_min=3, grid_dim_max=30, num_px_min=1, num_px_max=10):
        super(BasicCountingV4, self).__init__(grid_dim_min, grid_dim_max)
        self.num_px_min = num_px_min
        self.num_px_max = num_px_max

    def generateInputs(self, k):
        input_grids = []
        for _ in range(k):
            num_px = np.random.choice(np.arange(self.num_px_min, self.num_px_max))
            tmp_grid = self._generateInput(num_px)
            input_grids.append(tmp_grid)

        random.shuffle(input_grids)
        return input_grids

    def _generateInput(self, mpt):
        return grid_utils.generateRandomPixels(max_pixels_total=mpt,
                                               max_pixels_per_color=self.num_px_max,
                                               grid_dim_min=self.grid_dim_min,
                                               grid_dim_max=self.grid_dim_max,
                                               sparsity=0.8)

    def _generateOutput(self, input_grid):
        pixel_count = grid_utils.perColorPixelCountV2(input_grid)
        grid_px_count = np.zeros(16)
        grid_px_count[:10] = pixel_count
        grid_px_count = np.reshape(grid_px_count, [4, 4])
        return grid_px_count
