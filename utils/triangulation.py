import numpy as np
import matplotlib.pyplot as plt

from scipy.spatial import Delaunay


class Nodes():
    def __init__(self, points):
        self.points, self.idxs_inverse = self.process_points(points)

    @property
    def original_points(self):
        return np.take(self.points, self.idxs_inverse, axis = 0)
    
    @property
    def n(self):
        return self.points.shape[0]
    
    @property
    def d(self):
        return self.points.shape[1]        
    
    @staticmethod
    def process_points(points):
        points = np.array(points)

        if len(points.shape) == 1:
            d = 1
        elif len(points.shape) == 2:
            d = points.shape[1]
        else:
            raise ValueError('`points` need to be a sequence of coordinates, i.e., 2D array.')
        
        points = points.reshape((-1, d))
        points, idxs = np.unique(points, return_inverse = True, axis = 0)

        return points, idxs
    

class Delaunay1D():
    def __init__(self, points):
        self.points = points
        self.idxs = np.argsort(self.points.reshape(-1))

        self.simplices = np.stack([self.idxs[:-1], self.idxs[1:]], axis = 1)
        self.neighbors = np.array([list(range(1, len(self.simplices) + 1)), list(range(-1, len(self.simplices) - 1))]).T
        self.neighbors[-1, 0] = -1

    def find_simplex(self, x):
        filter = (x[... , None, None] <= self.points[self.idxs[1:]]) * (x[..., None, None] >= self.points[self.idxs[:-1]])
        grid = np.broadcast_to(np.array(range(len(self.simplices))), x.shape + (len(self.simplices),))[..., None]
        return grid[filter]


class Triangulation():
    def __init__(self, points):
        super().__init__()

        self.nodes = Nodes(points)

        assert self.nodes.d != 0 and self.nodes.n != 0, 'cannot triangulate empty set of points'
        assert self.nodes.n >= self.nodes.d + 1, 'dimension can be reduced'
        
        if self.nodes.d == 1:
            self.delaunay = Delaunay1D(self.nodes.points)
        else:
            self.delaunay = Delaunay(self.nodes.points, incremental = False)


    @property
    def n(self):
        return len(self.simplices)
    

    @property
    def d(self):
        return self.nodes.d


    @property
    def points(self):
        return self.nodes.points
    

    @property
    def simplices(self):
        return self.delaunay.simplices
    

    @property
    def neighbors(self):
        return self.delaunay.neighbors
    

    def find_simplex(self, x):
        return self.delaunay.find_simplex(x)
    
    def visualize(self, highlight_point = None, highlight_element = None):
        if self.d == 1:
            plt.figure()
            plt.plot(self.points, np.zeros(self.points.shape))
            plt.plot(self.points, np.zeros(self.points.shape), 'o')
            if highlight_element is not None and highlight_element != -1:
                xs = highlight_point[:, 0]
                plt.scatter(xs, np.zeros(xs.shape), marker = 'o', color = 'r', s = 50)
                plt.scatter(self.points[self.simplices[highlight_element]], np.zeros(self.points[self.simplices[highlight_element]].shape), marker = 's', s = 100, c = 'm')
            plt.show()
        elif self.d == 2:
            plt.figure()
            plt.triplot(self.points[:, 0], self.points[:, 1], self.simplices)
            plt.plot(self.points[:, 0], self.points[:, 1], 'o')
            if highlight_element is not None and highlight_element != -1:
                plt.scatter(highlight_point[:, 0], highlight_point[:, 1], marker = 'o', color = 'r', s = 50)
                plt.scatter(self.points[self.simplices[highlight_element]][:, 0], self.points[self.simplices[highlight_element]][:, 1], marker = 's', s = 100, c = 'm')
            plt.show()
        else:
            raise NotImplementedError()
