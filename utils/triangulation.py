import numpy as np
import matplotlib.pyplot as plt

from scipy.spatial import Delaunay


class Nodes():
    '''
    Triangulation Node
    '''

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
    '''
    Delaunay for unique points in 1D
    '''

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
    '''
    Extend Scipy`s Delaunay to handle 1D
    '''
    
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
    
    @property
    def reversed_neighbor_idxs(self):
        reversed_neighbor_idxs = - np.ones(self.neighbors.shape)
        for element, element_neighbors in enumerate(self.neighbors):
            for i, element_neighbor in enumerate(element_neighbors):
                if element_neighbor != -1:
                    found = False
                    for reversed_neighbor_idx, reversed_neighbor in enumerate(self.neighbors[element_neighbor]):
                        if element == reversed_neighbor:
                            found = True
                            reversed_neighbor_idxs[element, i] = reversed_neighbor_idx
                            break
                    if not found:
                        raise ValueError(f'not matching neighbor for element {element}')

        return reversed_neighbor_idxs
    

    def find_simplex(self, x):
        return self.delaunay.find_simplex(x)
    
    def visualize(self):
        if self.d == 1:
            plt.figure()
            plt.plot(self.points, np.zeros(self.points.shape))
            plt.plot(self.points, np.zeros(self.points.shape), 'o')
            plt.show()
        elif self.d == 2:
            plt.figure()
            plt.triplot(self.points[:, 0], self.points[:, 1], self.simplices)
            plt.plot(self.points[:, 0], self.points[:, 1], 'o')
            plt.show()
        else:
            raise NotImplementedError()