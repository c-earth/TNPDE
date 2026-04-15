import numpy as np
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
        self.neighbors = np.array([list(range(1, len(self.points) + 1)), list(range(-1, len(self.points) - 1))])
        self.neighbors[-1, 0] = -1

    def find_simplex(self, x):
        for i, idx in enumerate(self.idxs):
            if x <= self.points[idx]:
                return i - 1
        return -1    


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
        return self.nodes.n
    

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
    