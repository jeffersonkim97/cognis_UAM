import numpy as np
from base import LieAlgebra, LieGroup, EPS, wrap

class so2algebra(LieAlgebra): # euler angle body 3-2-1
    def __init__(self, param):
        super().__init__(param)
        assert param.shape == (1,1) or param.shape == (1,)
        self.param = np.reshape(wrap(param), (1,))

    def add(self, other):
        return so2algebra(self.param + other.param)

    def rmul(self, scalar):
        return so2algebra(scalar * self.param)
    
    def neg(self):
        return so2algebra(-self.param)

    @property
    def wedge(self):
        theta = self.param[0]
        return np.array([
            [0, -theta],
            [theta, 0]
        ])
    
    @property
    def ad_matrix(self):
        raise NotImplementedError("")

    @classmethod
    def vee(cls, w):
        theta = w[1,0]
        return np.array([theta])
    

class SO2group(LieGroup): # input: theta, output: cosine matrix 2x2
    def __init__(self, param):
        super().__init__(param)
        assert self.param.shape == (1, 1) or self.param.shape == (1, )
        self.param = np.reshape(wrap(param), (1,))

    @staticmethod
    def identity():
        return SO2group(np.array([0]))

    @property
    def to_matrix(self):
        theta = self.param[0]
        return np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ])

    @property
    def inv(self):
        return self(-self.param).to_matrix

    def product(self, other: "SO2group"):
        theta = self.param + other.param
        return np.array([theta])
    
    @property
    def Ad_matrix(self):
        return self.to_matrix
    
    @classmethod
    def to_vec(cls, X):
        theta = np.arctan2(X[1, 0], X[0, 0])
        return np.array([theta])
    
    @classmethod
    def log(cls, G: "SO2group") -> "so2algebra":
        return so2algebra(G.param)
    
    @classmethod
    def exp(cls, g: "so2algebra") -> "SO2group": # so2 -> SO2 matrix
        return SO2group(g.param)# return SO2 matrix


so2 = so2algebra
SO2 = SO2group    
