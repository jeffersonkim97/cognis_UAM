import numpy as np
from base import EPS, LieAlgebra, LieGroup
from SO2 import so2, SO2, wrap
    
class se2algebra(LieAlgebra):
    def __init__(self, param):
        super().__init__(param)
        assert param.shape == (3,1) or param.shape == (3,)
        self.theta = np.reshape(wrap(param[2]), (1,))
        self.w = so2(self.theta).wedge
        self.v = param[0:2]
        self.param = np.block([self.v, self.theta[0]])
    
    def add(self, other):
        return se2algebra(self.param + other.param)

    def rmul(self, scalar):
        return se2algebra(scalar * self.param)
    
    def neg(self):
        return se2algebra(-self.param)

    @property
    def wedge(self):
        return np.block([
            [self.w, self.v.reshape(2,1)],
            [0, 0, 0]
        ])
    
    @property
    def ad_matrix(self):
        x, y, theta = self.v[0], self.v[1], self.theta[0]
        return np.array([
            [0, -theta, y],
            [theta, 0, -x],
            [0, 0, 0]
        ])
    
    @classmethod
    def vee(cls, w):
        assert w.shape == (3,3)
        x = w[0,2]
        y = w[1,2]
        theta = w[1,0]
        return cls(np.array([x,y,theta])).param


class SE2group(LieGroup):
    def __init__(self, param):
        super().__init__(param)
        assert param.shape == (3,1) or param.shape == (3,)
        self.theta = np.reshape(wrap(param[2]), (1,))
        self.R = SO2(self.theta).to_matrix
        self.p = param[0:2]
        self.param = np.block([self.p, self.theta[0]])
    
    @staticmethod
    def identity():
        return SE2group(np.array([0,0,0]))
    
    @property
    def to_matrix(self):
        return np.block([
            [self.R, self.p.reshape(2,1)],
            [np.zeros((1,2)), 1]
        ])
    
    @property
    def inv(self):
        return np.block([[self.R.T, -self.R.T@self.p.reshape(2,1)],
                         [np.zeros((1,2)), 1]])
    
    def product(self, other):
        return np.block([[self.R@other.R, (self.R@self.p+other.p).reshape(2,1)],
                         [np.zeros((1,2)), 1]])

    @property
    def Ad_matrix(self):
        v = np.array([self.p[1], -self.p[0]])
        return np.block([[self.R, v.reshape(2,1)],
                         [np.zeros((1,2)), 1]])
    
    @classmethod
    def to_vec(cls, X):
        R = X[0:2,0:2]
        theta = SO2.to_vec(R)
        p = X[0:2,2]
        return np.block([p,theta])

    @classmethod
    def log(cls, G: "SE2group") -> se2algebra:
        v = G.p
        theta = G.param[2]
        with np.errstate(divide='ignore',invalid='ignore'):
            a = np.where(np.abs(theta) < 1e-3, 1 - theta**2/6 + theta**4/120, np.sin(theta)/theta)
            b = np.where(np.abs(theta) < 1e-3, theta/2 - theta**3/24 + theta**5/720, (1 - np.cos(theta))/theta)
        V_inv = np.array([
            [a, b],
            [-b, a]
        ])/(a**2 + b**2)
        p = V_inv@v
        return se2algebra(np.block([p, theta]))
    
    @classmethod
    def exp(cls, g: "se2algebra") -> "SE2group":

        theta = g.theta[0]
        with np.errstate(divide='ignore',invalid='ignore'):
            a = np.where(np.abs(theta) < 1e-3, 1 - theta**2/6 + theta**4/120, np.sin(theta)/theta)
            b = np.where(np.abs(theta) < 1e-3, theta/2 - theta**3/24 + theta**5/720, (1 - np.cos(theta))/theta)
        V = np.array([[a, -b], [b, a]])
        p = V@(g.v)
        return SE2group(np.block([p,theta]))
    
se2 = se2algebra
SE2 = SE2group

# def diff_correction(e: se2, n=100):
#     # computes (1 - exp(-ad_x)/ad_x = sum k=0^infty (-1)^k/(k+1)! (ad_x)^k
#     ad = e.ad_matrix
#     ad_i = np.eye(3)
#     s = np.zeros((3, 3))
#     for k in range(n):
#         s += ((-1)**k/math.factorial(k+1))*ad_i
#         ad_i = ad_i @ ad
#     return -np.linalg.inv(s)@((-e).exp.Ad_matrix)

# def se2_diff_correction(e: se2): # U
#     x = e.x
#     y = e.y
#     theta = e.theta
#     with np.errstate(divide='ignore',invalid='ignore'):
#         a = np.where(abs(theta) > 1e-3, -theta*np.sin(theta)/(2*(np.cos(theta) - 1)), 1 - theta**2/12 - theta**4/720)
#         b = np.where(abs(theta) > 1e-3, -(theta*x*np.sin(theta) + (1 - np.cos(theta))*(theta*y - 2*x))/(2*theta*(1 - np.cos(theta))), -y/2 + theta*x/12 - theta**3*x/720)
#         c = np.where(abs(theta) > 1e-3, -(theta*y*np.sin(theta) + (1 - np.cos(theta))*(-theta*x - 2*y))/(2*theta*(1 - np.cos(theta))), x/2 + theta*y/12 + theta**3*y/720)
#     return -np.array([
#         [a, theta/2, b],
#         [-theta/2, a, c],
#         [0, 0, 1]
#     ])

# def se2_diff_correction_inv(e: se2): # U_inv
#     x = e.x
#     y = e.y
#     theta = e.theta
#     with np.errstate(divide='ignore',invalid='ignore'):
#         a = np.where(abs(theta) > 1e-3, np.sin(theta)/theta, 1 - theta**2/6 + theta**4/120)
#         b = np.where(abs(theta) > 1e-3, (1  - np.cos(theta))/theta, theta/2 - theta**3/24)
#         c = np.where(abs(theta) > 1e-3, -(x*(theta*np.cos(theta) - theta + np.sin(theta) - np.sin(2*theta)/2) + y*(2*np.cos(theta) - np.cos(2*theta)/2 - 3/2))/(theta**2*(1 - np.cos(theta))), y/2 + theta*x/6 - theta**2*y/24 - theta**3*x/120 + theta**4*y/720)
#         d = np.where(abs(theta) > 1e-3, -(x*(-2*np.cos(theta) + np.cos(2*theta)/2 + 3/2) + y*(theta*np.cos(theta) - theta + np.sin(theta) - np.sin(2*theta)/2))/(theta**2*(1 - np.cos(theta))), -x/2 + theta*y/6 + theta**2*x/24 - theta**3*y/120 - theta**4*x/720)
#     return -np.array([
#         [a, -b, c],
#         [b, a, d],
#         [0, 0, 1]
#     ])