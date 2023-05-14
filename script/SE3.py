import numpy as np
from base import EPS, LieAlgebra, LieGroup
from SO3 import DCM, Euler, so3

"""
- exp & log: inputs should be in se3 or SE3 format, both return matrix lie group
"""


class se3algebra(LieAlgebra): # param: [x,y,z,theta1,theta2,theta3]
    def __init__(self, param):
        super().__init__(param)
        assert param.shape == (6,1) or param.shape == (6,)
        self.w = so3(param[3:6]).wedge
        self.v = param[0:3]

    def add(self, other):
        return se3algebra(self.param + other.param)

    def rmul(self, scalar):
        return se3algebra(scalar * self.param)
    
    def neg(self):
        return se3algebra(-self.param)

    @property
    def wedge(self):
        return np.block([
            [self.w, self.v.reshape(3,1)],
            [np.zeros((1,4))]
        ])

    @property
    def ad_matrix(self):
        """
        takes 6x1 lie algebra
        input vee operator [x,y,z,theta1,theta2,theta3]
        """
        v = self.v
        vx = np.array([[0, -v[2], v[1]],[v[2], 0, -v[0]],[-v[1],v[0],0]])
        return np.block([[self.w, vx],[np.zeros((3,3)), self.w]])
 
    @classmethod
    def vee(cls, w): # w is 4x4 Lie algebra matrix
        assert w.shape == (4,4)
        x = w[0,3]
        y = w[1,3]
        z = w[2,3]
        theta1 = w[2,1]
        theta2 = w[0,2]
        theta3 = w[1,0]
        return cls(np.array([x,y,z,theta1,theta2,theta3])).param

class SE3group(LieGroup):
    def __init__(self, param):
        super().__init__(param)
        assert param.shape == (6,1) or param.shape == (6,)
        self.param = param
        self.R = DCM.from_euler(Euler(param[3:6])).param
        self.p = param[0:3]

    @staticmethod
    def identity():
        return SE3group(np.array([0,0,0,0,0,0]))
    
    @property
    def to_matrix(self):
        return np.block([
            [self.R, self.p.reshape(3,1)],
            [np.zeros((1,3)), 1]
        ])

    @property
    def inv(self):  # input a matrix of SX form from casadi
        return np.block([[self.R.T, -self.R.T@self.p.reshape(3,1)],
                         [np.zeros((1,3)), 1]])
    
    def product(self, other):
        return np.block([[self.R@other.R, (self.R@self.p+other.p).reshape(3,1)],
                         [np.zeros((1,3)), 1]])

    @property
    def Ad_matrix(self): # Ad matrix of v(6x1) for SE3 Lie Group
        p = self.p
        px = np.array([[0, -p[2], p[1]],[p[2], 0, -p[0]],[-p[1],p[0],0]]) # skew-symmetric
        return np.block([[self.R, px@self.R],
                         [np.zeros((3,3)),self.R]])
    
    @classmethod
    def to_vec(cls, X):
        R = X[0:3, 0:3]
        theta = Euler.from_dcm(DCM(R)).param
        p = X[0:3,3]
        return np.block([p,theta])

    @classmethod
    def log(cls, G: "SE3group") -> "se3algebra": # SE3 matrix to se3 matrix
        X = G.to_matrix
        R = X[0:3, 0:3] # get the SO3 Lie group matrix
        theta = np.arccos((np.trace(R) - 1) / 2)
        wSkew = DCM.log(DCM(R)).wedge
        C1 = np.where(np.abs(theta)<EPS, 1 - theta ** 2 / 6 + theta ** 4 / 120, np.sin(theta)/theta)
        C2 = np.where(np.abs(theta)<EPS, 0.5 - theta ** 2 / 24 + theta ** 4 / 720, (1 - np.cos(theta)) / theta ** 2)
        print((1 / theta**2) * (1 - C1 / (2 * C2)))
        V_inv = (
            np.eye(3)
            - wSkew / 2
            + (1 / theta**2) * (1 - C1 / (2 * C2)) * wSkew @ wSkew
        )

        t = X[0:3,3]
        uInv = V_inv @ t
        return se3algebra(np.block([uInv, so3.vee(wSkew)]))
    
    @classmethod
    def exp(cls, g:"se3algebra") -> "SE3group": # Lie algebra to Lie group # vw is v in wedge form (se3 lie algebra)
        v = g.param # v = [x,y,z,theta1,theta2,theta3]
        v_so3 = v[3:6]  # grab only rotation terms for so3 uses ##corrected to v_so3 = v[3:6]
        X_so3 = so3(v_so3).wedge  # wedge operator for so3
        theta = np.linalg.norm(v[3:6])  # theta term using norm for sqrt(theta1**2+theta2**2+theta3**2)

        # translational components u
        u = np.array([v[0],v[1],v[2]])

        R = DCM.exp(so3(v_so3))  #'Dcm' for direction cosine matrix representation of so3 LieGroup Rotational
        C1 = np.where(np.abs(theta)<EPS, 1 - theta ** 2 / 6 + theta ** 4 / 120, np.sin(theta)/theta)
        C2 = np.where(np.abs(theta)<EPS, 0.5 - theta ** 2 / 24 + theta ** 4 / 720, (1 - np.cos(theta)) / theta ** 2)
        C = np.where(np.abs(theta)<EPS, 1/6 - theta ** 2 /120 + theta ** 4 / 5040, (1 - C1) / theta ** 2)

        V = np.eye(3) + C2 * X_so3 + C * X_so3 @ X_so3

        return SE3group(np.block([V@u, Euler.from_dcm(R).param]))
    
se3 = se3algebra
SE3 = SE3group