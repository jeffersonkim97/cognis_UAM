import numpy as np
from base import LieAlgebra, LieGroup, EPS, wrap

"""
so3: 
- use euler angle as element
- if you want the input be in other format, use SO3 class to do transfomation
"""

"""
SO3:
- to_matrix: return DCM
- to_vec: DCM return euler, others don't need to_vec
- exp & log: inputs should be in so3 or SO3 format, log return vector of algebra element, exp return the format of Lie gorup that you call
(e.g., DCM.exp(A) -> DCM, where A = so3(a))
"""

class so3algebra(LieAlgebra): # euler angle body 3-2-1
    def __init__(self, param):
        super().__init__(param)
        assert param.shape == (3,1) or param.shape == (3,)
        self.param = np.reshape(wrap(param), (3,))

    def add(self, other):
        return so3algebra(self.param + other.param)

    def rmul(self, scalar):
        return so3algebra(scalar * self.param)
    
    def neg(self):
        return so3algebra(-self.param)

    @property
    def wedge(self):
        theta1 = self.param[0]
        theta2 = self.param[1]
        theta3 = self.param[2]
        return np.array([
            [0, -theta3, theta2],
            [theta3, 0, -theta1],
            [-theta2, theta1, 0]
        ])
    
    @property
    def ad_matrix(self):
        raise NotImplementedError("")

    @classmethod
    def vee(cls, w):
        theta1 = w[2,1]
        theta2 = w[0,2]
        theta3 = w[1,0]
        return np.array([theta1,theta2,theta3])
    

class SO3DCM(LieGroup): # a SO3 direct cosine matrix (3x3)
    def __init__(self, param):
        super().__init__(param)
        assert param.shape == (3, 3)
        self.param = param

    @staticmethod
    def identity():
        return SO3DCM(np.eye(3))

    @property
    def to_matrix(self):
        return self.param

    @property
    def inv(self):
        return SO3DCM(self.param.T).param

    def product(self, other: "SO3DCM"):
        return self.param @ other.param
    
    @property
    def Ad_matrix(self):
        return self.to_matrix
    
    @classmethod
    def to_vec(cls, X): # getting euler angle
        return SO3Euler.from_dcm(X).param
    
    @classmethod
    def log(cls, G: "SO3DCM") -> "so3algebra":
        R = G.param
        theta = np.arccos((np.trace(R) - 1) / 2)
        A = np.where(np.abs(theta) < EPS, 1 - theta**2/6 + theta**4/120, np.sin(theta)/theta)
        return so3(so3.vee((R - R.T) / (A * 2)))
    
    @classmethod
    def exp(cls, g: "so3algebra") -> "SO3DCM": # so3 matrix -> SO3 matrix (DCM)
        v = g.param
        w = g.wedge
        theta = np.linalg.norm(v)
        A = np.where(np.abs(theta) < EPS, 1 - theta**2/6 + theta**4/120, np.sin(theta)/theta)
        B = np.where(np.abs(theta)<EPS, 0.5 - theta ** 2 / 24 + theta ** 4 / 720, (1 - np.cos(theta)) / theta ** 2)
        return DCM(np.eye(3) + A * w + B * w @ w) # return DCM

    # funcions of getting DCM from other format of angles
    @classmethod
    def from_quat(cls, quat:"SO3Quat")-> "SO3DCM":
        q = quat.param
        assert q.shape == (4, 1) or q.shape == (4,)
        R = np.zeros((3,3))
        a = q[0]
        b = q[1]
        c = q[2]
        d = q[3]
        aa = a * a
        ab = a * b
        ac = a * c
        ad = a * d
        bb = b * b
        bc = b * c
        bd = b * d
        cc = c * c
        cd = c * d
        dd = d * d
        R[0, 0] = aa + bb - cc - dd
        R[0, 1] = 2 * (bc - ad)
        R[0, 2] = 2 * (bd + ac)
        R[1, 0] = 2 * (bc + ad)
        R[1, 1] = aa + cc - bb - dd
        R[1, 2] = 2 * (cd - ab)
        R[2, 0] = 2 * (bd - ac)
        R[2, 1] = 2 * (cd + ab)
        R[2, 2] = aa + dd - bb - cc
        return SO3DCM(R)

    @classmethod
    def from_mrp(cls, mrp: "SO3MRP")-> "SO3DCM": 
        r = mrp.param
        assert r.shape == (4, 1) or r.shape == (4,)
        a = r[:3]
        X = so3(a).wedge
        n_sq = np.dot(a, a)
        X_sq = X @ X
        R = np.eye(3) + (8 * X_sq - 4 * (1 - n_sq) * X) / (1 + n_sq) ** 2
        # return transpose, due to convention difference in book
        return SO3DCM(R.T)

    @classmethod
    def from_euler(cls, euler:"SO3Euler")-> "SO3DCM":
        return cls.from_quat(SO3Quat.from_euler(euler))

class SO3Quat(LieGroup):
    def __init__(self, param):
        super().__init__(param)
        assert param.shape == (4,1) or param.shape == (4,)
        self.param = param

    @staticmethod
    def identity():
        return SO3Quat(np.array([1, 0, 0, 0]))
    
    @property
    def to_matrix(self):
        return SO3DCM.from_quat(self.param)

    @property
    def inv(self):
        return SO3Quat(np.vstack((-self.param[:3], self.param[3])))

    def product(self, other):
        a = self.param
        b = other.param
        r1 = a[0]
        v1 = a[1:]
        r2 = b[0]
        v2 = b[1:]
        res = np.zeros((4,))
        res[0] = r1 * r2 - np.dot(v1, v2)
        res[1:] = r1 * v2 + r2 * v1 + np.cross(v1, v2)
        return res
    
    @property
    def Ad_matrix(self):
        return self.to_matrix

    @classmethod
    def to_vec(cls, X):
        pass
    
    @classmethod
    def log(cls, G: "SO3Quat") -> "so3algebra": # Lie group to Lie algebra
        v = np.zeros((3,))
        q = G.param
        theta = 2 * np.arccos(q[0])
        c = np.sin(theta / 2)
        v[0] = theta * q[1] / c
        v[1] = theta * q[2] / c
        v[2] = theta * q[3] / c
        return so3(np.where(np.abs(c) > EPS, v, np.array([0, 0, 0])))
    
    @classmethod
    def exp(cls, g: "so3algebra") -> "SO3Quat": # exp: so3 element to quat
        q = np.zeros((4,))
        v = g.param
        theta = np.linalg.norm(v)
        q[0] = np.cos(theta / 2)
        c = np.sin(theta / 2)
        n = np.linalg.norm(v)
        q[1] = c * v[0] / n
        q[2] = c * v[1] / n
        q[3] = c * v[2] / n
        return cls(np.where(n > 1e-7, q, np.array([1, 0, 0, 0])))


    # funcions of getting Quat from other format of angles
    @classmethod
    def from_mrp(cls, mrp:"SO3MRP")->"SO3Quat":
        r = mrp.param
        assert r.shape == (4, 1) or r.shape == (4,)
        a = r[:3]
        q = np.zeros((4,))
        n_sq = np.dot(a, a)
        den = 1 + n_sq
        q[0] = (1 - n_sq) / den
        for i in range(3):
            q[i + 1] = 2 * a[i] / den
        return SO3Quat(np.where(r[3], -q, q))

    @classmethod
    def from_dcm(cls, R:"SO3DCM")->"SO3Quat":
        R = R.param
        assert R.shape == (3, 3)
        b1 = 0.5 * np.sqrt(1 + R[0, 0] + R[1, 1] + R[2, 2])
        b2 = 0.5 * np.sqrt(1 + R[0, 0] - R[1, 1] - R[2, 2])
        b3 = 0.5 * np.sqrt(1 - R[0, 0] + R[1, 1] - R[2, 2])
        b4 = 0.5 * np.sqrt(1 - R[0, 0] - R[1, 1] + R[2, 2])

        q1 = np.zeros((4,))
        q1[0] = b1
        q1[1] = (R[2, 1] - R[1, 2]) / (4 * b1)
        q1[2] = (R[0, 2] - R[2, 0]) / (4 * b1)
        q1[3] = (R[1, 0] - R[0, 1]) / (4 * b1)

        q2 = np.zeros((4,))
        q2[0] = (R[2, 1] - R[1, 2]) / (4 * b2)
        q2[1] = b2
        q2[2] = (R[0, 1] + R[1, 0]) / (4 * b2)
        q2[3] = (R[0, 2] + R[2, 0]) / (4 * b2)

        q3 = np.zeros((4,))
        q3[0] = (R[0, 2] - R[2, 0]) / (4 * b3)
        q3[1] = (R[0, 1] + R[1, 0]) / (4 * b3)
        q3[2] = b3
        q3[3] = (R[1, 2] + R[2, 1]) / (4 * b3)

        q4 = np.zeros((4,))
        q4[0] = (R[1, 0] - R[0, 1]) / (4 * b4)
        q4[1] = (R[0, 2] + R[2, 0]) / (4 * b4)
        q4[2] = (R[1, 2] + R[2, 1]) / (4 * b4)
        q4[3] = b4

        q = np.where(
            np.trace(R) > 0,
            q1,
            np.where(
                np.logical_and(R[0, 0] > R[1, 1], R[0, 0] > R[2, 2]),
                q2,
                np.where(R[1, 1] > R[2, 2], q3, q4),
            ),
        )
        return SO3Quat(q)

    @classmethod
    def from_euler(self, euler:"SO3Euler")-> "SO3Quat":
        e = euler.param
        assert e.shape == (3, 1) or e.shape == (3,)
        q = np.zeros((4,))
        cosPhi_2 = np.cos(e[0] / 2)
        cosTheta_2 = np.cos(e[1] / 2)
        cosPsi_2 = np.cos(e[2] / 2)
        sinPhi_2 = np.sin(e[0] / 2)
        sinTheta_2 = np.sin(e[1] / 2)
        sinPsi_2 = np.sin(e[2] / 2)
        q[0] = cosPhi_2 * cosTheta_2 * cosPsi_2 + sinPhi_2 * sinTheta_2 * sinPsi_2
        q[1] = sinPhi_2 * cosTheta_2 * cosPsi_2 - cosPhi_2 * sinTheta_2 * sinPsi_2
        q[2] = cosPhi_2 * sinTheta_2 * cosPsi_2 + sinPhi_2 * cosTheta_2 * sinPsi_2
        q[3] = cosPhi_2 * cosTheta_2 * sinPsi_2 - sinPhi_2 * sinTheta_2 * cosPsi_2
        return SO3Quat(q)

class SO3Euler(LieGroup):
    def __init__(self, param):
        super().__init__(param)
        assert param.shape == (3,1) or param.shape == (3,)
        self.param = param
    
    @staticmethod
    def identity():
        return SO3Euler(np.array([0, 0, 0]))
    
    @property
    def to_matrix(self):
        return SO3DCM.from_euler(self.param).param

    @property
    def inv(self, cls):
        return cls.from_dcm(SO3DCM.inv(SO3DCM.from_euler(self.param))).param


    def product(self, other: "SO3Euler"):
        return SO3Euler.from_dcm(SO3DCM(SO3DCM.from_euler(self) @ SO3DCM.from_euler(other))).param

    @property
    def Ad_matrix(self):
        return self.to_matrix
    
    @classmethod
    def to_vec(cls, X):
        pass

    @classmethod
    def log(cls, G: "SO3Euler") -> "so3algebra":
        return SO3DCM.log(SO3DCM.from_euler(G))
    
    @classmethod
    def exp(cls, g: "so3algebra") -> "SO3Euler":
        return cls.from_dcm(SO3DCM.exp(g))
    
    # funcions of getting Euler from other format of angles
    @classmethod
    def from_quat(cls, quat: "SO3Quat") -> "SO3Euler":
        q = quat.param
        assert q.shape == (4, 1) or q.shape == (4,)
        e = np.zeros((3,))
        a = q[0]
        b = q[1]
        c = q[2]
        d = q[3]
        e[0] = np.arctan2(2 * (a * b + c * d), 1 - 2 * (b**2 + c**2))
        e[1] = np.arcsin(2 * (a * c - d * b))
        e[2] = np.arctan2(2 * (a * d + b * c), 1 - 2 * (c**2 + d**2))
        return Euler(e)

    @classmethod
    def from_dcm(cls, R:"SO3DCM") -> "SO3Euler":
        return cls.from_quat(SO3Quat.from_dcm(R))

    @classmethod
    def from_mrp(cls, a:"SO3MRP") -> "SO3Euler":
        return cls.from_quat(SO3Quat.from_mrp(a))
    
class SO3MRP(LieGroup):
    def __init__(self, param):
        super().__init__(param)
        assert param.shape == (4, 1) or param.shape == (4,)
        self.param = param

    @staticmethod
    def identity():
        return SO3MRP(np.array([0, 0, 0, 0]))
    
    @property
    def to_matrix(self):
        return SO3DCM.from_mrp(self.param).param
    
    @property
    def inv(self):
        return np.block([-self.param[:3], self.param[3]])

    def product(self, other):
        a = self.param[:3]
        b = other.param[:3]
        na_sq = np.dot(a, a)
        nb_sq = np.dot(b, b)
        res = np.zeros((4,))
        den = 1 + na_sq * nb_sq - 2 * np.dot(b, a)
        res[:3] = ((1 - na_sq) * b + (1 - nb_sq) * a - 2 * np.cross(b, a)) / den
        res[3] = 0  # shadow state
        return res
    
    @property
    def Ad_matrix(self):
        return self.to_matrix

    def shadow(self):
        r = self
        assert r.shape == (4, 1) or r.shape == (4,)
        n_sq = np.dot(r[:3], r[:3])
        res = np.zeros((4, 1))
        res[:3] = -r[:3] / n_sq
        res[3] = np.logical_not(r[3])
        return res

    @classmethod
    def shadow_if_necessary(cls, r):
        assert r.shape == (4, 1) or r.shape == (4,)
        return np.where(np.linalg.norm(r[:3]) > 1, cls.shadow(r), r)
    
    @classmethod
    def to_vec(cls, R):
        return cls.from_dcm(R).param
    
    @classmethod
    def log(cls, G: "SO3MRP") -> "so3algebra":
        r = G.param
        n = np.linalg.norm(r[:3])
        return so3algebra(np.where(n > EPS, 4 * np.arctan(n) * r[:3] / n, np.array([0, 0, 0])))

    @classmethod
    def exp(cls, g: "so3algebra") -> "SO3MRP":
        v = g.param
        angle = np.linalg.norm(v)
        res = np.zeros((4,))
        res[:3] = np.tan(angle / 4) * v / angle
        res[3] = 0
        return cls(np.where(angle > EPS, res, np.array([0, 0, 0, 0])))

    @classmethod
    def from_quat(cls, quat:"SO3Quat")-> "SO3MRP":
        q = quat.param
        assert q.shape == (4, 1) or q.shape == (4,)
        x = np.zeros((4,))
        den = 1 + q[0]
        x[0] = q[1] / den
        x[1] = q[2] / den
        x[2] = q[3] / den
        x[3] = 0
        r = cls.shadow_if_necessary(x)
        r[3] = 0
        return SO3MRP(r)

    @classmethod
    def from_dcm(cls, R:"SO3DCM")-> "SO3MRP":
        return cls.from_quat(SO3Quat.from_dcm(R))

    @classmethod
    def from_euler(cls, e:"SO3Euler")-> "SO3MRP":
        return cls.from_quat(SO3Quat.from_euler(e))
    
DCM = SO3DCM
Euler = SO3Euler
Quat = SO3Quat
MRP = SO3MRP
so3 = so3algebra