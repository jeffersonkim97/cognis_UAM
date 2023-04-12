import numpy as np

# SE2
# Reference: Ethan Eade
class SE2:
    @staticmethod
    def rotmat(theta):
        return np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ])

    def __init__(self, x, y, theta):
        self.x = x
        self.y = y
        self.theta = theta

    def R(self):
        return self.rotmat(self.theta)
    
    def w(self):
        return np.array([self.theta])
    
    def wedge(self):
        return np.array([
            [0, -self.w()],
            [self.w(), 0]
        ])
    
    def p(self):
        return np.array([self.x, self.y])

    def __mul__(self, other):
        R = self.R() @ other.R()
        t = self.R() @ other.p() + self.p()

        theta = self.w() + other.w()

        return SE2(x=t[0], y=t[1], theta=theta)
    
    def SE2param(self):
        return [self.x, self.y, self.theta]
    
    def matrix(self):
        R = self.R()
        p = self.p()
        return np.array([
            [R[0,0], R[0,1], p[0]],
            [R[1,0], R[1,1], p[1]],
            [0, 0, 1]
        ])
    
    def inv(self):
        RT = self.R().T
        invU = -self.R().T@self.p()
        return np.array([
            [RT[0,0], RT[0,1], invU[0]],
            [RT[1,0], RT[1,1], invU[1]],
            [0, 0, 1]
        ])
    
    def log(self):
        R = self.R()
        theta = np.arctan2(R[2,1], R[1,1])

        if np.isclose(theta,0):
            A = 1 - theta**2/6*(1 - theta**2/20*(1 - theta**2/42))
            B = 1/2*(1 - theta**2/12*(1 - theta**2/30*(1 - theta**2/56)))
        else:
            A = np.sin(theta)/theta
            B = (1-np.cos(theta))/theta
        invV = 1/(A**2 + B**2)*np.array([[A, B],[-B, A]])

        return np.block([[invV@self.p()],[theta]])
    
    def __repr__(self):
        return repr(self.matrix())

class se2:
    @staticmethod
    def rotmat(theta):
        return np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ])

    def __init__(self, x, y, theta):
        self.x = x
        self.y = y
        self.theta = theta

    def R(self):
        return self.rotmat(self.theta)
    
    def p(self):
        return np.array([self.x, self.y])
    
    def w(self):
        return np.array([self.theta])

    def wedge(self):
        return np.array([
            [0, -self.theta],
            [self.theta, 0]
        ])
    
    def vee(self):
        return self.w()
    
    def exp(self):
        th = self.theta
        sx = np.sin(th)
        cx = np.cos(th)
        R = self.R()
        if np.isclose(th,0):
            A = 1 - th**2/6*(1 - th**2/20*(1 - th**2/42))
            B = 1/2*(1 - th**2/12*(1 - th**2/30*(1 - th**2/56)))
            V = np.array([
                [A, -B],
                [B, A]
            ])
        else:
            V = 1/th*np.array([[sx, -(1-cx)],[1-cx, sx]])

        u = V@self.p()
        
        return np.array([
            [R[0,0], R[0,1], u[0]],
            [R[1,0], R[1,1], u[1]],
            [0, 0, 1]
        ])