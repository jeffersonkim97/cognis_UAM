"""
Abstract base classes for Lie Groups and Lie Algebras
Note, these are not restricted to Matrix Lie Groups.
"""

import abc

import numpy as np


EPS = 1e-7

def wrap(x):
    return np.where(np.abs(x) >= np.pi, (x + np.pi) % (2 * np.pi) - np.pi, x)

class LieAlgebra:
    def __init__(self, param): # param should be np.array
        self.param = param

    def __add__(self, other):
        return self.add(other)

    def __sub__(self, other):
        return self.add(other.neg())

    def __rmul__(self, other): # scalar mul
        return self.rmul(other)

    def __neg__(self):
        return self.neg()
    
    def __eq__(self, other) -> bool:
        return np.linalg.norm(self.param - other.param) < EPS
    
    def __repr__(self):
        return repr(self.param)

    def __str__(self):
        return str(self.param)

    @abc.abstractmethod
    def add(self, other):
        """
        Add to elements of the Lie algebra
        """

    @abc.abstractmethod
    def neg(self):
        """
        Negative of Lie algebra
        """

    @abc.abstractmethod
    def rmul(self, other):
        """
        Add to elements of the Lie algebra
        """

    @property
    @abc.abstractmethod
    def wedge(self):
        """
        Map a vector to a Lie algebra matrix.
        """

    @property
    @abc.abstractmethod
    def ad_matrix(self):
        """
        Returns ad matrix of the element vector.
        """

    @classmethod
    @abc.abstractmethod
    def vee(cls, other):
        """
        Map a Lie algebra matrix to a avector
        """


class LieGroup(abc.ABC):
    """
    A Lie Group with group operator (*) is:
    (C)losed under operator (*)
    (A)ssociative with operator (*), (G1*G2)*G3 = G1*(G2*G3)
    (I)nverse: has an inverse such that G*G^-1 = e
    (N)uetral: has a neutral element: G*e = G
    Abstract base class, must implement:
    exp, identity, inv, log, product
    """

    def __init__(self, param):
        self.param = param

    def __mul__(self, other):
        """
        The * operator will be used as the Group multiplication operator
        (see product)
        """
        if not isinstance(other, type(self)):
            return TypeError("Lie Group types must match for product")
        # assert isinstance(type(other), LieGroup)
        return self.product(other)
    
    def __matmul__(self, other):
        """
        The * operator will be used as the Group multiplication operator
        (see product)
        """
        if not isinstance(other, type(self)):
            return TypeError("Lie Group types must match for product")
        # assert isinstance(type(other), LieGroup)
        return self.product(other)
    

    def __repr__(self):
        return repr(self.param)

    def __str__(self):
        return str(self.param)

    def __eq__(self, other) -> bool:
        return np.linalg.norm(self.param - other.param) < EPS

    @staticmethod
    @abc.abstractmethod
    def identity():
        """
        The identity element of the gorup, e
        """

    @property
    @abc.abstractmethod
    def to_matrix(self):
        """
        Returns the matrix lie group representation
        """    
    
    @property
    @abc.abstractmethod
    def inv(self):
        """
        The inverse operator G1*G1.inv() = e
        """ 
    
    @abc.abstractmethod
    def product(self, other):
        """
        The group operator (*), returns an element of the group: G1*G2 = G3
        """
    
    @property
    @abc.abstractmethod
    def Ad_matrix(self):
        """
        Ad matrix of the element
        """

    @classmethod
    @abc.abstractmethod
    def to_vec(cls, other):
        """
        Returns the element vector of the matrix lie group
        """
    
    @classmethod
    @abc.abstractmethod
    def log(cls, other):
        """
        Returns the Lie logarithm of a group element, an element of the
        Lie algebra
        """

    @classmethod
    @abc.abstractmethod
    def exp(g: LieAlgebra):
        """
        Compute the Lie group exponential of a Lie algebra element
        """   