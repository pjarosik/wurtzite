import wurtzite as wzt
import numpy as np
import matplotlib.pyplot as plt
from utils_1st import displace_love2
import scipy.integrate
import dataclasses
import copy
from scipy.integrate import quad_vec
import sys

import cupy as cp
from constants import *



@dataclasses.dataclass
class DislocationsState:
    d1: wzt.model.DislocationDef
    d2: wzt.model.DislocationDef
    d1_rt: cp.ndarray
    d2_rt: cp.ndarray

    @property
    def tuple(self):
        return self.d1, self.d2, self.d1_rt, self.d2_rt

    def copy(self, **kwargs):
        return DislocationsState(**{**self.__dict__, **kwargs})


