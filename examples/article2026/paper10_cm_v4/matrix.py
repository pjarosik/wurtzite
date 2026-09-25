import sympy
from sympy.matrices import Matrix
from sympy import *
b = symbols('b')
#print(Matrix([[1,b,0],[0,1,0],[0,0,1]]).inv())
print(Matrix([[1,b,0],[0,1,0],[0,0,1]])
     *Matrix([[1, -b, 0], [0, 1, 0], [0, 0, 1]]))
