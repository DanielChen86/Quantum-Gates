import numpy as np
import sympy as sym

total_level=3



def show(obj):
    if isinstance(obj,(Operator,Repre)):
        return sym.simplify(sym.Matrix(obj.Matrix))

'---------------------- number-basis converters ----------------------'

def order2(num):
    return int(np.log2(num))+1

def ntob(number,level=total_level):
    returning=list(int(elements) for elements in ntos(number)[::-1])
    return tuple([0]*(level-len(returning))+returning)


def ntos(number):
    return bin(number)[2:][::-1]

def bton(basis):
    result=0
    for i in range(len(basis)):
        result+=basis[i]*2**(len(basis)-i-1)
    return result




'---------------------- Operator ----------------------'


def allbasis(level=total_level):
    result=[]
    for i in range(2**level):
        result.append(ntob(i,level))
    return tuple(result)




class Operator():
    def __init__(self,func,on_which=0,controllers=[],angle=0.,MatrixRepre=sym.Matrix.eye(2**total_level),level=total_level):
        self.func=func
        self.level=level
        self.on_which=on_which
        self.controllers=controllers
        self.angle=angle
        self.MatrixRepre=MatrixRepre
        self.dic={'on_which':on_which,'controllers':controllers,'angle':angle,'MatrixRepre':MatrixRepre}
        self.Matrix=sym.zeros(2**level,2**level)
        for basis_i in allbasis(self.level):
            temp_state_dic=self.func(basis_i,**self.dic)
            for basis_f,value_f in temp_state_dic.items():
                self.Matrix[bton(basis_f),bton(basis_i)]=value_f

    def __repr__(self):
        return np.matrix(self.Matrix).__repr__()
    def __str__(self):
        return self.__repr__()

    
    def adjoint(self):
        return Repre(self.Matrix.adjoint())


    def __eq__(self,other):
        if isinstance(other,Operator):
            return self.Matrix==other.Matrix
        elif isinstance(other,Repre):
            return self.Matrix==other.Matrix

    def __mul__(self,other):
        if isinstance(other,Operator):
            return Repre(self.Matrix*other.Matrix)
        elif isinstance(other,Repre):
            return Repre(self.Matrix*other.Matrix)




class Repre():
    def __init__(self,Matrix):
        self.Matrix=Matrix
    
    def __repr__(self):
        return np.matrix(self.Matrix).__repr__()
    def __str__(self):
        return self.__repr__()
    
    def adjoint(self):
        return Repre(self.Matrix.adjoint())

        
    def __eq__(self,other):
        if isinstance(other,Operator):
            return self.Matrix==other.Matrix
        elif isinstance(other,Repre):
            return self.Matrix==other.Matrix

    def __mul__(self,other):
        if isinstance(other,Operator):
            return Repre(self.Matrix*other.Matrix)
        elif isinstance(other,Repre):
            return Repre(self.Matrix*other.Matrix)



'---------------------- single-qubit gates ----------------------'



def identity(basis,**kwargs):
    return {basis:1}



def NOT_single(basis,**kwargs):
    on_which=kwargs['on_which']
    basis=list(basis)
    if basis[on_which]==0:
        basis[on_which]=1
        return {tuple(basis):1}
    elif basis[on_which]==1:
        basis[on_which]=0
        return {tuple(basis):1}
    else:
        print ("error")
        return None



def Hadamard_single(basis,**kwargs):
    on_which=kwargs['on_which']
    basis=list(basis)
    if basis[on_which]==0:
        basis0=basis.copy()
        basis1=basis.copy()
        basis1[on_which]=1
        return {tuple(basis0):1/sym.sqrt(2),tuple(basis1):1/sym.sqrt(2)}
    elif basis[on_which]==1:
        basis0=basis.copy()
        basis1=basis.copy()
        basis0[on_which]=0
        return {tuple(basis0):1/sym.sqrt(2),tuple(basis1):-1/sym.sqrt(2)}
    else:
        print ("error")
        return None


def sigmaX_single(basis,**kwargs):
    on_which=kwargs['on_which']
    basis=list(basis)
    if basis[on_which]==0:
        basis[on_which]=1
        return {tuple(basis):1}
    elif basis[on_which]==1:
        basis[on_which]=0
        return {tuple(basis):1}
    else:
        print ("error")
        return None


def sigmaY_single(basis,**kwargs):
    on_which=kwargs['on_which']
    basis=list(basis)
    if basis[on_which]==0:
        basis[on_which]=1
        return {tuple(basis):sym.I}
    elif basis[on_which]==1:
        basis[on_which]=0
        return {tuple(basis):-sym.I}
    else:
        print ("error")
        return None


def sigmaZ_single(basis,**kwargs):
    on_which=kwargs['on_which']
    basis=list(basis)
    if basis[on_which]==0:
        return {tuple(basis):1}
    elif basis[on_which]==1:
        return {tuple(basis):-1}
    else:
        print ("error")
        return None



def Sphase_single(basis,**kwargs):
    on_which=kwargs['on_which']
    basis=list(basis)
    if basis[on_which]==0:
        return {tuple(basis):1}
    elif basis[on_which]==1:
        return {tuple(basis):sym.I}
    else:
        print ("error")
        return None


def Tpi8_single(basis,**kwargs):
    on_which=kwargs['on_which']
    basis=list(basis)
    if basis[on_which]==0:
        return {tuple(basis):1}
    elif basis[on_which]==1:
        return {tuple(basis):sym.exp(sym.I*sym.pi/4)}
    else:
        print ("error")
        return None


def RotationX_single(basis,**kwargs):
    on_which=kwargs['on_which']
    angle=kwargs['angle']
    basis=list(basis)
    MatrixU=sym.Matrix([[sym.cos(angle/2),-sym.I*sym.sin(angle/2)],
                        [-sym.I*sym.sin(angle/2),sym.cos(angle/2)]])
    if basis[on_which]==0:
        basis0=basis.copy()
        basis1=basis.copy()
        basis1[on_which]=1
        return {tuple(basis0):MatrixU[0,0],tuple(basis1):MatrixU[1,0]}
    elif basis[on_which]==1:
        basis0=basis.copy()
        basis1=basis.copy()
        basis0[on_which]=0
        return {tuple(basis0):MatrixU[0,1],tuple(basis1):MatrixU[1,1]}
    else:
        print ("error")
        return None

def RotationY_single(basis,**kwargs):
    on_which=kwargs['on_which']
    angle=kwargs['angle']
    basis=list(basis)
    MatrixU=sym.Matrix([[sym.cos(angle/2),-sym.sin(angle/2)],
                        [-sym.sin(angle/2),sym.cos(angle/2)]])
    if basis[on_which]==0:
        basis0=basis.copy()
        basis1=basis.copy()
        basis1[on_which]=1
        return {tuple(basis0):MatrixU[0,0],tuple(basis1):MatrixU[1,0]}
    elif basis[on_which]==1:
        basis0=basis.copy()
        basis1=basis.copy()
        basis0[on_which]=0
        return {tuple(basis0):MatrixU[0,1],tuple(basis1):MatrixU[1,1]}
    else:
        print ("error")
        return None

def RotationZ_single(basis,**kwargs):
    on_which=kwargs['on_which']
    angle=kwargs['angle']
    basis=list(basis)
    MatrixU=sym.Matrix([[sym.exp(-sym.I*angle/2),0],
                        [0,sym.exp(sym.I*angle/2)]])
    if basis[on_which]==0:
        basis0=basis.copy()
        basis1=basis.copy()
        basis1[on_which]=1
        return {tuple(basis0):MatrixU[0,0],tuple(basis1):MatrixU[1,0]}
    elif basis[on_which]==1:
        basis0=basis.copy()
        basis1=basis.copy()
        basis0[on_which]=0
        return {tuple(basis0):MatrixU[0,1],tuple(basis1):MatrixU[1,1]}
    else:
        print ("error")
        return None


def arbitrary_single(basis,**kwargs):
    on_which=kwargs['on_which']
    basis=list(basis)
    MatrixU=kwargs['MatrixRepre']
    if basis[on_which]==0:
        basis0=basis.copy()
        basis1=basis.copy()
        basis1[on_which]=1
        return {tuple(basis0):MatrixU[0,0],tuple(basis1):MatrixU[1,0]}
    elif basis[on_which]==1:
        basis0=basis.copy()
        basis1=basis.copy()
        basis0[on_which]=0
        return {tuple(basis0):MatrixU[0,1],tuple(basis1):MatrixU[1,1]}
    else:
        print ("error")
        return None



'---------------------- controlled gates ----------------------'


def AllControllersTrue(basis,controllers):
    result=True
    for i in controllers:
        result=(result and (basis[i])==1)
        if result==False:
            return result
    return result


def NOT(basis,**kwargs):
    controllers=kwargs['controllers']
    if kwargs['on_which'] in kwargs['controllers']:
        print ("error: on_which can't be in controllers")
        return None
    if AllControllersTrue(basis,controllers):
        return NOT_single(basis,**kwargs)
    else:
        return identity(basis,**kwargs)


def Hadamard(basis,**kwargs):
    controllers=kwargs['controllers']
    if kwargs['on_which'] in kwargs['controllers']:
        print ("error: on_which can't be in controllers")
        return None
    if AllControllersTrue(basis,controllers):
        return Hadamard_single(basis,**kwargs)
    else:
        return identity(basis,**kwargs)

def sigmaX(basis,**kwargs):
    controllers=kwargs['controllers']
    if kwargs['on_which'] in kwargs['controllers']:
        print ("error: on_which can't be in controllers")
        return None
    if AllControllersTrue(basis,controllers):
        return sigmaX_single(basis,**kwargs)
    else:
        return identity(basis,**kwargs)


def sigmaY(basis,**kwargs):
    controllers=kwargs['controllers']
    if kwargs['on_which'] in kwargs['controllers']:
        print ("error: on_which can't be in controllers")
        return None
    if AllControllersTrue(basis,controllers):
        return sigmaY_single(basis,**kwargs)
    else:
        return identity(basis,**kwargs)


def sigmaZ(basis,**kwargs):
    controllers=kwargs['controllers']
    if kwargs['on_which'] in kwargs['controllers']:
        print ("error: on_which can't be in controllers")
        return None
    if AllControllersTrue(basis,controllers):
        return sigmaZ_single(basis,**kwargs)
    else:
        return identity(basis,**kwargs)


def Sphase(basis,**kwargs):
    controllers=kwargs['controllers']
    if kwargs['on_which'] in kwargs['controllers']:
        print ("error: on_which can't be in controllers")
        return None
    if AllControllersTrue(basis,controllers):
        return Sphase_single(basis,**kwargs)
    else:
        return identity(basis,**kwargs)

def Tpi8(basis,**kwargs):
    controllers=kwargs['controllers']
    if kwargs['on_which'] in kwargs['controllers']:
        print ("error: on_which can't be in controllers")
        return None
    if AllControllersTrue(basis,controllers):
        return Tpi8_single(basis,**kwargs)
    else:
        return identity(basis,**kwargs)

def RotationX(basis,**kwargs):
    controllers=kwargs['controllers']
    if kwargs['on_which'] in kwargs['controllers']:
        print ("error: on_which can't be in controllers")
        return None
    if AllControllersTrue(basis,controllers):
        return RotationX_single(basis,**kwargs)
    else:
        return identity(basis,**kwargs)

def RotationY(basis,**kwargs):
    controllers=kwargs['controllers']
    if kwargs['on_which'] in kwargs['controllers']:
        print ("error: on_which can't be in controllers")
        return None
    if AllControllersTrue(basis,controllers):
        return RotationY_single(basis,**kwargs)
    else:
        return identity(basis,**kwargs)

def RotationZ(basis,**kwargs):
    controllers=kwargs['controllers']
    if kwargs['on_which'] in kwargs['controllers']:
        print ("error: on_which can't be in controllers")
        return None
    if AllControllersTrue(basis,controllers):
        return RotationZ_single(basis,**kwargs)
    else:
        return identity(basis,**kwargs)

def Toffoli(basis,**kwargs):
    controllers=kwargs['controllers']
    if kwargs['on_which'] in kwargs['controllers']:
        print ("error: on_which can't be in controllers")
        return None
    if len(controllers)!=2:
        print ("error")
    if AllControllersTrue(basis,controllers):
        return NOT_single(basis,**kwargs)
    else:
        return identity(basis,**kwargs)



def arbitrary(basis,**kwargs):
    controllers=kwargs['controllers']
    if kwargs['on_which'] in kwargs['controllers']:
        print ("error: on_which can't be in controllers")
        return None
    if AllControllersTrue(basis,controllers):
        return arbitrary_single(basis,**kwargs)
    else:
        return identity(basis,**kwargs)


'---------------------- several symbols ----------------------'

theta=sym.Symbol('theta')
phi=sym.Symbol('phi')



