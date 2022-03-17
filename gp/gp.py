import numpy as np
import random
import copy
import math
import sys
import warnings


# ----- Some parameter controlling the simulation 
# Global for now, will be moved later

# max depth for each tree creature:
MAX_DEPTH = 10


# These variables keep the list of available nodes and operations.
# They are set at the beginning of the program and only read by the functions
# below.
g_list_terminalnodes = []
g_list_operations = []


# used for re-scaling the epsilons used by the objective function
g_model_length = 1.0


#------------------------------------------------------------------------------
def compute_dot_product(v1, v2):
    return v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2]


def compute_length(v):
    return math.sqrt(v[0]**2 + v[1]**2 + v[2]**2)


def normalize(v):
    length = compute_length(v)
    if length == 0.0:
        return v
    return (v[0]/length, v[1]/length, v[2]/length)


# For numpy arrays:
def compute_length_vectorized(v):
    return np.sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2])


def normalize_vectorized(v):
    # return a modified fresh copy of v
    newv = copy.deepcopy(v)

    length = compute_length_vectorized(newv)

    # if a is a numpy array, a == 0.0 is returning a numpy array of bool
    # identify gradients with null-length and return them as is
    non_zero_idx = (length != 0.0)
    
    # re-scale the non-null gradients by their length
    newv[0][non_zero_idx] = v[0][non_zero_idx] / length[non_zero_idx]
    newv[1][non_zero_idx] = v[1][non_zero_idx] / length[non_zero_idx]
    newv[2][non_zero_idx] = v[2][non_zero_idx] / length[non_zero_idx]
    
    return newv


def compute_cross_product(v1, v2):
    return (-(v1[2]*v2[1])+v1[1]*v2[2], 
             v1[2]*v2[0]-v1[0]*v2[2],
             -(v1[1]*v2[0])+v1[0]*v2[1])


class Primitive(object):
    pass


class PlanePrimitive(Primitive):
    def __init__(self, parameters):
        # parameters: normalx normaly normalz dist
        self.normal_vec = (parameters[0], parameters[1], parameters[2])
        self.dist = parameters[3]

    def distance(self, position):
        return abs(self.dist - compute_dot_product(self.normal_vec, position))

    def signed_distance(self, position):
        d = compute_dot_product(self.normal_vec, position) - self.dist
        return -d

    def normal(self, position):
        return self.normal_vec

    def identifier(self):
        return 'plane'


class SpherePrimitive(Primitive):
    def __init__(self, parameters):
        # parameters: centerx centery centerz radius
        self.center = (parameters[0], parameters[1], parameters[2])
        self.radius = parameters[3]

    def distance(self, position):
        return abs(
            np.sqrt(
                (self.center[0]-position[0])**2 + 
                (self.center[1]-position[1])**2 + 
                (self.center[2]-position[2])**2) 
            - self.radius)

    def signed_distance(self, position):
        v = (self.center[0]-position[0], self.center[1]-position[1],
             self.center[2]-position[2])
        d = compute_length_vectorized(v) - self.radius
        return -d

    def normal(self, position):
        normal_vec = (self.center[0]-position[0], self.center[1]-position[1],
                      self.center[2]-position[2])
        return normalize_vectorized(normal_vec)

    def identifier(self):
        return 'sphere'


class CylinderPrimitive(Primitive):
    def __init__(self, parameters):
        # arguments: 
        # axis_dirx axis_diry axis_dirz axis_posx axis_posy axis_posz radius
        self.axis_dir = (parameters[0], parameters[1], parameters[2])
        self.axis_pos = (parameters[3], parameters[4], parameters[5])
        self.radius = parameters[6]

    def distance(self, position):
        diff = (position[0]-self.axis_pos[0], position[1]-self.axis_pos[1],
                position[2]-self.axis_pos[2])
        lamb = compute_dot_product(self.axis_dir, diff)
        v = (diff[0]-lamb*self.axis_dir[0], diff[1]-lamb*self.axis_dir[1],
             diff[2]-lamb*self.axis_dir[2])
        axis_dist = compute_length_vectorized(v)
        return abs(axis_dist - self.radius)

    def signed_distance(self, position):
        diff = (position[0]-self.axis_pos[0], position[1]-self.axis_pos[1],
                position[2]-self.axis_pos[2])
        lamb = compute_dot_product(self.axis_dir, diff)
        v = (diff[0]-lamb*self.axis_dir[0], diff[1]-lamb*self.axis_dir[1],
             diff[2]-lamb*self.axis_dir[2])
        axis_dist = compute_length_vectorized(v)
        d = axis_dist - self.radius
        return -d

    def normal(self, position):
        diff = (position[0]-self.axis_pos[0], position[1]-self.axis_pos[1],
                position[2]-self.axis_pos[2])
        lamb = compute_length_vectorized(self.axis_dir, diff)
        normal_vec = (diff[0]-lamb*self.axis_dir[0], 
                      diff[1]-lamb*self.axis_dir[1],
                      diff[2]-lamb*self.axis_dir[2])
        return normalize_vectorized(normal_vec)

    def identifier(self):
        return 'cylinder'


class TorusPrimitive(Primitive):
    def __init__(self, parameters):
        # parameters: 
        # normalx normaly normalz centerx centery centerz rminor rmajor
        self.normal_vec = (parameters[0], parameters[1], parameters[2])
        self.center = (parameters[3], parameters[4], parameters[5])
        self.rminor = parameters[6]
        self.rmajor = parameters[7]

    def distance(self, position):
        s = (position[0]-self.center[0], position[1]-self.center[1],
             position[2]-self.center[2])
        spin1 = compute_dot_product(self.normal_vec, s)
        spin0vec = (s[0]-spin1*self.normal_vec[0], 
                    s[1]-spin1*self.normal_vec[1],
                    s[2]-spin1*self.normal_vec[2])
        spin0 = compute_length_vectorized(spin0vec)
        spin0 = spin0 - self.rmajor
        return abs(np.sqrt(spin0*spin0 + spin1*spin1) - self.rminor)

    def signed_distance(self, position):
        s = (position[0]-self.center[0], position[1]-self.center[1],
             position[2]-self.center[2])
        spin1 = compute_dot_product(self.normal_vec, s)
        spin0vec = (s[0]-spin1*self.normal_vec[0], 
                    s[1]-spin1*self.normal_vec[1],
                    s[2]-spin1*self.normal_vec[2])
        spin0 = compute_length_vectorized(spin0vec)
        spin0 = spin0 - self.rmajor
        d = np.sqrt(spin0*spin0 + spin1*spin1) - self.rminor
        return -d

    def normal(self, position):
        s = (position[0]-self.center[0], position[1]-self.center[1],
             position[2]-self.center[2])
        spin1 = compute_dot_product(self.normal_vec, s)
        tmp = (spin1*self.normal_vec[0], spin1*self.normal_vec[1], 
               spin1*self.normal_vec[2])
        spin0vec = (s[0]-tmp[0],s[1]-tmp[1],s[2]-tmp[2])
        spin0 = compute_length_normalized(spin0vec)
        spin0 = spin0 - self.rmajor
        pln = compute_cross_product(s, self.normal_vec)
        plx = compute_cross_product(self.normal_vec, pln)
        plx = normalize_vectorized(plx)
        n = (spin0*plx[0]+tmp[0], spin0*plx[1]+tmp[1], spin0*plx[2]+tmp[2])
        n = n / np.sqrt(spin0*spin0 + spin1*spin1)
        return n

    def identifier(self):
        return 'torus'
        
class ConePrimitive(Primitive):
    def __init__(self, parameters):
        # parameters: 
        # axis_dirx axis_diry axis_dirz centerx centery centerz angle
        self.axis_dir = (parameters[0], parameters[1], parameters[2])
        self.center = (parameters[3], parameters[4], parameters[5])
        self.angle = parameters[6]

    def distance(self, position):
        s = (position[0]-self.center[0], position[1]-self.center[1], 
             position[2]-self.center[2])
        g = compute_dot_product(s, self.axis_dir)
        slen = compute_length_vectorized(s)
        sqrs = slen*slen
        f = sqrs - g*g

        #if f <= 0.0:
        #    f = 0.0
        #else:
        #    f = np.sqrt(f)
        f = np.maximum(f, 0.0)
        f = np.sqrt(f)

        da = np.cos(self.angle) * f
        db = -np.sin(self.angle) * g
        
        #if (g < 0.0) and ((da - db) < 0.0):
        #    return np.sqrt(sqrs)
        #return abs(da + db)
        
        res = abs(da + db)
        #idx = (g < 0.0) and ((da - db) < 0.0)
        idx = np.logical_and((g < 0.0), ((da - db) < 0.0))
        res[idx] = np.sqrt(sqrs)

        return res

    def signed_distance(self, position):
        s = (position[0]-self.center[0], position[1]-self.center[1], 
             position[2]-self.center[2])
        g = compute_dot_product(s, self.axis_dir)
        slen = compute_length_vectorized(s)
        sqrs = slen*slen
        f = sqrs - g*g

        #if f <= 0.0:
        #    f = 0.0
        #else:
        #    f = np.sqrt(f)
        f = np.maximum(f, 0.0)
        f = np.sqrt(f)

        da = np.cos(self.angle) * f
        db = -np.sin(self.angle) * g
        
        #if (g < 0.0) and ((da - db) < 0.0):
        #    return -(np.sqrt(sqrs))
        #return -(da + db)
        
        res = -(da + db)
        #idx = (g < 0.0) and ((da - db) < 0.0)
        idx = np.logical_and((g < 0.0), ((da - db) < 0.0))
        res[idx] = -np.sqrt(sqrs)

        return res


    def normal(self, position):
        s = (position[0]-self.center[0], position[1]-self.center[1],
             position[2]-self.center[2])
        pln = compute_cross_product(s, self.axis_dir)
        plx = compute_cross_product(self.axis_dir, pln)
        plx = normalize_vectorized(plx)
        n0 = np.cos(-self.angle)
        sa = np.sin(-self.angle)
        ny = (sa * self.axis_dir[0], sa * self.axis_dir[1], 
              sa * self.axis_dir[2])
        n = (n0 * plx[0] + ny[0], n0 * plx[1] + ny[1], n0 * plx[2] + ny[2])
        # return normalize(n) ? 
        return n

    def identifier(self):
        return 'cone'


# Added 2013/5/3 - Ellipsoid
def apply_inverse_rotation(p, theta, phi, psi):
    ctheta = math.cos(theta)
    stheta = math.sin(theta)
    cphi = math.cos(phi)
    sphi = math.sin(phi)
    cpsi = math.cos(psi)
    spsi = math.sin(psi)
    
    ptx = ctheta*cphi*p[0] + (spsi*stheta*cphi - cpsi*sphi)*p[1] + (cpsi*stheta*cphi + spsi*sphi)*p[2]
    pty = ctheta*sphi*p[0] + (spsi*stheta*sphi + cpsi*cphi)*p[1] + (spsi*stheta*cphi - cpsi*sphi)*p[2]
    ptz = -stheta*p[0] + spsi*ctheta*p[1] + cpsi*ctheta*p[2]

    pt = (ptx, pty, ptz)

    return pt


class EllipsoidPrimitive(Primitive):
    def __init__(self, parameters):
        assert(len(parameters) == 9)
        self.c = (parameters[0], parameters[1], parameters[2])
        self.r = (parameters[3], parameters[4], parameters[5])
        self.theta, self.phi, self.psi = (parameters[6], parameters[7], parameters[8])

    def distance(self, position):
        '''Approximation of the unsigned distance by the Taubin distance'''
        return abs(self.signed_distance(position))

    def signed_distance(self, position):
        '''Approximation of the signed distance by the Taubin distance'''
        val = self._eval(position)
        g = self._grad(position)
        # if abs(norm) > g_epsilon:
        #    val = val/norm
        norm = compute_length_vectorized(g)
        nz_idx = (norm != 0.0)
        val[nz_idx] = val[nz_idx] / norm[nz_idx]
        return val

    def normal(self, position):
        '''Compute the normal at the point position by computing the gradient
        and normalizing it.
        '''
        normal_vec = self._grad(position)
        return normalize_vectorized(normal_vec)

    def _grad(self, position):
        '''Compute the gradient of _eval at the given point position.
        The gradient is approximated by finite differences (for now).
        I will compute it analytically later and use the expression 
        here instead.
        '''
        epsilon = 1e-8
        
        x = position[0]
        y = position[1]
        z = position[2]
        f = self._eval(position)
        xp,yp,zp = (x+epsilon, y, z)
        fp = self._eval((xp,yp,zp))
        dfdx = (fp - f) / epsilon

        xp,yp,zp = (x, y+epsilon, z)
        fp = self._eval((xp,yp,zp))
        dfdy = (fp - f) / epsilon

        xp,yp,zp = (x, y, z+epsilon)
        fp = self._eval((xp,yp,zp))
        dfdz = (fp - f) / epsilon

        g = (dfdx, dfdy, dfdz)
        return g


    def _eval(self, position):
        '''Eval f(x,y,z) = (x/a)^2 + (y/b)^2 + (z/c)^2 - 1 at 
        (x,y,z) given by position'''
        p = (position[0] - self.c[0], position[1] - self.c[1], position[2] - self.c[2])
        pt = apply_inverse_rotation(p, self.theta, self.phi, self.psi)
        val = (pt[0]/self.r[0])**2 + (pt[1]/self.r[1])**2 + (pt[2]/self.r[2])**2 - 1.0
        return val

    def identifier(self):
        return 'ellipsoid'


def create_primitive_instance(name, parameters):
    ''' 
    Create an instance of the appropriate primitive based on the primitive
    name.
    '''
    lcname = name.lower()
    if lcname == 'plane':
        return PlanePrimitive(parameters)
    if lcname == 'sphere':
        return SpherePrimitive(parameters)
    if lcname == 'cylinder':
        return CylinderPrimitive(parameters)
    if lcname == 'torus':
        return TorusPrimitive(parameters)
    if lcname == 'cone':
        return ConePrimitive(parameters)
    if lcname == 'ellipsoid':
        return EllipsoidPrimitive(parameters)
    raise Exception('Unknown primitive')


def read_fit(fit_filename):
    f = open(fit_filename)

    list_primitives = []

    for line in f:
        elements = line.strip().split()
        if len(elements)==0:
            # empty line
            continue
        # elements will look like:
        # primitive_name parameter_1 parameter_2 ... parameter_n
        primitive_name = elements[0]
        parameters = []
        for i in range(1, len(elements)):
            parameters.append(float(elements[i]))

        list_primitives.append(
            create_primitive_instance(primitive_name, parameters))

    f.close()
    return list_primitives


# -----------------------------------------------------------------------------

# Creature representation.
# The representation of the trees corresponding to the programs
# being evolved.

class fwrapper(object):
    '''
    A wrapper for the functions that will be used on function 
    nodes. Its member variables are the name of the function,
    the function itself, and the number of parameters it takes.
    '''
    def __init__(self, function, childcount, name):
        self.function = function
        self.childcount = childcount
        self.name = name


class node(object):
    '''
    The class for function nodes (nodes with children). This is 
    initialized with an fwrapper. When evaluate is called, it 
    evaluates the child nodes and then applies the function
    to their results.
    '''
    def __init__(self, fw, children):
        self.function = fw.function
        self.name = fw.name
        self.children = children

    def evaluate(self, inp):
        results = [n.evaluate(inp) for n in self.children]
        return self.function(results)

    def display(self, indent=0):
        print (' ' * indent) + self.name
        for c in self.children:
            c.display(indent+1)

    def to_string(self):
        str_to_display = self.name + '['
        num_children = len(self.children)
        for i in range(num_children-1):
            str_to_display = str_to_display + self.children[i].to_string() + ','

        last_child = self.children[num_children-1]
        str_to_display = str_to_display + last_child.to_string()
        str_to_display = str_to_display + ']'
        return str_to_display

    def compute_number_nodes(self):
        ''' Compute the number of nodes (internal nodes and leaves) for 
        the tree.
        '''
        number_nodes = 0
        for c in self.children:
            number_nodes = number_nodes + c.compute_number_nodes()
        return 1 + number_nodes

    def max_depth(self):
        '''
        Returns the depth of the deepest branch of the tree
        '''
        max_depth_children = 0
        for c in self.children:
            max_depth_children = max(max_depth_children, c.max_depth())
        return 1 + max_depth_children


class paramnode(object):
    '''
    The class for nodes that only return one of the parameters
    passed to the program. Its evaluate method returns the parameter
    specified by idx.
    '''
    def __init__(self, idx):
        self.idx = idx

    def evaluate(self, inp):
        return inp[self.idx]

    def display(self, indent=0):
        print('%sp%d' % (' '*indent, self.idx))

    def to_string(self):
        str_to_display = 'x[' + str(self.idx) + ']'
        return str_to_display

    def compute_number_nodes(self):
        return 1

    def max_depth(self):
        '''
        Returns the depth of the deepest branch of the tree
        '''
        return 1


class constnode(object):
    '''
    Nodes that return a constant value. The evaluate method simply
    returns the value with which it was initialized.
    '''
    def __init__(self, v):
        self.v = v

    def evaluate(self, inp):
        return self.v

    def display(self, indent=0):
        print ('%s%d' % (' '*indent, self.v))

    def to_string(self):
        str_to_display = str(self.v)
        return str_to_display

    def compute_number_nodes(self):
        return 1

    def max_depth(self):
        '''
        Returns the depth of the deepest branch of the tree
        '''
        return 1


class terminalnode(object):
    '''
    Terminals are leaves. They serve as a wrapper to a function 
    to be evaluated at a point coordinate and return the corresponding 
    value.
    This class serves as a wrapper to the fitted primitives.

    Note: What about using fwrapper with self.children = 0 instead of 
    that ??
    TODO: rename, e.g. primitive?
    '''
    def __init__(self, function, name):
        self.function = function
        self.name = name

    def evaluate(self, inp):
        x = inp[0]
        y = inp[1]
        z = inp[2]
        return self.function([x,y,z])

    def display(self, indent=0):
        print ('%s%s' % (' '*indent, self.name))

    def to_string(self):
        str_to_display = self.name
        return str_to_display

    def compute_number_nodes(self):
        return 1

    def max_depth(self):
        '''
        Returns the depth of the deepest branch of the tree
        '''
        return 1


# -------------------------- Geometry Kernel --------------------------

# Operations:

def union(f1, f2):
    # used when f1 and f2 are scalar values
    # return max(f1, f2)
    # used when f1 and f2 are numpy arrays
    return np.maximum(f1, f2)

def intersection(f1, f2):
    # used when f1 and f2 are scalar values
    # return min(f1, f2)
    # used when f1 and f2 are numpy arrays
    return np.minimum(f1, f2)

def negation(f):
    return -f

def subtraction(f1, f2):
    return intersection(f1, negation(f2))


# Primitives:

def plane(p, parameter):
    '''
    Return the distance from p to the plane defined by the parameters 
    specified in the argument parameter.
    parameter: list of 3 elements defining a plane
    parameter[0], parameter[1]: angles defining a unit vector normal to 
    the plane
    parameter[2]: distance of the plane along the normal
    '''
    
    theta = parameter[0]
    phi = parameter[1]
    d = parameter[2]

    nx = math.cos(theta) * math.sin(phi)
    ny = math.sin(theta) * math.sin(phi)
    nz = math.cos(phi)

    return nx * p[0] + ny * p[1] + nz * p[2] + d


# --------------------------- instantiate some primitives ---------------

plane1 = terminalnode(lambda p: plane(p, [4.71239, 1.5708, 0.166147]), 'plane1')
plane2 = terminalnode(lambda p: plane(p, [4.71239, 1.5708, 0.000521799]), 'plane2')
plane3 = terminalnode(lambda p: plane(p, [3.14149, 1.56995, 0.416007]), 'plane3')
plane4 = terminalnode(lambda p: plane(p, [3.35771, 0.000863486, -0.583125]), 'plane4')
plane5 = terminalnode(lambda p: plane(p, [3.14139, 1.574, 0.167953]), 'plane5')
plane6 = terminalnode(lambda p: plane(p, [3.14159, 1.5708, 0]), 'plane6')
plane7 = terminalnode(lambda p: plane(p, [6.22281, 3.13679, 0.165366]), 'plane7')
plane8 = terminalnode(lambda p: plane(p, [1.9333, 3.14159, 0.416667]), 'plane8')
plane9 = terminalnode(lambda p: plane(p, [6.28306, 1.57155, -0.583099]), 'plane9')
plane10 = terminalnode(lambda p: plane(p, [3.11382, 3.13535, 0.837564]), 'plane10')
plane11 = terminalnode(lambda p: plane(p, [1.05459, 0, -1]), 'plane11')
plane12 = terminalnode(lambda p: plane(p, [3.14159, 1.5708, 0.833333]), 'plane12')
plane13 = terminalnode(lambda p: plane(p, [5.3107, 3.14159, 0]), 'plane13')
plane14 = terminalnode(lambda p: plane(p, [3.14159, 1.5708, 1]), 'plane14')


terminals_list = [plane1, plane2, plane3, plane4, plane5, plane6, plane7, plane8, plane9, plane10, plane11, plane12, plane13, plane14]


# ----------------------------- instantiate some operations -------------

unionw = fwrapper(lambda f: union(f[0],f[1]), 2, 'union')
intersectionw = fwrapper(lambda f: intersection(f[0],f[1]), 2, 'intersection')
negationw = fwrapper(lambda f: negation(f[0]), 1, 'negation')
subtractionw = fwrapper(lambda f: subtraction(f[0], f[1]), 2, 'subtraction')


operations_list = [unionw, intersectionw, negationw, subtractionw]


# The functions needed by the GP: objective function, creation of random 
# tree, mutation, crossover, .....


def create_initial_population():
    pass


def makerandomtree(maxdepth=4, opr=0.7):
    '''
    Create a random program.
    Return a new tree.
    Args:
        Is it needed anymore??
        maxdepth: maximum depth for the random tree
        opr: probability to draw an operation
    '''
    if random.random() < opr and maxdepth > 0:
        f = random.choice(g_list_operations)
        children = [makerandomtree(maxdepth-1, opr) for i in range(f.childcount)]
        return node(f, children)
    else:
        leaf = random.choice(g_list_terminalnodes)
        return leaf


def fix_dot(dot):
    ''' due to fp computation values for the dot product may be outside of 
    [-1,1], e.g. 1.00000001. Fix the values that go outside of this range.'''
    if dot < -1.0:
        return -1.0
    if dot > 1.0:
        return 1.0
    return dot


def scorefunction(tree, dataset):
    '''
    Use to test a program to see how close it gets to the correct
    answers for the dataset.
    
    Args:
        tree: the program that needs to be evaluated
        dataset: the dataset used for the evaluation
    '''

    # The program corresponds to an implicit surface that should 
    # evaluate to 0 on each input point from the dataset.
    # To remove useless cases (function is null everywhere) we also 
    # check how the gradient of the implicit surface deviates from the 
    # surface normal

    # 1) distance error
    # error tolerance

    #error_epsilon = 0.01
    error_epsilon = g_model_length * 0.01

    distance_error = 0
    for data in dataset:
        point = [data[0], data[1], data[2]]
        v = tree.evaluate(point)
        v = v / error_epsilon
        distance_error += math.exp(-v*v)


    # 2) normal deviation
    # normal deviation tolerance
    
    #normal_epsilon = 0.01
    normal_epsilon = 0.01 * g_model_length

    normal_epsilon_sq = normal_epsilon * normal_epsilon
    # step size used to compute the gradient numerically
    delta = 0.00001
    # accumulated error
    normal_error = 0
    for data in dataset:
        # p and normal at p
        point = [data[0], data[1], data[2]]
        normal = [data[3], data[4], data[5]]
        f = tree.evaluate(point)

        # p + dx 
        pointdx = [data[0]+delta, data[1], data[2]]
        df = tree.evaluate(pointdx)
        dfdx = (df - f) / delta

        # p + dy
        pointdy = [data[0], data[1]+delta, data[2]]
        df = tree.evaluate(pointdy)
        dfdy = (df - f) / delta

        # p + dz
        pointdz = [data[0], data[1], data[2]+delta]
        df = tree.evaluate(pointdz)
        dfdz = (df - f) / delta

        # in the xyzn file the normals are pointing to the outside
        # so I need to use -grad f
        gradf = [-dfdx, -dfdy, -dfdz]

        # normalize the gradient
        gradf = normalize(gradf)

        dot = compute_dot_product(gradf, normal)
        # make sure that values of dot are in [-1, 1]
        dot = fix_dot(dot)
        angle = math.acos(dot)
        theta = angle / normal_epsilon
        
        normal_error += math.exp(-theta**2)

    # penalize big models
    num_nodes = tree.compute_number_nodes()
    num_points = len(points)

    objective_value = (distance_error + normal_error - 
                       (1.0/2.0)*num_nodes*math.log(num_points))

    return max(objective_value, 0.0)


def scorefunction_BIC(tree, dataset):
    '''
    Use to test a program to see how close it gets to the correct
    answers for the dataset.
    
    Args:
        tree: the program that needs to be evaluated
        dataset: the dataset used for the evaluation

    See Pattern matching and machine learning by Bishop, pp. 217, 
    Eq. 4.137.
    
    Note the appropriate normalization function needs to be called:
    normalize_fitness_BIC.
    '''

    # The program corresponds to an implicit surface that should 
    # evaluate to 0 on each input point from the dataset.
    # To remove useless cases (function is null everywhere) we also 
    # check how the gradient of the implicit surface deviates from the 
    # surface normal

    # 1) distance error
    # error tolerance
    
    #error_epsilon = 0.01
    error_epsilon = 0.01 * g_model_length

    distance_error = 0
    for data in dataset:
        point = [data[0], data[1], data[2]]
        v = tree.evaluate(point)
        v = v / error_epsilon
        distance_error += math.exp(-v*v)


    # 2) normal deviation
    # normal deviation tolerance

    #normal_epsilon = 0.01
    normal_epsilon = 0.01 * g_model_length
    

    normal_epsilon_sq = normal_epsilon * normal_epsilon
    # step size used to compute the gradient numerically
    delta = 0.00001
    # accumulated error
    normal_error = 0
    for data in dataset:
        # p and normal at p
        point = [data[0], data[1], data[2]]
        normal = [data[3], data[4], data[5]]
        f = tree.evaluate(point)

        # p + dx 
        pointdx = [data[0]+delta, data[1], data[2]]
        df = tree.evaluate(pointdx)
        dfdx = (df - f) / delta

        # p + dy
        pointdy = [data[0], data[1]+delta, data[2]]
        df = tree.evaluate(pointdy)
        dfdy = (df - f) / delta

        # p + dz
        pointdz = [data[0], data[1], data[2]+delta]
        df = tree.evaluate(pointdz)
        dfdz = (df - f) / delta

        # in the xyzn file the normals are pointing to the outside
        # so I need to use -grad f
        gradf = [-dfdx, -dfdy, -dfdz]

        # normalize the gradient
        gradf = normalize(gradf)

        dot = compute_dot_product(gradf, normal)
        # make sure that values of dot are in [-1, 1]
        dot = fix_dot(dot)
        angle = math.acos(dot)
        theta = angle / normal_epsilon
        
        normal_error += math.exp(-theta**2)

    # penalize big models
    num_nodes = tree.compute_number_nodes()
    num_points = len(points)


    pr = 1.0/(2.0*num_points)*(distance_error + normal_error)
    objective_value = math.log(pr) - (1.0/2.0)*num_nodes*math.log(num_points)

    return objective_value



def fix_dot_np(dot):
    ''' due to fp computation values for the dot product may be outside of 
    [-1,1], e.g. 1.00000001. Fix the values that go outside of this range.'''
    below = dot < -1.0
    above = dot > 1.0
    dot[below] = -1.0
    dot[above] = 1.0
    return dot


def scorefunction_vectorized(tree, points, normals):
    '''
    Use to test how close a program  is to the correct
    answer associated to the dataset.
    This version is vectorized and assume numpy arrays as input.
    
    Args:
        tree: the program that needs to be evaluated.
        points: the point-set used for the evaluation (as a numpy array).
        normals: the normals at each point (as a numpy array).
    '''

    # The program corresponds to an implicit surface that should 
    # evaluate to 0 on each input point from the dataset.
    # To remove useless cases (function is null everywhere) we also 
    # check how the gradient of the implicit surface deviates from the 
    # surface normal

    # 1) distance error
    # error tolerance

    #error_epsilon = 0.01
    error_epsilon = 0.01 * g_model_length


    distance_error = 0
    point = [points[:,0], points[:,1], points[:,2]]
    v = tree.evaluate(point)
    v = v / error_epsilon
    distance_error = np.sum(np.exp(-v*v))


    # 2) normal deviation
    # normal deviation tolerance
    
    #normal_epsilon = 0.01
    normal_epsilon = 0.01 * g_model_length
    

    normal_epsilon_sq = normal_epsilon * normal_epsilon
    # step size used to compute the gradient numerically
    delta = 0.00001
    
    # accumulated error
    normal_error = 0
    point = [points[:,0], points[:,1], points[:,2]]
    normal = [normals[:,0], normals[:,1], normals[:,2]]
    f = tree.evaluate(point)
    
    # eval at p+dx
    pointdx = [points[:,0]+delta, points[:,1], points[:,2]]
    df = tree.evaluate(pointdx)
    dfdx = (df - f) / delta

    # eval at p+dy
    pointdy = [points[:,0], points[:,1]+delta, points[:,2]]
    df = tree.evaluate(pointdy)
    dfdy = (df - f) / delta

    # eval at p+dz
    pointdz = [points[:,0], points[:,1], points[:,2]+delta]
    df = tree.evaluate(pointdz)
    dfdz = (df - f) / delta

    # in the xyzn file the normals are pointing to the outside
    # so I need to use -grad f
    gradf = [-dfdx, -dfdy, -dfdz]

    # normalize the gradient
    gradf = normalize_vectorized(gradf)

    dot = compute_dot_product(gradf, normal)
    
    # make sure that values of dot are in [-1, 1]
    dot = fix_dot_np(dot)
    angle = np.arccos(dot)
    theta = angle / normal_epsilon

    normal_error = np.sum(np.exp(-theta**2))

    # penalize big models
    num_nodes = tree.compute_number_nodes()
    num_points = len(points)

    
    objective_value = (distance_error + normal_error - 
                       (1.0/2.0)*num_nodes*math.log(num_points))

    return np.maximum(objective_value, 0.0)


def scorefunction_vectorized_BIC(tree, points, normals):
    '''
    Use to test how close a program  is to the correct
    answer associated to the dataset.
    This version is vectorized and assume numpy arrays as input.
    
    Args:
        tree: the program that needs to be evaluated.
        points: the point-set used for the evaluation (as a numpy array).
        normals: the normals at each point (as a numpy array).

    See Pattern matching and machine learning by Bishop, pp. 217, 
    Eq. 4.137.
    
    Note the appropriate normalization function needs to be called:
    normalize_fitness_BIC.
    '''

    # The program corresponds to an implicit surface that should 
    # evaluate to 0 on each input point from the dataset.
    # To remove useless cases (function is null everywhere) we also 
    # check how the gradient of the implicit surface deviates from the 
    # surface normal

    # 1) distance error
    # error tolerance
    
    #error_epsilon = 0.01
    error_epsilon = 0.01 * g_model_length

    distance_error = 0
    point = [points[:,0], points[:,1], points[:,2]]
    v = tree.evaluate(point)
    v = v / error_epsilon
    distance_error = np.sum(np.exp(-v*v))


    # 2) normal deviation
    # normal deviation tolerance
    #normal_epsilon = 0.01
    normal_epsilon = 0.01 * g_model_length

    normal_epsilon_sq = normal_epsilon * normal_epsilon
    # step size used to compute the gradient numerically
    delta = 0.00001
    
    # accumulated error
    normal_error = 0
    point = [points[:,0], points[:,1], points[:,2]]
    normal = [normals[:,0], normals[:,1], normals[:,2]]
    f = tree.evaluate(point)
    
    # eval at p+dx
    pointdx = [points[:,0]+delta, points[:,1], points[:,2]]
    df = tree.evaluate(pointdx)
    dfdx = (df - f) / delta

    # eval at p+dy
    pointdy = [points[:,0], points[:,1]+delta, points[:,2]]
    df = tree.evaluate(pointdy)
    dfdy = (df - f) / delta

    # eval at p+dz
    pointdz = [points[:,0], points[:,1], points[:,2]+delta]
    df = tree.evaluate(pointdz)
    dfdz = (df - f) / delta

    # in the xyzn file the normals are pointing to the outside
    # so I need to use -grad f
    gradf = [-dfdx, -dfdy, -dfdz]

    # normalize the gradient
    gradf = normalize_vectorized(gradf)

    dot = compute_dot_product(gradf, normal)
    
    # make sure that values of dot are in [-1, 1]
    dot = fix_dot_np(dot)
    angle = np.arccos(dot)
    theta = angle / normal_epsilon

    normal_error = np.sum(np.exp(-theta**2))

    # penalize big models
    num_nodes = tree.compute_number_nodes()
    num_points = len(points)

    
    # likelihood
    pr = 1.0/(2.0*num_points) * (distance_error + normal_error)
    
    # when pr == 0.0, log(pr) triggers a 'math domain error'
    if pr == 0.0:
        pr = sys.float_info.min
    
    # objective function: log likelihood + BIC
    objective_value = math.log(pr) - (1.0/2.0)*num_nodes*math.log(num_points)


    return objective_value



def normalize_fitness(scores):
    '''
    Compute the normalized and adjusted fitness for 
    each individual in the population.
    
    Args:
        scores: array of 2-uple. The first element corresponds
        to the fitness value and the second element to the creature.
    '''
    normalized_scores = copy.deepcopy(scores)
    sum_adjusted_scores = 0
    unscaled_score = 0
    #for (score, creature) in scores:
    for i in range(len(scores)):
        unscaled_score = normalized_scores[i][0]
        # adjust the fitness value: f -> 1.0 / (1.0 + f)
        normalized_scores[i][0] = 1.0 / (1.0 + normalized_scores[i][0])
        # keep track of the total sum
        sum_adjusted_scores += normalized_scores[i][0]
        normalized_scores[i][2] = unscaled_score

    for i in range(len(normalized_scores)):
        # normalize the fitness value: f -> f / sum f
        normalized_scores[i][0] = normalized_scores[i][0] / sum_adjusted_scores

    return normalized_scores


def normalize_fitness_BIC(scores):
    '''
    Compute the normalized and adjusted fitness for 
    each individual in the population.
    
    Args:
        scores: array of 2-uple. The first element corresponds
        to the fitness value and the second element to the creature.


    normalize the BIC objective function
    '''
    normalized_scores = copy.deepcopy(scores)
    sum_adjusted_scores = 0
    unscaled_score = 0
    #for (score, creature) in scores:
    for i in range(len(scores)):
        unscaled_score = normalized_scores[i][0]

        # 1.0 - Exp[score] where:
        # score := Log[likelihood] + BIC]
        # this is in [0,1] and decreasing
        normalized_scores[i][0] = 1.0 - math.exp(normalized_scores[i][0])

        # keep track of the total sum
        sum_adjusted_scores += normalized_scores[i][0]
        normalized_scores[i][2] = unscaled_score

    for i in range(len(normalized_scores)):
        # normalize the fitness value: f -> f / sum f
        normalized_scores[i][0] = normalized_scores[i][0] / sum_adjusted_scores

    return normalized_scores


def mutate(tree, mutation_rate=0.1):
    '''
    Mutate a tree
    '''
    r = random.random()
    if r < mutation_rate:
        return _mutate(tree, probchange=0.1)
    else:
        return copy.deepcopy(tree)


def _mutate(tree, probchange=0.1):
    if random.random() < probchange:
        return makerandomtree()
    else:
        # find number of nodes in the tree
        number_nodes = tree.compute_number_nodes()
        # generate a random number between 0 and #nodes --> mutation point
        mutation_point = random.randint(0,number_nodes-1)
        # generate a new random tree (with limited max depth) --> new node
        new_node = makerandomtree()
        # call _replace with the mutation point and new node 
        # print('mutation point %d' % (mutation_point))
        return _replace(tree, new_node, mutation_point)


def _replace(tree, new_node, point):
    # replace the node of tree at position point by new_node
    copy_tree = copy.deepcopy(tree)
    return _replace_rec(copy_tree, new_node, point)


# This one looks like wrong too ......
def _replace_rec(tree, new_node, point):
    if point==0:
        return new_node
    else:
        result = copy.deepcopy(tree)
        if isinstance(result, node):
            for i in range(len(result.children)):
                c = result.children[i]
                number_nodes = c.compute_number_nodes()
                if point <= number_nodes:
                    result.children[i] = _replace_rec(c, new_node, point-1)
                    return result
                else:
                    # result.children[i] = copy.deepcopy(c)
                    point = point - (number_nodes)
        # return result


# The current problem with this crossover is that it lets the new trees
# potentially grow to infinity.
def crossover(tree1, tree2, crossover_rate=0.5):
    '''
    Breed two trees
    '''
    r = random.random()
    if r < crossover_rate:
        return _crossover(tree1, tree2)
    else:
        return (copy.deepcopy(tree1), copy.deepcopy(tree2))


def _crossover(tree1, tree2):
    # make copies of tree1 and tree2
    tree1_copy = copy.deepcopy(tree1)
    tree2_copy = copy.deepcopy(tree2)
    # find number of nodes #tree1 and #tree2 in tree1 and tree2
    number_nodes_tree1 = tree1.compute_number_nodes()
    number_nodes_tree2 = tree2.compute_number_nodes()
    # generate random numbers in [0,#tree1-1] and [0,#tree2-1] <-- crossover points
    crossover_point1 = random.randint(0, number_nodes_tree1 - 1)
    #print(crossover_point1)
    crossover_point2 = random.randint(0, number_nodes_tree2 - 1)
    #print(crossover_point2)
    # find node of tree1 at position #tree1 and node of tree2 at position #tree2
    node1 = find_node_at(tree1_copy, crossover_point1)
    #print(node1.to_string())
    node2 = find_node_at(tree2_copy, crossover_point2)
    #print(node2.to_string())
    # replace node #tree1 of tree1 with node #tree2 of tree2 and 
    #      node #tree2 of tree2 with node #tree1 of tree1
    new_tree1 = _replace_rec(tree1_copy, node2, crossover_point1)
    new_tree2 = _replace_rec(tree2_copy, node1, crossover_point2)
    
    # make sure the new creatures have a reasonable depth:
    return validate_crossover(tree1_copy, new_tree1, tree2_copy, new_tree2, MAX_DEPTH)
    # return (new_tree1, new_tree2)


def find_node_at(tree, point):
    if point == 0:
        return copy.deepcopy(tree)
    else:
        if isinstance(tree, node):
            for c in tree.children:
                number_nodes = c.compute_number_nodes()
                if point <= number_nodes:
                    return (find_node_at(c, point-1))
                else:
                    #point = point - (number_nodes + 1)
                    point = point - number_nodes


def validate_crossover(old_tree1, new_tree1, old_tree2, new_tree2, max_depth):
    '''
    Given old (before crossover) and new (after crossover) creatures
    check to see whether the maximum depth was exceeded in each tree.
    If either of the new individuals has exceeded the maximum depth
    then the old creatures are used.

    Note:
    old_tree1 and old_tree2 are copies of the original tree1 and tree2
    new_tree1 and new_tree2 are new creatures obtained after crossover.
    '''
    # compute max depth of new creatures.
    new_tree1_depth = new_tree1.max_depth()
    new_tree2_depth = new_tree2.max_depth()

    # if either of the new creatures has exceeded a user defined max_depth
    # then we keep the old creatures.
    if new_tree1_depth > max_depth or new_tree2_depth > max_depth:
        return (old_tree1, old_tree2)
    else:
        return (new_tree1, new_tree2)


def evolve(popsize, rankfunction, maxgen=500, mutationrate=0.1, breedingrate=0.4, pexp=0.7):
    '''
    Setup a competitive environment in which programs can evolve.
    Example of use:
    >>> import gp
    >>> rf=gp.getrankfunction(gp.buildhiddenset())
    >>> gp.evolve(500,rf,mutationrate=0.2,breedingrate=0.1,pexp=0.7)

    popsize: size of the population
    rankfunction: the function used on the list of programs to rank
                  them from best to worst
    mutationrate: probability of mutation
    breedingrate: probability of crossover
    probexp: rate of decline in the probability of selecting lower
             ranked programs.
             When pexp -> 1.0, it allows to select lower ranked programs
             When pexp -> 0, it allows to select only good ranked programs
    '''


    # TODO: no guarantee that selectindex() returns a number 
    # in [0, popsize]
    # if it returns a number > popsize an array index error will happen 
    # in the while loop below
    def selectindex():
        return int(math.log(random.random()) / math.log(pexp))


    def tournament(scores):
        # Assume scores is an array of size num_creatures * 2 where
        # scores[i][1] contains the creature representation and 
        # scores[i][0] 
        # contains its fitness function value.
        # Randomly select two creatures and return the index of the one 
        # with the lowest fitness function value (the best creature)
        number_creatures = len(scores)
        index1 = random.randint(0,number_creatures-1)
        index2 = random.randint(0,number_creatures-1)
        # check that index1 and index2 are different
        while index1 == index2:
            index2 = random.randint(0,number_creatures-1)
        
        if scores[index1][0] < scores[index2][0]:
            return index1
        else:
            return index2


    # create a random initial population
    population = [makerandomtree(opr=0.7,maxdepth=MAX_DEPTH) for _ in range(popsize)]
   
    # save original population
    #save_population_to_file(-1,population)

    for i in range(maxgen):
        scores = rankfunction(population)
        
        # print the score of the best creature
        # scores[i][0]: normalized score
        # scores[i][1]: creature
        # scores[i][2]: raw score
        print(scores[0][2])
        
        # the two best creatures go automatically in the next population
        newpop = [scores[0][1], scores[1][1]]

        while len(newpop) < popsize:
            # wouldn't it be better to do: 
            # 1) crossover
            # 2) mutation
            # 3) selection?

            #creature1_index = selectindex()
            creature1_index = tournament(scores)
            #creature2_index = selectindex()
            creature2_index = tournament(scores)

            p1,p2 = crossover(scores[creature1_index][1], scores[creature2_index][1], crossover_rate=breedingrate)
            new_p1 = mutate(p1, mutation_rate=mutationrate)
            new_p2 = mutate(p2, mutation_rate=mutationrate)
            newpop.append(new_p1)
            newpop.append(new_p2)

        population = newpop

        # dump the population in a temp file
        #save_population_to_file(i, population)

        # display the best individual every 10 iterations
        if (i!=0) and (i%10==0):
            print(scores[0][1].to_string())

    # print the score of the best creature
    print(scores[0][1].to_string())
    
    # Return the best creature
    return scores[0][1]


def getrankfunction(dataset):
    '''
    For now keep it unchanged.
    Later I may need to apply some scaling. 
    '''
    def rankfunction(population):
        scores = [[scorefunction(t,dataset), t, 0.0] for t in population]
        normalized_scores = normalize_fitness(scores)
        normalized_scores.sort()
        return normalized_scores
    return rankfunction


def getrankfunction_BIC(dataset):
    '''
    For now keep it unchanged.
    Later I may need to apply some scaling. 
    '''
    def rankfunction(population):
        scores = [[scorefunction_BIC(t,dataset), t, 0.0] for t in population]
        normalized_scores = normalize_fitness_BIC(scores)
        #normalized_scores.sort()
        normalized_scores.sort(key=lambda score: score[0])
        return normalized_scores
    return rankfunction


def getrankfunction_vectorized(dataset):
    '''
    Return a rank function using the vectorized score function and 
    the dataset passed as argument.
    '''
    dataset_np = np.array(dataset)
    points = dataset_np[:,0:3]
    normals = dataset_np[:,3:6]
    def rankfunction(population):
        scores = [[scorefunction_vectorized(t, points, normals), t, 0.0] for t in population]
        normalized_scores = normalize_fitness(scores)
        #normalized_scores.sort()
        normalized_scores.sort(key=lambda score: score[0])
        return normalized_scores
    return rankfunction


def getrankfunction_vectorized_BIC(dataset):
    '''
    Return a rank function using the vectorized score function and 
    the dataset passed as argument.
    '''
    dataset_np = np.array(dataset)
    points = dataset_np[:,0:3]
    normals = dataset_np[:,3:6]
    def rankfunction(population):
        scores = [[scorefunction_vectorized_BIC(t, points, normals), t, 0.0] for t in population]
        normalized_scores = normalize_fitness_BIC(scores)
        #normalized_scores.sort()
        normalized_scores.sort(key=lambda score: score[0])
        return normalized_scores
    return rankfunction


def read_pwn(filename):
    '''
    Load pwn data contained in filename.
    Return a list of point with normals (i.e. vector of dimension 6).
    '''

    f = open(filename)
    coordinates_normals = []

    # first read the number of points
    header = f.readline()
    # split according to spaces ' ' -> gives an array
    header_array = header.split() 
    number_points = int(header_array[0])

    # this is the array where we will store the coordinates
    # one coordinate is an array of dimension 3
    coordinates = []
    for i in range(number_points):
        # store one point
        point = []
        coord_line = f.readline()
        coord_array = coord_line.split()
        # store the x,y,z in point
        point.append(float(coord_array[0]))
        point.append(float(coord_array[1]))
        point.append(float(coord_array[2]))

        # add the point in the list of coordinates
        coordinates.append(point)

    # check that we have read enough points
    assert(len(coordinates)==number_points)


    # this is the array where we will store the normal vectors
    # one normal vector is an array of dimension 3
    normals = []
    for i in range(number_points):
        # store one normal vector
        normal = []
        normal_line = f.readline()
        normal_array = normal_line.split()
        # store the x,y,z coordinates in normal
        normal.append(float(normal_array[0]))
        normal.append(float(normal_array[1]))
        normal.append(float(normal_array[2]))

        # add the normal vector in the list of normals
        normals.append(normal)

    # check that we have read enouth normal vectors
    assert(len(normals)==number_points)


    # they should both be equal to the number of points
    assert(len(coordinates)==len(normals))


    # now add the 6-vector made of coord and normal coord to 
    # coordinates_normals
    for i in range(number_points):
        coord = coordinates[i]
        coord.extend(normals[i])
        coordinates_normals.append(coord)

    
    f.close()
    
    # return the list of coords and normals
    return coordinates_normals


def display(population):
    ''' Display each creature of a given population by printing its string
    representation.
    '''
    print('Size of the population: ')
    print(len(population))
    for creature in population:
        print(creature.to_string())
    print('-'*80)


def save_population_to_file(iteration, population, file_name='temp_population.txt'):
    '''
    Save the current population in a file for later inspection.
    '''
    with open(file_name, 'a') as f:
        # write each creature to a file
        f.write('Iteration: %d\n' % iteration)
        for creature in population:
            f.write(creature.to_string())
            f.write('\n')


def read_operations(file_name):
    '''
    Load the list operations that can be used as internal nodes
    to the evolved trees.
    
    The returned object will have to be passed 
    to makerandomtree() via evolve().

    Args:
        file_name: name of the file with the operations to be used
    '''

    # list of available operations
    available_operations = ['union','intersection','subtraction','negation']
    
    # currently the symbol 'operations_list' is already used globally
    op_list = []
    op_name_list = []

    with open(file_name) as f:
        for line in f:
            line = line.strip()
            # each line should contains an operation name
            if (len(line) == 0):
                continue

            op_name = line.split()[0]
            if not(op_name in available_operations):
                print('Unknown operation')
                continue
            op_name_list.append(op_name)

    # remove duplicates
    op_name_set = set(op_name_list)
    op_name_list = list(op_name_set)

    # create all the wrappers
    # note: currently the names unionw, intersectionw, negationw and 
    # subtractionw are already used
    union_w = fwrapper(lambda f: union(f[0],f[1]), 2, 'union')
    intersection_w = fwrapper(lambda f: intersection(f[0],f[1]), 2, 'intersection')
    negation_w = fwrapper(lambda f: negation(f[0]), 1, 'negation')
    subtraction_w = fwrapper(lambda f: subtraction(f[0], f[1]), 2, 'subtraction')

    # map operation names to operation wrappers
    operations_map = {'union': union_w, 'intersection': intersection_w, 'negation': negation_w, 'subtraction': subtraction_w}

    for op_name in op_name_list:
        op_list.append(operations_map[op_name])

    return op_list


def read_primitives(file_name):
    '''
    Load the list of primitives that can be used as leaves for
    the evolved trees.
    
    The returned object will have to be passed 
    to makerandomtree() via evolve().
    
    Args:
        file_name: name of the file with the primitives information
    '''


    # list of available primitive types:
    available_primitive_types = ['plane']

    # map available primitive type to number of parameters
    # TODO: move somewhere else (global variable? some
    # configuration variable? somewhere else?)
    # it should go with the geometric kernel.
    parameter_numbers_map = {'plane':3}

    # To collect the instantiated primitives
    primitives_list = []

    with open(file_name) as f:
        for line in f.readlines():
            # remove trailing '\n'
            line = line.strip()
            # A line contains at least a primitive type and a primitive 
            # name
            if len(line) < 2:
                continue
            # fields are separated by 1 space
            fields = line.split()

            primitive_type = fields[0]
            primitive_name = fields[1]

            # skip unknown primitives
            if not(primitive_type in available_primitive_types):
                print('Unknown primitive')
                continue

            # number of parameters depends on the primitive type
            parameters = []
            for i in range(parameter_numbers_map[primitive_type]):
                parameters.append(float(fields[2+i]))

            # instantiate the primitive
            primitive = None

            if primitive_type == 'plane':
                primitive = terminalnode(lambda p: plane(p, parameters), primitive_name)
            else:
                # unknown primitive_type
                print('Unknown primitive')
                pass

            primitives_list.append(primitive)

    return primitives_list


def read_xyzn(xyzn_filename):
    f = open(xyzn_filename)
    
    coords_and_normals = []
    for line in f:
        data = line.strip().split()
        if len(data)==0:
            # empty line - skip
            continue
        
        if len(data) != 6:
            raise Exception('Error while reading the file ' + xyzn_filename)

        pn = (float(data[0]), float(data[1]), float(data[2]), 
              float(data[3]), float(data[4]), float(data[5]))
        coords_and_normals.append(pn)
    
    f.close()
    return coords_and_normals


def compute_bounding_box(pointset):
    '''
    Compute and the bounding box of the point-set given as input. 
    The bounding-box is defined as a pair of two points. 
    The first point corresponds to the lower left corner of the box. 
    The second point corresponds to the upper right corner of the box. 
    '''
    max_float = sys.float_info.max
    min_float = -max_float
    left_corner = [max_float, max_float, max_float]
    right_corner = [min_float, min_float, min_float]
    
    # assume that pointset is a list of 6-tuple
    # elements 0-2 correspond to coordinates
    # elements 3-5 correspond to normals
    for pn in pointset:
        x, y, z, nx, ny, nz = pn
        coord = (x, y, z)
        normal = (nx, ny, nz)
        left_corner = [min(coord[0], left_corner[0]), 
                       min(coord[1], left_corner[1]), 
                       min(coord[2], left_corner[2])]

        right_corner = [max(coord[0], right_corner[0]), 
                        max(coord[1], right_corner[1]), 
                        max(coord[2], right_corner[2])]

    lc = tuple(left_corner)
    rc = tuple(right_corner)
    return (lc, rc)


def compute_bounding_box_length(pointset):
    '''
    Compute and return the length of the diagonal of the bounding box for 
    the point-set given as input.
    '''
    bbox = compute_bounding_box(pointset)
    left_corner, right_corner = bbox
    diag = (right_corner[0]-left_corner[0], right_corner[1]-left_corner[1],
            right_corner[2]-left_corner[2])
    return compute_length(diag)


def create_list_operations():
    unionw = fwrapper(lambda f: union(f[0],f[1]), 2, 'union')
    intersectionw = fwrapper(lambda f: intersection(f[0],f[1]), 2, 'intersection')
    negationw = fwrapper(lambda f: negation(f[0]), 1, 'negation')
    subtractionw = fwrapper(lambda f: subtraction(f[0], f[1]), 2, 'subtraction')

    list_operations = [unionw, intersectionw, negationw, subtractionw]
    return list_operations


def create_list_terminalnodes_WRONG(list_primitives):
    ''' Create a list of terminal nodes from a list of primitive shapes.'''

    list_terminalnodes = []
    count = 0
    for primitive in list_primitives:
        name = primitive.identifier() + str(count)
        tn = terminalnode(lambda p: primitive.signed_distance(p), name)
        list_terminalnodes.append(tn)
        count = count + 1

    return list_terminalnodes


def create_list_terminalnodes(list_primitives):
    ''' Create a list of terminal nodes from a list of primitive shapes.'''

    list_terminalnodes = []
    count = 0
    for i in range(len(list_primitives)):
        name = list_primitives[i].identifier() + str(count)
        tn = terminalnode(list_primitives[i].signed_distance, name)
        list_terminalnodes.append(tn)
        count = count + 1

    return list_terminalnodes


def save_creature_to_file(creature, filename):
    f = open(filename, "w")
    f.write(creature.to_string())
    f.close()


def save_primitives_list_to_file(tnodes, filename):
    '''
    Save the list of primitives names in a file. Names are separated by a comma.
    '''
    f = open(filename, "w")
    num_prim = len(tnodes)
    for i in range(num_prim-1):
        f.write(tnodes[i].name)
        f.write(',')

    f.write(tnodes[num_prim-1].name)
    f.write('\n')
        
    f.close()


# Test the GP with the 2torus.pwn dataset
# which corresponds to the fitted primitives in terminals_list
def test_main():
    ''' Test the GP with the 2torus.pwn dataset, which corresponds to the 
    fitted primitives in terminals_list
    '''
    dataset = read_pwn('data/2torus.pwn')
    # rf = getrankfunction(dataset)
    rf = getrankfunction_vectorized(dataset)
    evolve(50, rf, maxgen=50, mutationrate=0.1, breedingrate=0.5, pexp=0.8)


def main(fit_file, xyzn_file, expression_file="expression.txt", 
         primitives_file="list_primitives.txt", popsize=150, maxgen=3000, mutationrate=0.3, breedingrate=0.3):
    global g_list_terminalnodes
    global g_list_operations
    list_primitives = read_fit(fit_file)
    g_list_terminalnodes = create_list_terminalnodes(list_primitives)
    g_list_operations = create_list_operations()

    dataset = read_xyzn(xyzn_file)

    # set the model length that will be used by the objective function
    global g_model_length
    g_model_length = compute_bounding_box_length(dataset)


    
    rf = getrankfunction_vectorized(dataset)
    
    # Experiments with a different objective function
    #rf = getrankfunction_vectorized_BIC(dataset)
    # This did not work. When increasing the model complexity, 
    # the M makes the term M Log[N] much bigger than the improvement 
    # brought on the log likelihood by using a more complex model. 


    # originally I used:
    # mutationrate = 0.4
    # breedingrate = 0.5
    # I used 2000 iterations for most of the experiments
    # for the data-set body it does not seem sufficient so I am 
    # increasing it to 3000
    best_creature = evolve(popsize, rf, maxgen=maxgen, mutationrate=mutationrate, 
                           breedingrate=breedingrate, pexp=0.8)
    

    save_creature_to_file(best_creature, expression_file)
    save_primitives_list_to_file(g_list_terminalnodes, primitives_file)


#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
# Tests
# used to test the various functions and classes.
# Note: most of these tests require some data file and they will fail to 
# work without it.
# I have changed the directories and file names recently so most of them 
# will fail. I keep them for memory (to remember what I did)
#

def test_construct_and_display_trees():
    # simple tree with one node, two leaves
    tree_1 = node(unionw, [plane1,plane2])
    print(tree_1.to_string())

    # tree with two levels
    tree_2 = node(intersectionw, [tree_1,plane2])
    print(tree_2.to_string())

    # tree with an unary operation
    tree_3 = node(negationw, [tree_2])
    print(tree_3.to_string())


def test_create_random_tree():
    # default arguments
    tree_1 = makerandomtree()
    print(tree_1.to_string())

    # change arguments
    tree_2 = makerandomtree(maxdepth=10)
    print(tree_2.to_string())

    tree_3 = makerandomtree(maxdepth=10, opr=0.7)
    print(tree_3.to_string())

    tree_4 = makerandomtree(maxdepth=10, opr=0.1)
    print(tree_4.to_string())


def test_create_population():
    # population of 5; random tree created with default arguments
    print('population of 5; default arguments to makerandomtree()')
    population = [makerandomtree() for _ in range(5)]
    for creature in population:
        print(creature.to_string())

    # population of 10; random tree created with default arguments
    print('population of 10; default arguments to makerandomtree()')
    population = [makerandomtree() for _ in range(10)]
    for creature in population:
        print(creature.to_string())

    # population of 5; random tree with max depth of 10
    print('population of 5; max depth of 10 when creating the random trees')
    population = [makerandomtree(maxdepth=10) for _ in range(5)]
    for creature in population:
        print(creature.to_string())

    
'''
Call crossover() 10 times on two trees and look at the distribution
of the created creatures
'''
def test_crossover():
    tree_1 = makerandomtree(maxdepth=10)
    tree_2 = makerandomtree(maxdepth=10)
    print('Original trees')
    print(tree_1.to_string())
    print(tree_2.to_string())
    print(tree_1.compute_number_nodes())
    print(tree_2.compute_number_nodes())
    print('')

    print('crossover rate: 0.7')
    for i in range(10):
        new_tree1,new_tree2 = crossover(tree_1, tree_2, crossover_rate=0.7)
        print('new tree 1:')
        print(new_tree1.to_string())
        print('new tree 2')
        print(new_tree2.to_string())

    print('')
    print('crossover rate: 0.4')
    for i in range(10):
        new_tree1,new_tree2 = crossover(tree_1, tree_2, crossover_rate=0.4)
        print('new tree 1:')
        print(new_tree1.to_string())
        print('new tree 2')
        print(new_tree2.to_string())


def test_mutate():
    '''
    Call mutate() a number of times and look at the distribution
    of the mutated creatures.
    '''

    tree_1 = makerandomtree(maxdepth=10)
    print('Original tree')
    print(tree_1.to_string())
    print('')

    for i in range(10):
        result = mutate(tree_1,mutation_rate=0.1)
        print(result.to_string())


def test_read_primitives():
    file_name = 'data/2torus_primitives.txt'
    prim_list = read_primitives(file_name)
    
    # look at the content of prim_list
    print('Number of primitives: %d' % (len(prim_list)))
    print('Primitives: ')
    for prim in prim_list:
        print(prim.to_string())


def test_read_operations():
    file_name = 'data/2torus_operations.txt'
    op_list = read_operations(file_name)

    # look at the content of op_list
    print('Number of operations: %d' % (len(op_list)))
    print('Operations: ')
    for op in op_list:
        print(op.name)


def test_eval_terminalnodes_vectorized():
    xyzn_file = '../data/2torus.xyzn'
    fit_file = '../data/2torus.fit'
    list_primitives = read_fit(fit_file)
    list_terminalnodes = create_list_terminalnodes(list_primitives)

    dataset = read_xyzn(xyzn_file)
    
    dataset_np = np.array(dataset)
    points = dataset_np[:,0:3]
    normals = dataset_np[:,3:6]
    
    scores = [[scorefunction_vectorized(t, points, normals), t] for t in list_terminalnodes]
    # normalized_scores = normalize_fitness(scores)
    # normalized_scores.sort()
    
    print('Score of each primitive node')
    for s in scores:
        print(s[1].to_string())
        print(s[0])


def test_eval_terminalnodes():
    xyzn_file = '../data/2torus.xyzn'
    fit_file = '../data/2torus.fit'
    list_primitives = read_fit(fit_file)
    list_terminalnodes = create_list_terminalnodes(list_primitives)

    dataset = read_xyzn(xyzn_file)
    
    scores = [[scorefunction(t, dataset), t] for t in list_terminalnodes]
    # normalized_scores = normalize_fitness(scores)
    # normalized_scores.sort()
    
    print('Score of each primitive node')
    for s in scores:
        print(s[1].to_string())
        print(s[0])


def test_compute_bounding_box_length():
    xyzn_filename = '../data/fandisk.xyzn'
    ps = read_xyzn(xyzn_filename)
    bbox = compute_bounding_box(ps)
    l = compute_bounding_box_length(ps)
    print('bounding box: ')
    print(bbox)
    print('length: ')
    print(l)



# ----------------------------------------------------------------------
# ----------------------------------------------------------------------


def usage(progname):
    print('Usage:')
    print(progname + ' primitives.fit point_set.xyzn best_creature.txt primitives.txt pop_size max_iterations muration_rate crossover_rate\n')


# Main:
# Simple way to run and test
if __name__ == "__main__":
    # test_main()
    args = sys.argv
    progname = args[0]
    args = args[1:len(args)]
    if len(args) != 8:
        usage(progname)
        sys.exit(1)

    main(args[0], args[1], args[2], args[3], popsize=int(args[4]), maxgen=int(args[5]), mutationrate=float(args[6]), breedingrate=float(args[7]))
