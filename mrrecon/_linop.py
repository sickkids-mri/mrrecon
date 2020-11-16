"""Low level implementation of linear operations.

CuPy user-defined kernels are defined here.

Related documentation:
https://docs.cupy.dev/en/stable/tutorial/kernel.html

The CUDA finite difference operations are implemented through CuPy
`ElementwiseKernel`s, with raw argument specifiers. The kernel code is written
in C++, except the data types of the inputs and outputs use NumPy/CuPy data
type names, e.g. float32, int64, etc. The code inside the kernel uses C++ data
type names. The single capital letters in front of the variable names are data
type placeholders, whose values are determined at runtime. `i` is a special
variable that indicates the index of the loop. The finite difference kernels
all have the same steps: 1) Convert the linear index to multidimensional
indices. 2) Shift the index corresponding to the dimension that finite
differences are calculated along. 3) Handle indices at the boundaries.
4) Convert back to linear index. 5) Calculate difference.
"""
import importlib
if importlib.util.find_spec("cupy") is not None:
    import cupy as cp


def _finite_difference_1d(x, out, shift, axis):
    """Computes the finite difference of a 1D array.

    Args:
        x (CuPy array): Input 1D array.
        out (CuPy array): Output array. Must have the same shape as `x`.
        shift (int): Number of elements to shift the subtracted array. Should
            have a value of 1 or -1.
        axis (int): The axis along which differences are calculated. Should
            always have a value of 0 for the 1D case.
    """
    na, = x.shape

    inputs = 'raw T x, int64 shift, int64 na'
    outputs = 'T out'

    calc_indices = '''
        long long a = i;
        '''

    shift_a = '''
        a = a - shift;

        if (a == -1) a = na - 1;
        else if (a == na) a = 0;
        '''

    index_and_subtract = '''
        long long j = a;
        out = x[i] - x[j];
        '''

    code = calc_indices + shift_a + index_and_subtract
    name = 'finite_difference_1d_0'

    kernel = cp.ElementwiseKernel(inputs, outputs, code, name)
    kernel(x, shift, na, out)


def _finite_difference_2d(x, out, shift, axis):
    """Computes the finite difference of a 2D array.

    Args:
        x (CuPy array): Input 2D array.
        out (CuPy array): Output array. Must have the same shape as `x`.
        shift (int): Number of elements to shift the subtracted array. Should
            have a value of 1 or -1.
        axis (int): The axis along which differences are calculated.
    """
    nb, na = x.shape

    inputs = 'raw T x, int64 shift, int64 nb, int64 na'
    outputs = 'T out'

    calc_indices = '''
        long long b = i / na;
        long long a = i % na;
        '''

    shift_b = '''
        b = b - shift;

        if (b == -1) b = nb - 1;
        else if (b == nb) b = 0;
        '''

    shift_a = '''
        a = a - shift;

        if (a == -1) a = na - 1;
        else if (a == na) a = 0;
        '''

    index_and_subtract = '''
        long long j = b * na + a;
        out = x[i] - x[j];
        '''

    if axis == 0:
        code = calc_indices + shift_b + index_and_subtract
        name = 'finite_difference_2d_0'
    elif axis == 1:
        code = calc_indices + shift_a + index_and_subtract
        name = 'finite_difference_2d_1'

    kernel = cp.ElementwiseKernel(inputs, outputs, code, name)
    kernel(x, shift, nb, na, out)


def _finite_difference_3d(x, out, shift, axis):
    """Computes the finite difference of a 3D array.

    Args:
        x (CuPy array): Input 3D array.
        out (CuPy array): Output array. Must have the same shape as `x`.
        shift (int): Number of elements to shift the subtracted array. Should
            have a value of 1 or -1.
        axis (int): The axis along which differences are calculated.
    """
    nc, nb, na = x.shape

    inputs = 'raw T x, int64 shift, int64 nc, int64 nb, int64 na'
    outputs = 'T out'

    precalculations = '''
        long long nb_na = nb * na;
        '''

    calc_indices = '''
        long long c = i / nb_na;
        long long b = i % nb_na / na;
        long long a = i % nb_na % na;
        '''

    shift_c = '''
        c = c - shift;

        if (c == -1) c = nc - 1;
        else if (c == nc) c = 0;
        '''

    shift_b = '''
        b = b - shift;

        if (b == -1) b = nb - 1;
        else if (b == nb) b = 0;
        '''

    shift_a = '''
        a = a - shift;

        if (a == -1) a = na - 1;
        else if (a == na) a = 0;
        '''

    index_and_subtract = '''
        long long j = c * nb_na + b * na + a;
        out = x[i] - x[j];
        '''

    if axis == 0:
        code = precalculations + calc_indices + shift_c + index_and_subtract
        name = 'finite_difference_3d_0'
    elif axis == 1:
        code = precalculations + calc_indices + shift_b + index_and_subtract
        name = 'finite_difference_3d_1'
    elif axis == 2:
        code = precalculations + calc_indices + shift_a + index_and_subtract
        name = 'finite_difference_3d_2'

    kernel = cp.ElementwiseKernel(inputs, outputs, code, name)
    kernel(x, shift, nc, nb, na, out)


def _finite_difference_4d(x, out, shift, axis):
    """Computes the finite difference of a 4D array.

    Args:
        x (CuPy array): Input 4D array.
        out (CuPy array): Output array. Must have the same shape as `x`.
        shift (int): Number of elements to shift the subtracted array. Should
            have a value of 1 or -1.
        axis (int): The axis along which differences are calculated.
    """
    nd, nc, nb, na = x.shape

    inputs = 'raw T x, int64 shift, int64 nd, int64 nc, int64 nb, int64 na'
    outputs = 'T out'

    precalculations = '''
        long long nb_na = nb * na;
        long long nc_nb_na = nc * nb_na;
        '''

    calc_indices = '''
        long long d = i / nc_nb_na;
        long long c = i % nc_nb_na / nb_na;
        long long b = i % nc_nb_na % nb_na / na;
        long long a = i % nc_nb_na % nb_na % na;
        '''

    shift_d = '''
        d = d - shift;

        if (d == -1) d = nd - 1;
        else if (d == nd) d = 0;
        '''

    shift_c = '''
        c = c - shift;

        if (c == -1) c = nc - 1;
        else if (c == nc) c = 0;
        '''

    shift_b = '''
        b = b - shift;

        if (b == -1) b = nb - 1;
        else if (b == nb) b = 0;
        '''

    shift_a = '''
        a = a - shift;

        if (a == -1) a = na - 1;
        else if (a == na) a = 0;
        '''

    index_and_subtract = '''
        long long j = d * nc_nb_na + c * nb_na + b * na + a;
        out = x[i] - x[j];
        '''

    if axis == 0:
        code = precalculations + calc_indices + shift_d + index_and_subtract
        name = 'finite_difference_4d_0'
    elif axis == 1:
        code = precalculations + calc_indices + shift_c + index_and_subtract
        name = 'finite_difference_4d_1'
    elif axis == 2:
        code = precalculations + calc_indices + shift_b + index_and_subtract
        name = 'finite_difference_4d_2'
    elif axis == 3:
        code = precalculations + calc_indices + shift_a + index_and_subtract
        name = 'finite_difference_4d_3'

    kernel = cp.ElementwiseKernel(inputs, outputs, code, name)
    kernel(x, shift, nd, nc, nb, na, out)


def _finite_difference_5d(x, out, shift, axis):
    """Computes the finite difference of a 5D array.

    Args:
        x (CuPy array): Input 5D array.
        out (CuPy array): Output array. Must have the same shape as `x`.
        shift (int): Number of elements to shift the subtracted array. Should
            have a value of 1 or -1.
        axis (int): The axis along which differences are calculated.
    """
    ne, nd, nc, nb, na = x.shape

    inputs = 'raw T x, int64 shift, int64 ne, int64 nd, int64 nc, int64 nb, int64 na'  # noqa
    outputs = 'T out'

    precalculations = '''
        long long nb_na = nb * na;
        long long nc_nb_na = nc * nb_na;
        long long nd_nc_nb_na = nd * nc_nb_na;
        '''

    calc_indices = '''
        long long e = i / nd_nc_nb_na;
        long long d = i % nd_nc_nb_na / nc_nb_na;
        long long c = i % nd_nc_nb_na % nc_nb_na / nb_na;
        long long b = i % nd_nc_nb_na % nc_nb_na % nb_na / na;
        long long a = i % nd_nc_nb_na % nc_nb_na % nb_na % na;
        '''

    shift_e = '''
        e = e - shift;

        if (e == -1) e = ne - 1;
        else if (e == ne) e = 0;
        '''

    shift_d = '''
        d = d - shift;

        if (d == -1) d = nd - 1;
        else if (d == nd) d = 0;
        '''

    shift_c = '''
        c = c - shift;

        if (c == -1) c = nc - 1;
        else if (c == nc) c = 0;
        '''

    shift_b = '''
        b = b - shift;

        if (b == -1) b = nb - 1;
        else if (b == nb) b = 0;
        '''

    shift_a = '''
        a = a - shift;

        if (a == -1) a = na - 1;
        else if (a == na) a = 0;
        '''

    index_and_subtract = '''
        long long j = e * nd_nc_nb_na + d * nc_nb_na + c * nb_na + b * na + a;
        out = x[i] - x[j];
        '''

    if axis == 0:
        code = precalculations + calc_indices + shift_e + index_and_subtract
        name = 'finite_difference_5d_0'
    elif axis == 1:
        code = precalculations + calc_indices + shift_d + index_and_subtract
        name = 'finite_difference_5d_1'
    elif axis == 2:
        code = precalculations + calc_indices + shift_c + index_and_subtract
        name = 'finite_difference_5d_2'
    elif axis == 3:
        code = precalculations + calc_indices + shift_b + index_and_subtract
        name = 'finite_difference_5d_3'
    elif axis == 4:
        code = precalculations + calc_indices + shift_a + index_and_subtract
        name = 'finite_difference_5d_4'

    kernel = cp.ElementwiseKernel(inputs, outputs, code, name)
    kernel(x, shift, ne, nd, nc, nb, na, out)


_finite_difference = {}
_finite_difference[1] = _finite_difference_1d
_finite_difference[2] = _finite_difference_2d
_finite_difference[3] = _finite_difference_3d
_finite_difference[4] = _finite_difference_4d
_finite_difference[5] = _finite_difference_5d
