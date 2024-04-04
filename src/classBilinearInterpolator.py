"""
Bilinear interpolation 
https://github.com/Dorian210/interpylate/
"""

import warnings 

class BiLinearRegularGridInterpolator:
    """
    Bi-linear grid interpolator : interpolate a 2D array between the indices using a bi-linear method :
    F(x, y) = a + bx + cy + dxy
    """
    def __init__(self):
        """
        Create the interpolator.
        """
        pass
    
    def _get_corners(self, image, i, j):
        i0, j0, i1, j1 = i, j, i + 1, j + 1
        l = image[i0, j0]
        m = image[i1, j0]
        n = image[i0, j1]
        o = image[i1, j1]
        return (l, m, n, o)

    def _make_coefs(self, image, i, j):
        l, m, n, o = self._get_corners(image, i, j)
        a = l
        b = m - a
        c = n - a
        d = o - a - b - c
        return [a, b, c, d]

    def _get_inds_coords_axis(self, continuous_inds_axis, axis_size):
        inds = continuous_inds_axis.astype('int')
        mask_negative = inds<0
        inds[mask_negative] = 0
        max_ind = axis_size - 2
        mask_too_large = inds>max_ind
        inds[mask_too_large] = max_ind
        outside_axis = mask_negative.any() or mask_too_large.any()
        coords = continuous_inds_axis - inds
        return (inds, coords, outside_axis)

    def _get_inds_coords(self, shape, continuous_inds):
        inds = [None]*2
        coords = [None]*2
        outside = False
        for axis in range(2):
            (inds[axis], coords[axis], outside_axis) = self._get_inds_coords_axis(continuous_inds[axis], shape[axis])
            outside = outside_axis or outside
        if outside:
            warnings.warn("Interpolate outside of the array !")
        return (inds, coords)

    def evaluate(self, image, continuous_inds):
        """
        Evaluate the interpolation at the coordinates given as input.

        Parameters
        ----------
        image : numpy.ndarray
            The array to interpolate.
        continuous_inds : numpy.ndarray of float
            Coordinates where the interpolation is computed.
            It's shape should be (2, n).

        Returns
        -------
        evaluated : numpy.ndarray of float
            Interpolated values at the coordinates given.
            If ``continuous_inds`` is of shape (2, n), the output will be of shape (n,).
        """
        ([i, j], [x, y]) = self._get_inds_coords(image.shape, continuous_inds)
        [a, b, c, d] = self._make_coefs(image, i, j)
        evaluated = (a + b*x +c*y + d*x*y)
        return evaluated

    def grad(self, image, continuous_inds, evaluate_too=False):
        """
        Compute the gradient of the interpolated array.

        Parameters
        ----------
        image : numpy.ndarray
            The array to interpolate.
        continuous_inds : numpy.ndarray of float
            Coordinates where the gradient is computed.
            It's shape should be (2, n).
        evaluate_too : bool, optional
            Set to True to evaluate on these points too, by default False

        Returns
        -------
        [grad_x, grad_y] : list of numpy.ndarray of float
            The derivative of the interpolated array in each axis's direction.
            If ``continuous_inds`` is of shape (2, n), each of the output will be of shape (n,).
        """
        ([i, j], [x, y]) = self._get_inds_coords(image.shape, continuous_inds)
        [a, b, c, d] = self._make_coefs(image, i, j)
        grad_x = (b + d*y)
        grad_y = (c + d*x)
        if evaluate_too:
            evaluated = (a + b*x +c*y + d*x*y)
            return [grad_x, grad_y], evaluated
        return [grad_x, grad_y]
