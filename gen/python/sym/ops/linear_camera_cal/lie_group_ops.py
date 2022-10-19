# -----------------------------------------------------------------------------
# This file was autogenerated by symforce from template:
#     ops/CLASS/lie_group_ops.py.jinja
# Do NOT modify by hand.
# -----------------------------------------------------------------------------

import math
import typing as T

import numpy

import sym  # pylint: disable=unused-import


class LieGroupOps(object):
    """
    Python LieGroupOps implementation for <class 'symforce.cam.linear_camera_cal.LinearCameraCal'>.
    """

    @staticmethod
    def from_tangent(vec, epsilon):
        # type: (T.Union[T.Sequence[float], numpy.ndarray], float) -> sym.LinearCameraCal

        # Total ops: 0

        # Input arrays
        if not isinstance(vec, numpy.ndarray):
            if len(vec) != 4:
                raise IndexError(
                    "vec is expected to have length 4; instead had length {}".format(len(vec))
                )
            vec = numpy.array(vec).reshape((4, 1))
        elif vec.shape == (4,):
            vec = vec.reshape((4, 1))
        elif vec.shape != (4, 1):
            raise IndexError(
                "vec is expected to have shape (4, 1) or (4,); instead had shape {}".format(
                    vec.shape
                )
            )

        # Intermediate terms (0)

        # Output terms
        _res = [0.0] * 4
        _res[0] = vec[0, 0]
        _res[1] = vec[1, 0]
        _res[2] = vec[2, 0]
        _res[3] = vec[3, 0]
        return sym.LinearCameraCal.from_storage(_res)

    @staticmethod
    def to_tangent(a, epsilon):
        # type: (sym.LinearCameraCal, float) -> numpy.ndarray

        # Total ops: 0

        # Input arrays
        _a = a.data

        # Intermediate terms (0)

        # Output terms
        _res = numpy.zeros((4, 1))
        _res[0, 0] = _a[0]
        _res[1, 0] = _a[1]
        _res[2, 0] = _a[2]
        _res[3, 0] = _a[3]
        return _res

    @staticmethod
    def retract(a, vec, epsilon):
        # type: (sym.LinearCameraCal, T.Union[T.Sequence[float], numpy.ndarray], float) -> sym.LinearCameraCal

        # Total ops: 4

        # Input arrays
        _a = a.data
        if not isinstance(vec, numpy.ndarray):
            if len(vec) != 4:
                raise IndexError(
                    "vec is expected to have length 4; instead had length {}".format(len(vec))
                )
            vec = numpy.array(vec).reshape((4, 1))
        elif vec.shape == (4,):
            vec = vec.reshape((4, 1))
        elif vec.shape != (4, 1):
            raise IndexError(
                "vec is expected to have shape (4, 1) or (4,); instead had shape {}".format(
                    vec.shape
                )
            )

        # Intermediate terms (0)

        # Output terms
        _res = [0.0] * 4
        _res[0] = _a[0] + vec[0, 0]
        _res[1] = _a[1] + vec[1, 0]
        _res[2] = _a[2] + vec[2, 0]
        _res[3] = _a[3] + vec[3, 0]
        return sym.LinearCameraCal.from_storage(_res)

    @staticmethod
    def local_coordinates(a, b, epsilon):
        # type: (sym.LinearCameraCal, sym.LinearCameraCal, float) -> numpy.ndarray

        # Total ops: 4

        # Input arrays
        _a = a.data
        _b = b.data

        # Intermediate terms (0)

        # Output terms
        _res = numpy.zeros((4, 1))
        _res[0, 0] = -_a[0] + _b[0]
        _res[1, 0] = -_a[1] + _b[1]
        _res[2, 0] = -_a[2] + _b[2]
        _res[3, 0] = -_a[3] + _b[3]
        return _res
