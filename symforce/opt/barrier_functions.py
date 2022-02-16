# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

from symforce import sympy as sm
from symforce import typing as T


def symmetric_power_barrier(
    x: T.Scalar,
    x_nominal: T.Scalar,
    error_nominal: T.Scalar,
    dist_zero_to_nominal: T.Scalar,
    power: T.Scalar,
) -> T.Scalar:
    """
    A symmetric barrier cenetered around x = 0, meaning the error at -x is equal to the error at x.
    The barrier passes through the points (x_nominal, error_nominal) and
    (x_nominal - dist_zero_to_nominal, 0) with a curve of the form x^power. The parameterization
    of the barrier by these variables is convenient because it allows setting a constant
    penalty for a nominal point, then adjusting the `width` and `steepness` of the curve
    independently. For example, the barrier with power = 1 will look like:

                 **              |              **
                  **             |             ** - (x_nominal, error_nominal) is a fixed point
                   **            |            **
                    **           |           **   <- x^power is the shape of the curve
                     **          |          **
                      **         |         **
             ----------*********************---------
                                 |         |<-->| dist_zero_to_nominal is the distance from
                                                  x_nominal to the point at which the error is zero

    Note that when applying the barrier function to a residual used in a least-squares problem, a
    power = 1 will lead to a quadratic cost in the optimization problem because the cost
    equals 1/2 * residual^2. For example:

    Cost (1/2 * residual^2) when the residual is a symmetric barrier with power = 1 (shown above):

               *                |                *
               **               |               ** - (x_nominal, 1/2 * error_nominal^2)
                *               |               *
                **              |              ** <- x^(2*power) is the shape of the cost curve
                 ***            |            ***
                   ***          |          ***
             ---------*********************---------
                                |         |<-->| dist_zero_to_nominal

    Args:
        x: The point at which we want to evaluate the barrier function.
        x_nominal: x-value of the point at which the error is equal to error_nominal.
        error_nominal: Error returned when x equals x_nominal.
        dist_zero_to_nominal: Distance from x_nominal to the closest point at which the error is
            zero. Note that dist_zero_to_nominal must be less than x_nominal and greater than zero.
        power: The power used to describe the curve of the error tails.
    """
    x_zero_error = x_nominal - dist_zero_to_nominal
    return error_nominal * sm.Pow(
        sm.Max(0, sm.Abs(x) - x_zero_error) / dist_zero_to_nominal, power,
    )


def min_max_power_barrier(
    x: T.Scalar,
    x_nominal_lower: T.Scalar,
    x_nominal_upper: T.Scalar,
    error_nominal: T.Scalar,
    dist_zero_to_nominal: T.Scalar,
    power: T.Scalar,
) -> T.Scalar:
    """
    A symmetric barrier centered between x_nominal_lower and x_nominal_upper. See
    symmetric_power_barrier for a detailed description of the barrier function.
    As an example, the barrier with power = 1 will look like:

                                     **          |              **
                                      **         |             **
    (x_nominal_lower, error_nominal) - **        |            ** - (x_nominal_upper, error_nominal)
                                        **       |           **
                                         **      |          ** <- x^power is the shape of the curve
                                          **     |         **
                                 ----------*****************---------
                  dist_zero_to_nominal |<->|     |         |<->| dist_zero_to_nominal

    Args:
        x: The point at which we want to evaluate the barrier function.
        x_nominal_lower: x-value of the point at which the error is equal to error_nominal on
            the left-hand side of the barrier function.
        x_nominal_upper: x-value of the point at which the error is equal to error_nominal on
            the right-hand side of the barrier function.
        error_nominal: Error returned when x equals x_nominal_lower or x_nominal_upper.
        dist_zero_to_nominal: The distance from either of the x_nominal points to the region of
            zero error. Must be less than half the distance between x_nominal_lower and
            x_nominal_upper, and must be greater than zero.
        power: The power used to describe the curve of the error tails. Note that
            when applying the barrier function to a residual used in a least-squares problem,
            a power = 1 will lead to a quadratic cost in the optimization problem.
    """
    center = (x_nominal_lower + x_nominal_upper) / 2
    x_shifted = x - center
    x_nominal_shifted = x_nominal_upper - center
    return symmetric_power_barrier(
        x=x_shifted,
        x_nominal=x_nominal_shifted,
        error_nominal=error_nominal,
        dist_zero_to_nominal=dist_zero_to_nominal,
        power=power,
    )


def min_max_linear_barrier(
    x: T.Scalar,
    x_nominal_lower: T.Scalar,
    x_nominal_upper: T.Scalar,
    error_nominal: T.Scalar,
    dist_zero_to_nominal: T.Scalar,
) -> T.Scalar:
    """
    Applies "min_max_power_barrier" with power = 1. When applied to a residual of a least-squares
    problem, this produces a quadratic cost in the optimization problem because
    cost = 1/2 * residual^2. See "min_max_power_barrier" for more details.
    """
    return min_max_power_barrier(
        x=x,
        x_nominal_lower=x_nominal_lower,
        x_nominal_upper=x_nominal_upper,
        error_nominal=error_nominal,
        dist_zero_to_nominal=dist_zero_to_nominal,
        power=1,
    )