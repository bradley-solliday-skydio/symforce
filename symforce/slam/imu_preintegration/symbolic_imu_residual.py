# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

import symforce.symbolic as sf
from symforce import typing as T

# Contains symbolic implementation of imu tangent residual


#def imu_residual(
#    state_i: sf.V9,
#    state_j: sf.V9,
#    accel_bias: sf.V3,
#    gyro_bias: sf.V3,
#    # More arguments that need to be passed in, but that will be handled by 
#
#)

# below pasted for refrence
def imu_preintegration_update(
    # Initial state
    state: sf.V9,
    state_cov: sf.M99,
    state_D_accel_bias: sf.M93,
    state_D_gyro_bias: sf.M93,
    # Biases and noise model
    accel_bias: sf.V3,
    gyro_bias: sf.V3,
    accel_cov: sf.M33,
    gyro_cov: sf.M33,
    # Measurement
    accel: sf.V3,
    gyro: sf.V3,
    dt: T.Scalar,
    # Singularity handling
    epsilon: T.Scalar,
) -> T.Tuple[sf.V9, sf.Matrix, sf.Matrix, sf.Matrix]:
    pass