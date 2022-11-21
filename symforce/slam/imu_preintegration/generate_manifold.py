# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

from pathlib import Path

from symforce import codegen
from symforce import logger
from symforce import typing as T
from symforce.slam.imu_preintegration.manifold_symbolic_pim import imu_manifold_preintegration_update
from symforce.slam.imu_preintegration.manifold_symbolic_pim import internal_imu_residual


def generate_imu_manifold_preintegration(
    config: codegen.CodegenConfig, output_dir: T.Openable
) -> Path:
    """
    Generate the IMU preintegration update function.
    """
    cg_update = codegen.Codegen.function(
        imu_manifold_preintegration_update,
        config=config,
        output_names=[
            "new_DR",
            "new_Dv",
            "new_Dp",
            "new_covarinace",
            "new_DR_D_gyro_bias",
            "new_Dp_D_gyro_bias",
            "new_Dp_D_accel_bias",
            "new_Dv_D_gyro_bias",
            "new_Dv_D_accel_bias",
        ],
    )
    cg_update.generate_function(output_dir=output_dir, skip_directory_nesting=True)

    cg_residual = codegen.Codegen.function(
        internal_imu_residual,
        config=config,
    ).with_linearization(which_args=["rot_i", "vel_i", "pos_i", "rot_j", "vel_j", "pos_j", "gyro_bias_j", "accel_bias_j"])
    cg_residual.generate_function(output_dir=output_dir, skip_directory_nesting=True)


    return Path(output_dir)
