#include "./preintegrated_imu_measurements.h"

#include <sym/factors/imu_manifold_preintegration_update.h>

namespace sym {

template <typename Scalar>
PreintegratedImuMeasurements<Scalar>::PreintegratedImuMeasurements(const Vector3& accel_bias,
                                                                   const Vector3& gyro_bias)
    : DR(),
      Dv{Vector3::Zero()},
      Dp{Vector3::Zero()},
      state_covariance{Matrix9::Zero()},  // covariance of [DR.ToTangent(), Dv, Dp]
      // Consider having this be just a 9x6 matrix instead
      DR_D_gyro_bias{Matrix3::Zero()},
      Dv_D_gyro_bias{Matrix3::Zero()},
      Dv_D_accel_bias{Matrix3::Zero()},
      Dp_D_gyro_bias{Matrix3::Zero()},
      Dp_D_accel_bias{Matrix3::Zero()},
      gyro_bias{gyro_bias},
      accel_bias{accel_bias},
      integrated_dt{0.0} {}

template <typename Scalar>
void PreintegratedImuMeasurements<Scalar>::IntegrateMeasurement(
    const Vector3& measured_accel, const Vector3& measured_gyro, const Matrix3& accel_cov,
    const Matrix3& gyro_cov, const Scalar dt, const Scalar epsilon) {
  Rot3<Scalar> new_DR;
  Vector3 new_Dv;
  Vector3 new_Dp;
  Matrix9 new_covariance;
  Matrix3 new_DR_D_gyro_bias;
  Matrix3 new_Dv_D_gyro_bias;
  Matrix3 new_Dv_D_accel_bias;
  Matrix3 new_Dp_D_gyro_bias;
  Matrix3 new_Dp_D_accel_bias;
  ImuManifoldPreintegrationUpdate<Scalar>(
      DR, Dv, Dp, state_covariance, DR_D_gyro_bias, Dv_D_gyro_bias, Dv_D_accel_bias, Dp_D_gyro_bias,
      Dp_D_accel_bias, gyro_bias, accel_bias, gyro_cov, accel_cov, measured_gyro, measured_accel, dt,
      epsilon, &new_DR, &new_Dv, &new_Dp, &new_covariance, &new_DR_D_gyro_bias, &new_Dv_D_gyro_bias,
      &new_Dv_D_accel_bias, &new_Dp_D_gyro_bias, &new_Dp_D_accel_bias);

  // No point in moving because the data lives in this class, i.e., a copy can't be avoided
  DR = new_DR;
  Dv = new_Dv;
  Dp = new_Dp;
  state_covariance = new_covariance;
  DR_D_gyro_bias = new_DR_D_gyro_bias;
  Dv_D_gyro_bias = new_Dv_D_gyro_bias;
  Dv_D_accel_bias = new_Dv_D_accel_bias;
  Dp_D_gyro_bias = new_Dp_D_gyro_bias;
  Dp_D_accel_bias = new_Dp_D_accel_bias;

  integrated_dt += dt;
}

}  // namespace sym

template struct sym::PreintegratedImuMeasurements<double>;
template struct sym::PreintegratedImuMeasurements<float>;
