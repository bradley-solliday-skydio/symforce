#include <Eigen/Dense>

#include <sym/rot3.h>
#include <sym/util/epsilon.h>

namespace sym {

template <typename Scalar>
struct PreintegratedImuMeasurements {
  using Vector3 = Eigen::Matrix<Scalar, 3, 1>;
  using Matrix3 = Eigen::Matrix<Scalar, 3, 3>;
  using Matrix9 = Eigen::Matrix<Scalar, 9, 9>;

  sym::Rot3<Scalar> DR;
  Vector3 Dv;
  Vector3 Dp;
  Matrix9 state_covariance;
  Matrix3 DR_D_gyro_bias;
  Matrix3 Dv_D_gyro_bias;
  Matrix3 Dv_D_accel_bias;
  Matrix3 Dp_D_gyro_bias;
  Matrix3 Dp_D_accel_bias;
  Vector3 gyro_bias;
  Vector3 accel_bias;

  double integrated_dt;

  PreintegratedImuMeasurements(const Vector3& accel_bias, const Vector3& gyro_bias);

  // NOTE: Aaron thinks update should be part of the Factor
  void IntegrateMeasurement(const Vector3& measured_accel, const Vector3& measured_gyro,
                            const Matrix3& accel_cov, const Matrix3& gyro_cov, const Scalar dt,
                            const Scalar epsilon = kDefaultEpsilon<Scalar>);
};


}  // namespace sym

extern template struct sym::PreintegratedImuMeasurements<double>;
extern template struct sym::PreintegratedImuMeasurements<float>;
