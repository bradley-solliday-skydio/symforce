#include "./imu_factor.h"

#include <sym/factors/internal_imu_factor.h>

namespace sym {

template <typename Scalar>
ImuFactor<Scalar>::ImuFactor(const PreintegratedImuMeasurements<Scalar>& pim)
    : pim_{pim}, sqrt_info_{pim.state_covariance_.inverse().llt().matrixU()} {}

template <typename Scalar>
void ImuFactor<Scalar>::operator()(
    const sym::Pose3<Scalar>& pose_i, const Eigen::Matrix<Scalar, 3, 1>& vel_i,
    const sym::Pose3<Scalar>& pose_j, const Eigen::Matrix<Scalar, 3, 1>& vel_j,
    const Eigen::Matrix<Scalar, 3, 1>& gyro_bias_j, const Eigen::Matrix<Scalar, 3, 1>& accel_bias_j,
    const Eigen::Matrix<Scalar, 3, 1>& gravity, const Scalar epsilon,
    Eigen::Matrix<Scalar, 9, 1>* const res, Eigen::Matrix<Scalar, 9, 24>* const jacobian,
    Eigen::Matrix<Scalar, 24, 24>* const hessian, Eigen::Matrix<Scalar, 24, 1>* const rhs) const {
  InternalImuFactor(pose_i.rotation(), vel_i, pose_i.position(), pose_j.rotation(), vel_j,
                    pose_j.position(), gyro_bias_j, accel_bias_j, pim_.DR_, pim_.Dv_, pim_.Dp_,
                    sqrt_info_, pim_.DR_D_gyro_bias_, pim_.Dp_D_accel_bias_, pim_.Dp_D_gyro_bias_,
                    pim_.Dv_D_accel_bias_, pim_.Dv_D_gyro_bias_, pim_.accel_bias_, pim_.gyro_bias_,
                    gravity, pim_.integrated_dt_, epsilon, res, jacobian, hessian, rhs);
}

}  // namespace sym

extern template class sym::ImuFactor<double>;
extern template class sym::ImuFactor<float>;
