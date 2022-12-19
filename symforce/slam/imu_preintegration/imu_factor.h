#include <Eigen/Dense>

#include <sym/pose3.h>

#include "./preintegrated_imu_measurements.h"

namespace sym {

template <typename Scalar>
class ImuFactor {
 private:
  const PreintegratedImuMeasurements<Scalar> pim_;
  const Eigen::Matrix<Scalar, 9, 9> sqrt_info_;

 public:
  // NOTE: By passing passing the pim in already formed, a copy is necessary. Can't even move it
  // because all the data is inlined in the pim if I am not mistaken.
  ImuFactor(const PreintegratedImuMeasurements<Scalar>& pim);

  // TODO: Add doc-string explaing that gravity points down and what the pose and velocities
  // (realisticall, will want to put this on the symbolic function, then copy theand paste [with
  // modifications] here)
  void operator()(const sym::Pose3<Scalar>& pose_i, const Eigen::Matrix<Scalar, 3, 1>& vel_i,
                  const sym::Pose3<Scalar>& pose_j, const Eigen::Matrix<Scalar, 3, 1>& vel_j,
                  const Eigen::Matrix<Scalar, 3, 1>& gyro_bias_j,
                  const Eigen::Matrix<Scalar, 3, 1>& accel_bias_j,
                  const Eigen::Matrix<Scalar, 3, 1>& gravity, const Scalar epsilon,
                  Eigen::Matrix<Scalar, 9, 1>* const res = nullptr,
                  Eigen::Matrix<Scalar, 9, 24>* const jacobian = nullptr,
                  Eigen::Matrix<Scalar, 24, 24>* const hessian = nullptr,
                  Eigen::Matrix<Scalar, 24, 1>* const rhs = nullptr) const;
};

}  // namespace sym

extern template class sym::ImuFactor<double>;
extern template class sym::ImuFactor<float>;
