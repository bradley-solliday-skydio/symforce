// -----------------------------------------------------------------------------
// This file was autogenerated by symforce from template:
//     cpp_templates/geo_package/ops/CLASS/lie_group_ops.cc.jinja
// Do NOT modify by hand.
// -----------------------------------------------------------------------------

#include "./lie_group_ops.h"

#include <algorithm>
#include <cmath>

#include <sym/rot3.h>

namespace sym {

template <typename Scalar>
sym::Rot3<Scalar> LieGroupOps<Rot3<Scalar>>::FromTangent(const TangentVec& vec,
                                                         const Scalar epsilon) {
  // Total ops: 16

  // Input arrays

  // Intermediate terms (3)
  const Scalar _tmp0 = std::sqrt(std::pow(epsilon, Scalar(2)) + std::pow(vec(0, 0), Scalar(2)) +
                                 std::pow(vec(1, 0), Scalar(2)) + std::pow(vec(2, 0), Scalar(2)));
  const Scalar _tmp1 = (Scalar(1) / Scalar(2)) * _tmp0;
  const Scalar _tmp2 = std::sin(_tmp1) / _tmp0;

  // Output terms (1)
  Eigen::Matrix<Scalar, 4, 1> _res;

  _res[0] = _tmp2 * vec(0, 0);
  _res[1] = _tmp2 * vec(1, 0);
  _res[2] = _tmp2 * vec(2, 0);
  _res[3] = std::cos(_tmp1);

  return sym::Rot3<Scalar>(_res);
}

template <typename Scalar>
typename LieGroupOps<Rot3<Scalar>>::TangentVec LieGroupOps<Rot3<Scalar>>::ToTangent(
    const sym::Rot3<Scalar>& a, const Scalar epsilon) {
  // Total ops: 19

  // Input arrays
  const Eigen::Matrix<Scalar, 4, 1>& _a = a.Data();

  // Intermediate terms (2)
  const Scalar _tmp0 = std::min<Scalar>(std::fabs(_a[3]), 1 - epsilon);
  const Scalar _tmp1 = 2 * (2 * std::min<Scalar>(0, (((_a[3]) > 0) - ((_a[3]) < 0))) + 1) *
                       std::acos(_tmp0) / std::sqrt(1 - std::pow(_tmp0, Scalar(2)));

  // Output terms (1)
  Eigen::Matrix<Scalar, 3, 1> _res;

  _res(0, 0) = _a[0] * _tmp1;
  _res(1, 0) = _a[1] * _tmp1;
  _res(2, 0) = _a[2] * _tmp1;

  return _res;
}

template <typename Scalar>
sym::Rot3<Scalar> LieGroupOps<Rot3<Scalar>>::Retract(const sym::Rot3<Scalar>& a,
                                                     const TangentVec& vec, const Scalar epsilon) {
  // Total ops: 49

  // Input arrays
  const Eigen::Matrix<Scalar, 4, 1>& _a = a.Data();

  // Intermediate terms (8)
  const Scalar _tmp0 = std::sqrt(std::pow(epsilon, Scalar(2)) + std::pow(vec(0, 0), Scalar(2)) +
                                 std::pow(vec(1, 0), Scalar(2)) + std::pow(vec(2, 0), Scalar(2)));
  const Scalar _tmp1 = (Scalar(1) / Scalar(2)) * _tmp0;
  const Scalar _tmp2 = std::cos(_tmp1);
  const Scalar _tmp3 = std::sin(_tmp1) / _tmp0;
  const Scalar _tmp4 = _a[2] * _tmp3;
  const Scalar _tmp5 = _a[3] * _tmp3;
  const Scalar _tmp6 = _a[1] * _tmp3;
  const Scalar _tmp7 = _a[0] * _tmp3;

  // Output terms (1)
  Eigen::Matrix<Scalar, 4, 1> _res;

  _res[0] = _a[0] * _tmp2 - _tmp4 * vec(1, 0) + _tmp5 * vec(0, 0) + _tmp6 * vec(2, 0);
  _res[1] = _a[1] * _tmp2 + _tmp4 * vec(0, 0) + _tmp5 * vec(1, 0) - _tmp7 * vec(2, 0);
  _res[2] = _a[2] * _tmp2 + _tmp5 * vec(2, 0) - _tmp6 * vec(0, 0) + _tmp7 * vec(1, 0);
  _res[3] = _a[3] * _tmp2 - _tmp4 * vec(2, 0) - _tmp6 * vec(1, 0) - _tmp7 * vec(0, 0);

  return sym::Rot3<Scalar>(_res);
}

template <typename Scalar>
typename LieGroupOps<Rot3<Scalar>>::TangentVec LieGroupOps<Rot3<Scalar>>::LocalCoordinates(
    const sym::Rot3<Scalar>& a, const sym::Rot3<Scalar>& b, const Scalar epsilon) {
  // Total ops: 57

  // Input arrays
  const Eigen::Matrix<Scalar, 4, 1>& _a = a.Data();
  const Eigen::Matrix<Scalar, 4, 1>& _b = b.Data();

  // Intermediate terms (3)
  const Scalar _tmp0 = _a[0] * _b[0] + _a[1] * _b[1] + _a[2] * _b[2] + _a[3] * _b[3];
  const Scalar _tmp1 = std::min<Scalar>(std::fabs(_tmp0), 1 - epsilon);
  const Scalar _tmp2 = 2 * (2 * std::min<Scalar>(0, (((_tmp0) > 0) - ((_tmp0) < 0))) + 1) *
                       std::acos(_tmp1) / std::sqrt(1 - std::pow(_tmp1, Scalar(2)));

  // Output terms (1)
  Eigen::Matrix<Scalar, 3, 1> _res;

  _res(0, 0) = _tmp2 * (-_a[0] * _b[3] - _a[1] * _b[2] + _a[2] * _b[1] + _a[3] * _b[0]);
  _res(1, 0) = _tmp2 * (_a[0] * _b[2] - _a[1] * _b[3] - _a[2] * _b[0] + _a[3] * _b[1]);
  _res(2, 0) = _tmp2 * (-_a[0] * _b[1] + _a[1] * _b[0] - _a[2] * _b[3] + _a[3] * _b[2]);

  return _res;
}

}  // namespace sym

// Explicit instantiation
template struct sym::LieGroupOps<sym::Rot3<double>>;
template struct sym::LieGroupOps<sym::Rot3<float>>;
