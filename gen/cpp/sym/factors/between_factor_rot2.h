// -----------------------------------------------------------------------------
// This file was autogenerated by symforce from template:
//     function/FUNCTION.h.jinja
// Do NOT modify by hand.
// -----------------------------------------------------------------------------

#pragma once

#include <Eigen/Dense>

#include <sym/rot2.h>

namespace sym {

/**
 * Residual that penalizes the difference between between(a, b) and a_T_b.
 *
 * In vector space terms that would be:
 *     (b - a) - a_T_b
 *
 * In lie group terms:
 *     local_coordinates(a_T_b, between(a, b))
 *     to_tangent(compose(inverse(a_T_b), compose(inverse(a), b)))
 *
 * Args:
 *     sqrt_info: Square root information matrix to whiten residual. This can be computed from
 *                a covariance matrix as the cholesky decomposition of the inverse. In the case
 *                of a diagonal it will contain 1/sigma values. Must match the tangent dim.
 *     jacobian: (1x2) jacobian of res wrt args a (1), b (1)
 *     hessian: (2x2) Gauss-Newton hessian for args a (1), b (1)
 *     rhs: (2x1) Gauss-Newton rhs for args a (1), b (1)
 */
template <typename Scalar>
void BetweenFactorRot2(const sym::Rot2<Scalar>& a, const sym::Rot2<Scalar>& b,
                       const sym::Rot2<Scalar>& a_T_b, const Eigen::Matrix<Scalar, 1, 1>& sqrt_info,
                       const Scalar epsilon, Eigen::Matrix<Scalar, 1, 1>* const res = nullptr,
                       Eigen::Matrix<Scalar, 1, 2>* const jacobian = nullptr,
                       Eigen::Matrix<Scalar, 2, 2>* const hessian = nullptr,
                       Eigen::Matrix<Scalar, 2, 1>* const rhs = nullptr) {
  // Total ops: 40

  // Input arrays
  const Eigen::Matrix<Scalar, 2, 1>& _a = a.Data();
  const Eigen::Matrix<Scalar, 2, 1>& _b = b.Data();
  const Eigen::Matrix<Scalar, 2, 1>& _a_T_b = a_T_b.Data();

  // Intermediate terms (16)
  const Scalar _tmp0 = _a[0] * _b[0] + _a[1] * _b[1];
  const Scalar _tmp1 = _a_T_b[1] * _tmp0;
  const Scalar _tmp2 = _a[0] * _b[1] - _a[1] * _b[0];
  const Scalar _tmp3 = _a_T_b[0] * _tmp2;
  const Scalar _tmp4 = _a_T_b[0] * _tmp0 + _a_T_b[1] * _tmp2;
  const Scalar _tmp5 = _tmp4 + epsilon * ((((_tmp4) > 0) - ((_tmp4) < 0)) + Scalar(0.5));
  const Scalar _tmp6 = std::atan2(-_tmp1 + _tmp3, _tmp5);
  const Scalar _tmp7 = std::pow(Scalar(_tmp1 - _tmp3), Scalar(2));
  const Scalar _tmp8 = std::pow(_tmp5, Scalar(2));
  const Scalar _tmp9 = _tmp4 / _tmp5 + _tmp7 / _tmp8;
  const Scalar _tmp10 = _tmp7 + _tmp8;
  const Scalar _tmp11 = _tmp8 * _tmp9 / _tmp10;
  const Scalar _tmp12 = _tmp11 * sqrt_info(0, 0);
  const Scalar _tmp13 = std::pow(sqrt_info(0, 0), Scalar(2));
  const Scalar _tmp14 = _tmp13 * std::pow(_tmp5, Scalar(4)) * std::pow(_tmp9, Scalar(2)) /
                        std::pow(_tmp10, Scalar(2));
  const Scalar _tmp15 = _tmp11 * _tmp13 * _tmp6;

  // Output terms (4)
  if (res != nullptr) {
    Eigen::Matrix<Scalar, 1, 1>& _res = (*res);

    _res(0, 0) = _tmp6 * sqrt_info(0, 0);
  }

  if (jacobian != nullptr) {
    Eigen::Matrix<Scalar, 1, 2>& _jacobian = (*jacobian);

    _jacobian(0, 0) = -_tmp12;
    _jacobian(0, 1) = _tmp12;
  }

  if (hessian != nullptr) {
    Eigen::Matrix<Scalar, 2, 2>& _hessian = (*hessian);

    _hessian(0, 0) = _tmp14;
    _hessian(1, 0) = -_tmp14;
    _hessian(0, 1) = 0;
    _hessian(1, 1) = _tmp14;
  }

  if (rhs != nullptr) {
    Eigen::Matrix<Scalar, 2, 1>& _rhs = (*rhs);

    _rhs(0, 0) = -_tmp15;
    _rhs(1, 0) = _tmp15;
  }
}  // NOLINT(readability/fn_size)

// NOLINTNEXTLINE(readability/fn_size)
}  // namespace sym
