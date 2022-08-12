// -----------------------------------------------------------------------------
// This file was autogenerated by symforce from template:
//     function/FUNCTION.h.jinja
// Do NOT modify by hand.
// -----------------------------------------------------------------------------

#pragma once

#include <Eigen/Dense>

#include <sym/pose3.h>

namespace sym {

/**
 * Residual that penalizes the difference between a value and prior (desired / measured value).
 *
 * In vector space terms that would be:
 *     prior - value
 *
 * In lie group terms:
 *     to_tangent(compose(inverse(value), prior))
 *
 * Args:
 *     sqrt_info: Square root information matrix to whiten residual. This can be computed from
 *                a covariance matrix as the cholesky decomposition of the inverse. In the case
 *                of a diagonal it will contain 1/sigma values. Must match the tangent dim.
 *     jacobian: (6x6) jacobian of res wrt arg value (6)
 *     hessian: (6x6) Gauss-Newton hessian for arg value (6)
 *     rhs: (6x1) Gauss-Newton rhs for arg value (6)
 */
template <typename Scalar>
void PriorFactorPose3(const sym::Pose3<Scalar>& value, const sym::Pose3<Scalar>& prior,
                      const Eigen::Matrix<Scalar, 6, 6>& sqrt_info, const Scalar epsilon,
                      Eigen::Matrix<Scalar, 6, 1>* const res = nullptr,
                      Eigen::Matrix<Scalar, 6, 6>* const jacobian = nullptr,
                      Eigen::Matrix<Scalar, 6, 6>* const hessian = nullptr,
                      Eigen::Matrix<Scalar, 6, 1>* const rhs = nullptr) {
  // Total ops: 785

  // Input arrays
  const Eigen::Matrix<Scalar, 7, 1>& _value = value.Data();
  const Eigen::Matrix<Scalar, 7, 1>& _prior = prior.Data();

  // Intermediate terms (97)
  const Scalar _tmp0 = _prior[2] * _value[3];
  const Scalar _tmp1 = _prior[1] * _value[0];
  const Scalar _tmp2 = _prior[0] * _value[1];
  const Scalar _tmp3 = _prior[3] * _value[2];
  const Scalar _tmp4 = _prior[3] * _value[3];
  const Scalar _tmp5 = _prior[2] * _value[2];
  const Scalar _tmp6 = _prior[1] * _value[1];
  const Scalar _tmp7 = _prior[0] * _value[0];
  const Scalar _tmp8 = _tmp4 + _tmp5 + _tmp6 + _tmp7;
  const Scalar _tmp9 = std::min<Scalar>(std::fabs(_tmp8), 1 - epsilon);
  const Scalar _tmp10 = (2 * std::min<Scalar>(0, (((_tmp8) > 0) - ((_tmp8) < 0))) + 1) *
                        std::acos(_tmp9) / std::sqrt(Scalar(1 - std::pow(_tmp9, Scalar(2))));
  const Scalar _tmp11 = _tmp10 * (-_tmp0 + _tmp1 - _tmp2 + _tmp3);
  const Scalar _tmp12 = 2 * _tmp11;
  const Scalar _tmp13 = -_prior[6] + _value[6];
  const Scalar _tmp14 = -_prior[5] + _value[5];
  const Scalar _tmp15 = -_prior[4] + _value[4];
  const Scalar _tmp16 = _prior[2] * _value[0];
  const Scalar _tmp17 = _prior[1] * _value[3];
  const Scalar _tmp18 = _prior[0] * _value[2];
  const Scalar _tmp19 = _prior[3] * _value[1];
  const Scalar _tmp20 = 2 * _tmp10;
  const Scalar _tmp21 = _tmp20 * (-_tmp16 - _tmp17 + _tmp18 + _tmp19);
  const Scalar _tmp22 = -_prior[0] * _value[3] - _prior[1] * _value[2] + _prior[2] * _value[1] +
                        _prior[3] * _value[0];
  const Scalar _tmp23 = _tmp22 * sqrt_info(0, 0);
  const Scalar _tmp24 = _tmp12 * sqrt_info(0, 2) + _tmp13 * sqrt_info(0, 5) +
                        _tmp14 * sqrt_info(0, 4) + _tmp15 * sqrt_info(0, 3) + _tmp20 * _tmp23 +
                        _tmp21 * sqrt_info(0, 1);
  const Scalar _tmp25 = _tmp20 * _tmp22;
  const Scalar _tmp26 = _tmp12 * sqrt_info(1, 2) + _tmp13 * sqrt_info(1, 5) +
                        _tmp14 * sqrt_info(1, 4) + _tmp15 * sqrt_info(1, 3) +
                        _tmp21 * sqrt_info(1, 1) + _tmp25 * sqrt_info(1, 0);
  const Scalar _tmp27 = _tmp12 * sqrt_info(2, 2) + _tmp13 * sqrt_info(2, 5) +
                        _tmp14 * sqrt_info(2, 4) + _tmp15 * sqrt_info(2, 3) +
                        _tmp21 * sqrt_info(2, 1) + _tmp25 * sqrt_info(2, 0);
  const Scalar _tmp28 = 2 * sqrt_info(3, 2);
  const Scalar _tmp29 = _tmp11 * _tmp28 + _tmp13 * sqrt_info(3, 5) + _tmp14 * sqrt_info(3, 4) +
                        _tmp15 * sqrt_info(3, 3) + _tmp21 * sqrt_info(3, 1) +
                        _tmp25 * sqrt_info(3, 0);
  const Scalar _tmp30 = _tmp12 * sqrt_info(4, 2) + _tmp13 * sqrt_info(4, 5) +
                        _tmp14 * sqrt_info(4, 4) + _tmp15 * sqrt_info(4, 3) +
                        _tmp21 * sqrt_info(4, 1) + _tmp25 * sqrt_info(4, 0);
  const Scalar _tmp31 = _tmp12 * sqrt_info(5, 2) + _tmp13 * sqrt_info(5, 5) +
                        _tmp14 * sqrt_info(5, 4) + _tmp15 * sqrt_info(5, 3) +
                        _tmp21 * sqrt_info(5, 1) + _tmp25 * sqrt_info(5, 0);
  const Scalar _tmp32 = std::pow(_tmp22, Scalar(2));
  const Scalar _tmp33 = _tmp4 + _tmp5 + _tmp6 + _tmp7;
  const Scalar _tmp34 = (((_tmp33) > 0) - ((_tmp33) < 0));
  const Scalar _tmp35 = std::fabs(_tmp33);
  const Scalar _tmp36 = epsilon - 1;
  const Scalar _tmp37 = _tmp34 * ((((_tmp35 + _tmp36) > 0) - ((_tmp35 + _tmp36) < 0)) - 1);
  const Scalar _tmp38 = std::min<Scalar>(_tmp35, -_tmp36);
  const Scalar _tmp39 = std::pow(_tmp38, Scalar(2)) - 1;
  const Scalar _tmp40 = -_tmp39;
  const Scalar _tmp41 = std::min<Scalar>(0, _tmp34) + Scalar(1) / Scalar(2);
  const Scalar _tmp42 = _tmp41 * std::acos(_tmp38);
  const Scalar _tmp43 = _tmp38 * _tmp42 / (_tmp40 * std::sqrt(_tmp40));
  const Scalar _tmp44 = _tmp37 * _tmp43;
  const Scalar _tmp45 = _tmp32 * _tmp44;
  const Scalar _tmp46 = _tmp16 + _tmp17 - _tmp18 - _tmp19;
  const Scalar _tmp47 = _tmp44 * _tmp46;
  const Scalar _tmp48 = _tmp22 * _tmp47;
  const Scalar _tmp49 = _tmp0 - _tmp1 + _tmp2 - _tmp3;
  const Scalar _tmp50 = _tmp41 / _tmp39;
  const Scalar _tmp51 = _tmp37 * _tmp50;
  const Scalar _tmp52 = _tmp49 * _tmp51;
  const Scalar _tmp53 = _tmp22 * _tmp52;
  const Scalar _tmp54 = _tmp42 / std::sqrt(_tmp40);
  const Scalar _tmp55 = 2 * _tmp54;
  const Scalar _tmp56 = _tmp46 * _tmp55;
  const Scalar _tmp57 = _tmp33 * _tmp54;
  const Scalar _tmp58 = 2 * _tmp57;
  const Scalar _tmp59 = _tmp22 * _tmp51;
  const Scalar _tmp60 = _tmp46 * _tmp59;
  const Scalar _tmp61 = _tmp44 * _tmp49;
  const Scalar _tmp62 = _tmp22 * _tmp61;
  const Scalar _tmp63 = _tmp49 * _tmp55;
  const Scalar _tmp64 = _tmp32 * _tmp51;
  const Scalar _tmp65 =
      _tmp45 * sqrt_info(0, 0) - _tmp48 * sqrt_info(0, 1) - _tmp53 * sqrt_info(0, 2) +
      _tmp56 * sqrt_info(0, 2) + _tmp58 * sqrt_info(0, 0) - _tmp60 * sqrt_info(0, 1) -
      _tmp62 * sqrt_info(0, 2) - _tmp63 * sqrt_info(0, 1) + _tmp64 * sqrt_info(0, 0);
  const Scalar _tmp66 =
      _tmp45 * sqrt_info(1, 0) - _tmp48 * sqrt_info(1, 1) - _tmp53 * sqrt_info(1, 2) +
      _tmp56 * sqrt_info(1, 2) + _tmp58 * sqrt_info(1, 0) - _tmp60 * sqrt_info(1, 1) -
      _tmp62 * sqrt_info(1, 2) - _tmp63 * sqrt_info(1, 1) + _tmp64 * sqrt_info(1, 0);
  const Scalar _tmp67 =
      _tmp45 * sqrt_info(2, 0) - _tmp48 * sqrt_info(2, 1) - _tmp53 * sqrt_info(2, 2) +
      _tmp56 * sqrt_info(2, 2) + _tmp58 * sqrt_info(2, 0) - _tmp60 * sqrt_info(2, 1) -
      _tmp62 * sqrt_info(2, 2) - _tmp63 * sqrt_info(2, 1) + _tmp64 * sqrt_info(2, 0);
  const Scalar _tmp68 = _tmp28 * _tmp54;
  const Scalar _tmp69 = _tmp45 * sqrt_info(3, 0) + _tmp46 * _tmp68 - _tmp48 * sqrt_info(3, 1) -
                        _tmp53 * sqrt_info(3, 2) + _tmp58 * sqrt_info(3, 0) -
                        _tmp60 * sqrt_info(3, 1) - _tmp62 * sqrt_info(3, 2) -
                        _tmp63 * sqrt_info(3, 1) + _tmp64 * sqrt_info(3, 0);
  const Scalar _tmp70 = _tmp52 * sqrt_info(4, 2);
  const Scalar _tmp71 = -_tmp22 * _tmp70 + _tmp45 * sqrt_info(4, 0) - _tmp48 * sqrt_info(4, 1) +
                        _tmp56 * sqrt_info(4, 2) + _tmp58 * sqrt_info(4, 0) -
                        _tmp60 * sqrt_info(4, 1) - _tmp62 * sqrt_info(4, 2) -
                        _tmp63 * sqrt_info(4, 1) + _tmp64 * sqrt_info(4, 0);
  const Scalar _tmp72 =
      _tmp45 * sqrt_info(5, 0) - _tmp48 * sqrt_info(5, 1) - _tmp53 * sqrt_info(5, 2) +
      _tmp56 * sqrt_info(5, 2) + _tmp58 * sqrt_info(5, 0) - _tmp60 * sqrt_info(5, 1) -
      _tmp62 * sqrt_info(5, 2) - _tmp63 * sqrt_info(5, 1) + _tmp64 * sqrt_info(5, 0);
  const Scalar _tmp73 = std::pow(_tmp46, Scalar(2));
  const Scalar _tmp74 = _tmp51 * _tmp73;
  const Scalar _tmp75 = _tmp23 * _tmp46;
  const Scalar _tmp76 = _tmp47 * _tmp49;
  const Scalar _tmp77 = _tmp22 * _tmp55;
  const Scalar _tmp78 = _tmp46 * _tmp52;
  const Scalar _tmp79 = _tmp44 * _tmp73;
  const Scalar _tmp80 = -_tmp44 * _tmp75 - _tmp51 * _tmp75 + _tmp58 * sqrt_info(0, 1) +
                        _tmp63 * sqrt_info(0, 0) + _tmp74 * sqrt_info(0, 1) +
                        _tmp76 * sqrt_info(0, 2) + _tmp77 * sqrt_info(0, 2) +
                        _tmp78 * sqrt_info(0, 2) + _tmp79 * sqrt_info(0, 1);
  const Scalar _tmp81 =
      -_tmp48 * sqrt_info(1, 0) + _tmp58 * sqrt_info(1, 1) - _tmp60 * sqrt_info(1, 0) +
      _tmp63 * sqrt_info(1, 0) + _tmp74 * sqrt_info(1, 1) + _tmp76 * sqrt_info(1, 2) +
      _tmp77 * sqrt_info(1, 2) + _tmp78 * sqrt_info(1, 2) + _tmp79 * sqrt_info(1, 1);
  const Scalar _tmp82 =
      -_tmp48 * sqrt_info(2, 0) + _tmp58 * sqrt_info(2, 1) - _tmp60 * sqrt_info(2, 0) +
      _tmp63 * sqrt_info(2, 0) + _tmp74 * sqrt_info(2, 1) + _tmp76 * sqrt_info(2, 2) +
      _tmp77 * sqrt_info(2, 2) + _tmp78 * sqrt_info(2, 2) + _tmp79 * sqrt_info(2, 1);
  const Scalar _tmp83 = _tmp49 * sqrt_info(3, 0);
  const Scalar _tmp84 = _tmp22 * _tmp68 - _tmp48 * sqrt_info(3, 0) + _tmp55 * _tmp83 +
                        _tmp58 * sqrt_info(3, 1) - _tmp60 * sqrt_info(3, 0) +
                        _tmp74 * sqrt_info(3, 1) + _tmp76 * sqrt_info(3, 2) +
                        _tmp78 * sqrt_info(3, 2) + _tmp79 * sqrt_info(3, 1);
  const Scalar _tmp85 = _tmp46 * _tmp70 - _tmp48 * sqrt_info(4, 0) + _tmp58 * sqrt_info(4, 1) -
                        _tmp60 * sqrt_info(4, 0) + _tmp63 * sqrt_info(4, 0) +
                        _tmp74 * sqrt_info(4, 1) + _tmp76 * sqrt_info(4, 2) +
                        _tmp77 * sqrt_info(4, 2) + _tmp79 * sqrt_info(4, 1);
  const Scalar _tmp86 =
      -_tmp48 * sqrt_info(5, 0) + _tmp58 * sqrt_info(5, 1) - _tmp60 * sqrt_info(5, 0) +
      _tmp63 * sqrt_info(5, 0) + _tmp74 * sqrt_info(5, 1) + _tmp76 * sqrt_info(5, 2) +
      _tmp77 * sqrt_info(5, 2) + _tmp78 * sqrt_info(5, 2) + _tmp79 * sqrt_info(5, 1);
  const Scalar _tmp87 = _tmp37 * std::pow(_tmp49, Scalar(2));
  const Scalar _tmp88 = _tmp50 * _tmp87;
  const Scalar _tmp89 = _tmp43 * _tmp87;
  const Scalar _tmp90 = -_tmp23 * _tmp52 - _tmp23 * _tmp61 - _tmp56 * sqrt_info(0, 0) +
                        _tmp58 * sqrt_info(0, 2) + _tmp76 * sqrt_info(0, 1) -
                        _tmp77 * sqrt_info(0, 1) + _tmp78 * sqrt_info(0, 1) +
                        _tmp88 * sqrt_info(0, 2) + _tmp89 * sqrt_info(0, 2);
  const Scalar _tmp91 =
      -_tmp53 * sqrt_info(1, 0) - _tmp56 * sqrt_info(1, 0) + _tmp58 * sqrt_info(1, 2) -
      _tmp62 * sqrt_info(1, 0) + _tmp76 * sqrt_info(1, 1) - _tmp77 * sqrt_info(1, 1) +
      _tmp78 * sqrt_info(1, 1) + _tmp88 * sqrt_info(1, 2) + _tmp89 * sqrt_info(1, 2);
  const Scalar _tmp92 =
      -_tmp53 * sqrt_info(2, 0) - _tmp56 * sqrt_info(2, 0) + _tmp58 * sqrt_info(2, 2) -
      _tmp62 * sqrt_info(2, 0) + _tmp76 * sqrt_info(2, 1) - _tmp77 * sqrt_info(2, 1) +
      _tmp78 * sqrt_info(2, 1) + _tmp88 * sqrt_info(2, 2) + _tmp89 * sqrt_info(2, 2);
  const Scalar _tmp93 = -_tmp22 * _tmp44 * _tmp83 + _tmp28 * _tmp57 - _tmp56 * sqrt_info(3, 0) -
                        _tmp59 * _tmp83 + _tmp76 * sqrt_info(3, 1) - _tmp77 * sqrt_info(3, 1) +
                        _tmp78 * sqrt_info(3, 1) + _tmp88 * sqrt_info(3, 2) +
                        _tmp89 * sqrt_info(3, 2);
  const Scalar _tmp94 =
      -_tmp53 * sqrt_info(4, 0) - _tmp56 * sqrt_info(4, 0) + _tmp58 * sqrt_info(4, 2) -
      _tmp62 * sqrt_info(4, 0) + _tmp76 * sqrt_info(4, 1) - _tmp77 * sqrt_info(4, 1) +
      _tmp78 * sqrt_info(4, 1) + _tmp88 * sqrt_info(4, 2) + _tmp89 * sqrt_info(4, 2);
  const Scalar _tmp95 = _tmp87 * sqrt_info(5, 2);
  const Scalar _tmp96 = _tmp43 * _tmp95 + _tmp50 * _tmp95 - _tmp53 * sqrt_info(5, 0) -
                        _tmp56 * sqrt_info(5, 0) + _tmp58 * sqrt_info(5, 2) -
                        _tmp62 * sqrt_info(5, 0) + _tmp76 * sqrt_info(5, 1) -
                        _tmp77 * sqrt_info(5, 1) + _tmp78 * sqrt_info(5, 1);

  // Output terms (4)
  if (res != nullptr) {
    Eigen::Matrix<Scalar, 6, 1>& _res = (*res);

    _res(0, 0) = _tmp24;
    _res(1, 0) = _tmp26;
    _res(2, 0) = _tmp27;
    _res(3, 0) = _tmp29;
    _res(4, 0) = _tmp30;
    _res(5, 0) = _tmp31;
  }

  if (jacobian != nullptr) {
    Eigen::Matrix<Scalar, 6, 6>& _jacobian = (*jacobian);

    _jacobian(0, 0) = _tmp65;
    _jacobian(1, 0) = _tmp66;
    _jacobian(2, 0) = _tmp67;
    _jacobian(3, 0) = _tmp69;
    _jacobian(4, 0) = _tmp71;
    _jacobian(5, 0) = _tmp72;
    _jacobian(0, 1) = _tmp80;
    _jacobian(1, 1) = _tmp81;
    _jacobian(2, 1) = _tmp82;
    _jacobian(3, 1) = _tmp84;
    _jacobian(4, 1) = _tmp85;
    _jacobian(5, 1) = _tmp86;
    _jacobian(0, 2) = _tmp90;
    _jacobian(1, 2) = _tmp91;
    _jacobian(2, 2) = _tmp92;
    _jacobian(3, 2) = _tmp93;
    _jacobian(4, 2) = _tmp94;
    _jacobian(5, 2) = _tmp96;
    _jacobian(0, 3) = sqrt_info(0, 3);
    _jacobian(1, 3) = sqrt_info(1, 3);
    _jacobian(2, 3) = sqrt_info(2, 3);
    _jacobian(3, 3) = sqrt_info(3, 3);
    _jacobian(4, 3) = sqrt_info(4, 3);
    _jacobian(5, 3) = sqrt_info(5, 3);
    _jacobian(0, 4) = sqrt_info(0, 4);
    _jacobian(1, 4) = sqrt_info(1, 4);
    _jacobian(2, 4) = sqrt_info(2, 4);
    _jacobian(3, 4) = sqrt_info(3, 4);
    _jacobian(4, 4) = sqrt_info(4, 4);
    _jacobian(5, 4) = sqrt_info(5, 4);
    _jacobian(0, 5) = sqrt_info(0, 5);
    _jacobian(1, 5) = sqrt_info(1, 5);
    _jacobian(2, 5) = sqrt_info(2, 5);
    _jacobian(3, 5) = sqrt_info(3, 5);
    _jacobian(4, 5) = sqrt_info(4, 5);
    _jacobian(5, 5) = sqrt_info(5, 5);
  }

  if (hessian != nullptr) {
    Eigen::Matrix<Scalar, 6, 6>& _hessian = (*hessian);

    _hessian(0, 0) = std::pow(_tmp65, Scalar(2)) + std::pow(_tmp66, Scalar(2)) +
                     std::pow(_tmp67, Scalar(2)) + std::pow(_tmp69, Scalar(2)) +
                     std::pow(_tmp71, Scalar(2)) + std::pow(_tmp72, Scalar(2));
    _hessian(1, 0) = _tmp65 * _tmp80 + _tmp66 * _tmp81 + _tmp67 * _tmp82 + _tmp69 * _tmp84 +
                     _tmp71 * _tmp85 + _tmp72 * _tmp86;
    _hessian(2, 0) = _tmp65 * _tmp90 + _tmp66 * _tmp91 + _tmp67 * _tmp92 + _tmp69 * _tmp93 +
                     _tmp71 * _tmp94 + _tmp72 * _tmp96;
    _hessian(3, 0) = _tmp65 * sqrt_info(0, 3) + _tmp66 * sqrt_info(1, 3) +
                     _tmp67 * sqrt_info(2, 3) + _tmp69 * sqrt_info(3, 3) +
                     _tmp71 * sqrt_info(4, 3) + _tmp72 * sqrt_info(5, 3);
    _hessian(4, 0) = _tmp65 * sqrt_info(0, 4) + _tmp66 * sqrt_info(1, 4) +
                     _tmp67 * sqrt_info(2, 4) + _tmp69 * sqrt_info(3, 4) +
                     _tmp71 * sqrt_info(4, 4) + _tmp72 * sqrt_info(5, 4);
    _hessian(5, 0) = _tmp65 * sqrt_info(0, 5) + _tmp66 * sqrt_info(1, 5) +
                     _tmp67 * sqrt_info(2, 5) + _tmp69 * sqrt_info(3, 5) +
                     _tmp71 * sqrt_info(4, 5) + _tmp72 * sqrt_info(5, 5);
    _hessian(0, 1) = 0;
    _hessian(1, 1) = std::pow(_tmp80, Scalar(2)) + std::pow(_tmp81, Scalar(2)) +
                     std::pow(_tmp82, Scalar(2)) + std::pow(_tmp84, Scalar(2)) +
                     std::pow(_tmp85, Scalar(2)) + std::pow(_tmp86, Scalar(2));
    _hessian(2, 1) = _tmp80 * _tmp90 + _tmp81 * _tmp91 + _tmp82 * _tmp92 + _tmp84 * _tmp93 +
                     _tmp85 * _tmp94 + _tmp86 * _tmp96;
    _hessian(3, 1) = _tmp80 * sqrt_info(0, 3) + _tmp81 * sqrt_info(1, 3) +
                     _tmp82 * sqrt_info(2, 3) + _tmp84 * sqrt_info(3, 3) +
                     _tmp85 * sqrt_info(4, 3) + _tmp86 * sqrt_info(5, 3);
    _hessian(4, 1) = _tmp80 * sqrt_info(0, 4) + _tmp81 * sqrt_info(1, 4) +
                     _tmp82 * sqrt_info(2, 4) + _tmp84 * sqrt_info(3, 4) +
                     _tmp85 * sqrt_info(4, 4) + _tmp86 * sqrt_info(5, 4);
    _hessian(5, 1) = _tmp80 * sqrt_info(0, 5) + _tmp81 * sqrt_info(1, 5) +
                     _tmp82 * sqrt_info(2, 5) + _tmp84 * sqrt_info(3, 5) +
                     _tmp85 * sqrt_info(4, 5) + _tmp86 * sqrt_info(5, 5);
    _hessian(0, 2) = 0;
    _hessian(1, 2) = 0;
    _hessian(2, 2) = std::pow(_tmp90, Scalar(2)) + std::pow(_tmp91, Scalar(2)) +
                     std::pow(_tmp92, Scalar(2)) + std::pow(_tmp93, Scalar(2)) +
                     std::pow(_tmp94, Scalar(2)) + std::pow(_tmp96, Scalar(2));
    _hessian(3, 2) = _tmp90 * sqrt_info(0, 3) + _tmp91 * sqrt_info(1, 3) +
                     _tmp92 * sqrt_info(2, 3) + _tmp93 * sqrt_info(3, 3) +
                     _tmp94 * sqrt_info(4, 3) + _tmp96 * sqrt_info(5, 3);
    _hessian(4, 2) = _tmp90 * sqrt_info(0, 4) + _tmp91 * sqrt_info(1, 4) +
                     _tmp92 * sqrt_info(2, 4) + _tmp93 * sqrt_info(3, 4) +
                     _tmp94 * sqrt_info(4, 4) + _tmp96 * sqrt_info(5, 4);
    _hessian(5, 2) = _tmp90 * sqrt_info(0, 5) + _tmp91 * sqrt_info(1, 5) +
                     _tmp92 * sqrt_info(2, 5) + _tmp93 * sqrt_info(3, 5) +
                     _tmp94 * sqrt_info(4, 5) + _tmp96 * sqrt_info(5, 5);
    _hessian(0, 3) = 0;
    _hessian(1, 3) = 0;
    _hessian(2, 3) = 0;
    _hessian(3, 3) = std::pow(sqrt_info(0, 3), Scalar(2)) + std::pow(sqrt_info(1, 3), Scalar(2)) +
                     std::pow(sqrt_info(2, 3), Scalar(2)) + std::pow(sqrt_info(3, 3), Scalar(2)) +
                     std::pow(sqrt_info(4, 3), Scalar(2)) + std::pow(sqrt_info(5, 3), Scalar(2));
    _hessian(4, 3) = sqrt_info(0, 3) * sqrt_info(0, 4) + sqrt_info(1, 3) * sqrt_info(1, 4) +
                     sqrt_info(2, 3) * sqrt_info(2, 4) + sqrt_info(3, 3) * sqrt_info(3, 4) +
                     sqrt_info(4, 3) * sqrt_info(4, 4) + sqrt_info(5, 3) * sqrt_info(5, 4);
    _hessian(5, 3) = sqrt_info(0, 3) * sqrt_info(0, 5) + sqrt_info(1, 3) * sqrt_info(1, 5) +
                     sqrt_info(2, 3) * sqrt_info(2, 5) + sqrt_info(3, 3) * sqrt_info(3, 5) +
                     sqrt_info(4, 3) * sqrt_info(4, 5) + sqrt_info(5, 3) * sqrt_info(5, 5);
    _hessian(0, 4) = 0;
    _hessian(1, 4) = 0;
    _hessian(2, 4) = 0;
    _hessian(3, 4) = 0;
    _hessian(4, 4) = std::pow(sqrt_info(0, 4), Scalar(2)) + std::pow(sqrt_info(1, 4), Scalar(2)) +
                     std::pow(sqrt_info(2, 4), Scalar(2)) + std::pow(sqrt_info(3, 4), Scalar(2)) +
                     std::pow(sqrt_info(4, 4), Scalar(2)) + std::pow(sqrt_info(5, 4), Scalar(2));
    _hessian(5, 4) = sqrt_info(0, 4) * sqrt_info(0, 5) + sqrt_info(1, 4) * sqrt_info(1, 5) +
                     sqrt_info(2, 4) * sqrt_info(2, 5) + sqrt_info(3, 4) * sqrt_info(3, 5) +
                     sqrt_info(4, 4) * sqrt_info(4, 5) + sqrt_info(5, 4) * sqrt_info(5, 5);
    _hessian(0, 5) = 0;
    _hessian(1, 5) = 0;
    _hessian(2, 5) = 0;
    _hessian(3, 5) = 0;
    _hessian(4, 5) = 0;
    _hessian(5, 5) = std::pow(sqrt_info(0, 5), Scalar(2)) + std::pow(sqrt_info(1, 5), Scalar(2)) +
                     std::pow(sqrt_info(2, 5), Scalar(2)) + std::pow(sqrt_info(3, 5), Scalar(2)) +
                     std::pow(sqrt_info(4, 5), Scalar(2)) + std::pow(sqrt_info(5, 5), Scalar(2));
  }

  if (rhs != nullptr) {
    Eigen::Matrix<Scalar, 6, 1>& _rhs = (*rhs);

    _rhs(0, 0) = _tmp24 * _tmp65 + _tmp26 * _tmp66 + _tmp27 * _tmp67 + _tmp29 * _tmp69 +
                 _tmp30 * _tmp71 + _tmp31 * _tmp72;
    _rhs(1, 0) = _tmp24 * _tmp80 + _tmp26 * _tmp81 + _tmp27 * _tmp82 + _tmp29 * _tmp84 +
                 _tmp30 * _tmp85 + _tmp31 * _tmp86;
    _rhs(2, 0) = _tmp24 * _tmp90 + _tmp26 * _tmp91 + _tmp27 * _tmp92 + _tmp29 * _tmp93 +
                 _tmp30 * _tmp94 + _tmp31 * _tmp96;
    _rhs(3, 0) = _tmp24 * sqrt_info(0, 3) + _tmp26 * sqrt_info(1, 3) + _tmp27 * sqrt_info(2, 3) +
                 _tmp29 * sqrt_info(3, 3) + _tmp30 * sqrt_info(4, 3) + _tmp31 * sqrt_info(5, 3);
    _rhs(4, 0) = _tmp24 * sqrt_info(0, 4) + _tmp26 * sqrt_info(1, 4) + _tmp27 * sqrt_info(2, 4) +
                 _tmp29 * sqrt_info(3, 4) + _tmp30 * sqrt_info(4, 4) + _tmp31 * sqrt_info(5, 4);
    _rhs(5, 0) = _tmp24 * sqrt_info(0, 5) + _tmp26 * sqrt_info(1, 5) + _tmp27 * sqrt_info(2, 5) +
                 _tmp29 * sqrt_info(3, 5) + _tmp30 * sqrt_info(4, 5) + _tmp31 * sqrt_info(5, 5);
  }
}  // NOLINT(readability/fn_size)

// NOLINTNEXTLINE(readability/fn_size)
}  // namespace sym
