// -----------------------------------------------------------------------------
// This file was autogenerated by symforce from template:
//     function/FUNCTION.h.jinja
// Do NOT modify by hand.
// -----------------------------------------------------------------------------

#pragma once

#include <Eigen/Dense>

#include <sym/rot3.h>

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
 *     jacobian: (3x6) jacobian of res wrt args a (3), b (3)
 *     hessian: (6x6) Gauss-Newton hessian for args a (3), b (3)
 *     rhs: (6x1) Gauss-Newton rhs for args a (3), b (3)
 */
template <typename Scalar>
void BetweenFactorRot3(const sym::Rot3<Scalar>& a, const sym::Rot3<Scalar>& b,
                       const sym::Rot3<Scalar>& a_T_b, const Eigen::Matrix<Scalar, 3, 1>& sqrt_info,
                       const Scalar epsilon, Eigen::Matrix<Scalar, 3, 1>* const res = nullptr,
                       Eigen::Matrix<Scalar, 3, 6>* const jacobian = nullptr,
                       Eigen::Matrix<Scalar, 6, 6>* const hessian = nullptr,
                       Eigen::Matrix<Scalar, 6, 1>* const rhs = nullptr) {
  // Total ops: 475

  // Input arrays
  const Eigen::Matrix<Scalar, 4, 1>& _a = a.Data();
  const Eigen::Matrix<Scalar, 4, 1>& _b = b.Data();
  const Eigen::Matrix<Scalar, 4, 1>& _a_T_b = a_T_b.Data();

  // Intermediate terms (152)
  const Scalar _tmp0 = _a[3] * _b[3];
  const Scalar _tmp1 = _a[2] * _b[2];
  const Scalar _tmp2 = _a[0] * _b[0];
  const Scalar _tmp3 = _a[1] * _b[1];
  const Scalar _tmp4 = _tmp0 + _tmp1 + _tmp2 + _tmp3;
  const Scalar _tmp5 = _a_T_b[3] * _tmp4;
  const Scalar _tmp6 = _a[3] * _b[1];
  const Scalar _tmp7 = _a[2] * _b[0];
  const Scalar _tmp8 = _a[0] * _b[2];
  const Scalar _tmp9 = _a[1] * _b[3];
  const Scalar _tmp10 = _tmp6 - _tmp7 + _tmp8 - _tmp9;
  const Scalar _tmp11 = _a_T_b[1] * _tmp10;
  const Scalar _tmp12 = _a[3] * _b[0];
  const Scalar _tmp13 = _a[2] * _b[1];
  const Scalar _tmp14 = _a[0] * _b[3];
  const Scalar _tmp15 = _a[1] * _b[2];
  const Scalar _tmp16 = _tmp12 + _tmp13 - _tmp14 - _tmp15;
  const Scalar _tmp17 = _a_T_b[0] * _tmp16;
  const Scalar _tmp18 = _a[3] * _b[2];
  const Scalar _tmp19 = _a[2] * _b[3];
  const Scalar _tmp20 = _a[0] * _b[1];
  const Scalar _tmp21 = _a[1] * _b[0];
  const Scalar _tmp22 = _tmp18 - _tmp19 - _tmp20 + _tmp21;
  const Scalar _tmp23 = _a_T_b[2] * _tmp22;
  const Scalar _tmp24 = _tmp11 + _tmp17 + _tmp23 + _tmp5;
  const Scalar _tmp25 = 2 * std::min<Scalar>(0, (((_tmp24) > 0) - ((_tmp24) < 0))) + 1;
  const Scalar _tmp26 = 2 * _tmp25;
  const Scalar _tmp27 = 1 - epsilon;
  const Scalar _tmp28 = std::min<Scalar>(_tmp27, std::fabs(_tmp24));
  const Scalar _tmp29 = std::acos(_tmp28) / std::sqrt(Scalar(1 - std::pow(_tmp28, Scalar(2))));
  const Scalar _tmp30 = _tmp26 * _tmp29;
  const Scalar _tmp31 = sqrt_info(0, 0) * (-_a_T_b[0] * _tmp4 - _a_T_b[1] * _tmp22 +
                                           _a_T_b[2] * _tmp10 + _a_T_b[3] * _tmp16);
  const Scalar _tmp32 = _tmp30 * _tmp31;
  const Scalar _tmp33 = sqrt_info(1, 0) * (_a_T_b[0] * _tmp22 - _a_T_b[1] * _tmp4 -
                                           _a_T_b[2] * _tmp16 + _a_T_b[3] * _tmp10);
  const Scalar _tmp34 = _tmp30 * _tmp33;
  const Scalar _tmp35 =
      -_a_T_b[0] * _tmp10 + _a_T_b[1] * _tmp16 - _a_T_b[2] * _tmp4 + _a_T_b[3] * _tmp22;
  const Scalar _tmp36 = _tmp26 * sqrt_info(2, 0);
  const Scalar _tmp37 = _tmp29 * _tmp35 * _tmp36;
  const Scalar _tmp38 = (Scalar(1) / Scalar(2)) * _tmp0;
  const Scalar _tmp39 = (Scalar(1) / Scalar(2)) * _tmp1;
  const Scalar _tmp40 = (Scalar(1) / Scalar(2)) * _tmp2;
  const Scalar _tmp41 = (Scalar(1) / Scalar(2)) * _tmp3;
  const Scalar _tmp42 = -_tmp38 - _tmp39 - _tmp40 - _tmp41;
  const Scalar _tmp43 = _a_T_b[3] * _tmp42;
  const Scalar _tmp44 = (Scalar(1) / Scalar(2)) * _tmp6;
  const Scalar _tmp45 = (Scalar(1) / Scalar(2)) * _tmp7;
  const Scalar _tmp46 = (Scalar(1) / Scalar(2)) * _tmp8;
  const Scalar _tmp47 = (Scalar(1) / Scalar(2)) * _tmp9;
  const Scalar _tmp48 = -_tmp44 + _tmp45 - _tmp46 + _tmp47;
  const Scalar _tmp49 = -_a_T_b[1] * _tmp48;
  const Scalar _tmp50 = (Scalar(1) / Scalar(2)) * _tmp18;
  const Scalar _tmp51 = (Scalar(1) / Scalar(2)) * _tmp19;
  const Scalar _tmp52 = (Scalar(1) / Scalar(2)) * _tmp20;
  const Scalar _tmp53 = (Scalar(1) / Scalar(2)) * _tmp21;
  const Scalar _tmp54 = _tmp50 - _tmp51 - _tmp52 + _tmp53;
  const Scalar _tmp55 = _a_T_b[2] * _tmp54;
  const Scalar _tmp56 = (Scalar(1) / Scalar(2)) * _tmp12;
  const Scalar _tmp57 = (Scalar(1) / Scalar(2)) * _tmp13;
  const Scalar _tmp58 = (Scalar(1) / Scalar(2)) * _tmp14;
  const Scalar _tmp59 = (Scalar(1) / Scalar(2)) * _tmp15;
  const Scalar _tmp60 = _tmp56 + _tmp57 - _tmp58 - _tmp59;
  const Scalar _tmp61 = _a_T_b[0] * _tmp60;
  const Scalar _tmp62 = _tmp11 + _tmp17 + _tmp23 + _tmp5;
  const Scalar _tmp63 = std::fabs(_tmp62);
  const Scalar _tmp64 = std::min<Scalar>(_tmp27, _tmp63);
  const Scalar _tmp65 = 1 - std::pow(_tmp64, Scalar(2));
  const Scalar _tmp66 = std::acos(_tmp64);
  const Scalar _tmp67 = _tmp66 / std::sqrt(_tmp65);
  const Scalar _tmp68 = _tmp26 * _tmp67;
  const Scalar _tmp69 = _tmp68 * sqrt_info(0, 0);
  const Scalar _tmp70 = _a_T_b[3] * _tmp60;
  const Scalar _tmp71 = _a_T_b[0] * _tmp42;
  const Scalar _tmp72 = _a_T_b[2] * _tmp48;
  const Scalar _tmp73 = _a_T_b[1] * _tmp54;
  const Scalar _tmp74 = _tmp72 + _tmp73;
  const Scalar _tmp75 = _tmp70 + _tmp71 + _tmp74;
  const Scalar _tmp76 = _tmp25 * ((((_tmp27 - _tmp63) > 0) - ((_tmp27 - _tmp63) < 0)) + 1) *
                        (((_tmp62) > 0) - ((_tmp62) < 0));
  const Scalar _tmp77 = _tmp64 * _tmp66 * _tmp76 / (_tmp65 * std::sqrt(_tmp65));
  const Scalar _tmp78 = _tmp31 * _tmp77;
  const Scalar _tmp79 = _tmp76 / _tmp65;
  const Scalar _tmp80 = _tmp31 * _tmp79;
  const Scalar _tmp81 =
      _tmp69 * (_tmp43 + _tmp49 + _tmp55 - _tmp61) + _tmp75 * _tmp78 - _tmp75 * _tmp80;
  const Scalar _tmp82 = _tmp33 * _tmp79;
  const Scalar _tmp83 = _tmp33 * _tmp77;
  const Scalar _tmp84 = -_a_T_b[1] * _tmp60;
  const Scalar _tmp85 = _a_T_b[2] * _tmp42;
  const Scalar _tmp86 = _a_T_b[0] * _tmp48;
  const Scalar _tmp87 = _a_T_b[3] * _tmp54;
  const Scalar _tmp88 = _tmp86 + _tmp87;
  const Scalar _tmp89 = _tmp68 * sqrt_info(1, 0);
  const Scalar _tmp90 = -_tmp75 * _tmp82 + _tmp75 * _tmp83 + _tmp89 * (_tmp84 - _tmp85 + _tmp88);
  const Scalar _tmp91 = _tmp35 * sqrt_info(2, 0);
  const Scalar _tmp92 = _tmp79 * _tmp91;
  const Scalar _tmp93 = _tmp77 * _tmp91;
  const Scalar _tmp94 = _a_T_b[1] * _tmp42;
  const Scalar _tmp95 = _a_T_b[3] * _tmp48;
  const Scalar _tmp96 = -_a_T_b[0] * _tmp54;
  const Scalar _tmp97 = _a_T_b[2] * _tmp60;
  const Scalar _tmp98 = _tmp36 * _tmp67;
  const Scalar _tmp99 =
      -_tmp75 * _tmp92 + _tmp75 * _tmp93 + _tmp98 * (_tmp94 + _tmp95 + _tmp96 - _tmp97);
  const Scalar _tmp100 = _tmp44 - _tmp45 + _tmp46 - _tmp47;
  const Scalar _tmp101 = _a_T_b[0] * _tmp100;
  const Scalar _tmp102 = -_tmp50 + _tmp51 + _tmp52 - _tmp53;
  const Scalar _tmp103 = _a_T_b[3] * _tmp102;
  const Scalar _tmp104 = _a_T_b[3] * _tmp100;
  const Scalar _tmp105 = _a_T_b[0] * _tmp102;
  const Scalar _tmp106 = _tmp105 + _tmp97;
  const Scalar _tmp107 = _tmp104 + _tmp106 + _tmp94;
  const Scalar _tmp108 =
      _tmp107 * _tmp78 - _tmp107 * _tmp80 + _tmp69 * (-_tmp101 + _tmp103 + _tmp84 + _tmp85);
  const Scalar _tmp109 = _a_T_b[1] * _tmp100;
  const Scalar _tmp110 = -_a_T_b[2] * _tmp102;
  const Scalar _tmp111 = _tmp110 + _tmp61;
  const Scalar _tmp112 =
      -_tmp107 * _tmp82 + _tmp107 * _tmp83 + _tmp89 * (-_tmp109 + _tmp111 + _tmp43);
  const Scalar _tmp113 = -_a_T_b[2] * _tmp100;
  const Scalar _tmp114 = _a_T_b[1] * _tmp102;
  const Scalar _tmp115 = _tmp114 + _tmp70;
  const Scalar _tmp116 =
      -_tmp107 * _tmp92 + _tmp107 * _tmp93 + _tmp98 * (_tmp113 + _tmp115 - _tmp71);
  const Scalar _tmp117 = -_tmp56 - _tmp57 + _tmp58 + _tmp59;
  const Scalar _tmp118 = _a_T_b[2] * _tmp117;
  const Scalar _tmp119 = _tmp104 + _tmp118;
  const Scalar _tmp120 = _a_T_b[1] * _tmp117;
  const Scalar _tmp121 = _tmp101 + _tmp120;
  const Scalar _tmp122 = _tmp121 + _tmp85 + _tmp87;
  const Scalar _tmp123 = _tmp122 * _tmp78 - _tmp122 * _tmp80 + _tmp69 * (_tmp119 - _tmp94 + _tmp96);
  const Scalar _tmp124 = _a_T_b[3] * _tmp117;
  const Scalar _tmp125 =
      -_tmp122 * _tmp82 + _tmp122 * _tmp83 + _tmp89 * (_tmp113 + _tmp124 + _tmp71 - _tmp73);
  const Scalar _tmp126 = -_a_T_b[0] * _tmp117;
  const Scalar _tmp127 = _tmp109 + _tmp126;
  const Scalar _tmp128 =
      -_tmp122 * _tmp92 + _tmp122 * _tmp93 + _tmp98 * (_tmp127 + _tmp43 - _tmp55);
  const Scalar _tmp129 = _tmp38 + _tmp39 + _tmp40 + _tmp41;
  const Scalar _tmp130 = _a_T_b[0] * _tmp129;
  const Scalar _tmp131 = _tmp124 + _tmp130;
  const Scalar _tmp132 = _tmp131 + _tmp74;
  const Scalar _tmp133 = _tmp132 * _tmp79;
  const Scalar _tmp134 = _a_T_b[3] * _tmp129;
  const Scalar _tmp135 = _tmp134 + _tmp49;
  const Scalar _tmp136 =
      _tmp132 * _tmp78 - _tmp133 * _tmp31 + _tmp69 * (_tmp126 + _tmp135 + _tmp55);
  const Scalar _tmp137 = _a_T_b[2] * _tmp129;
  const Scalar _tmp138 =
      _tmp132 * _tmp83 - _tmp133 * _tmp33 + _tmp89 * (-_tmp120 - _tmp137 + _tmp88);
  const Scalar _tmp139 = _a_T_b[1] * _tmp129;
  const Scalar _tmp140 = _tmp139 + _tmp95;
  const Scalar _tmp141 =
      _tmp132 * _tmp93 - _tmp133 * _tmp91 + _tmp98 * (-_tmp118 + _tmp140 + _tmp96);
  const Scalar _tmp142 = _tmp106 + _tmp140;
  const Scalar _tmp143 = _tmp103 + _tmp137;
  const Scalar _tmp144 = _tmp142 * _tmp78 - _tmp142 * _tmp80 + _tmp69 * (_tmp143 + _tmp84 - _tmp86);
  const Scalar _tmp145 = -_tmp142 * _tmp82 + _tmp142 * _tmp83 + _tmp89 * (_tmp111 + _tmp135);
  const Scalar _tmp146 =
      -_tmp142 * _tmp92 + _tmp142 * _tmp93 + _tmp98 * (_tmp115 - _tmp130 - _tmp72);
  const Scalar _tmp147 = _tmp121 + _tmp143;
  const Scalar _tmp148 = _tmp147 * _tmp79;
  const Scalar _tmp149 =
      _tmp147 * _tmp78 - _tmp148 * _tmp31 + _tmp69 * (-_tmp105 + _tmp119 - _tmp139);
  const Scalar _tmp150 =
      _tmp147 * _tmp83 - _tmp148 * _tmp33 + _tmp89 * (_tmp113 - _tmp114 + _tmp131);
  const Scalar _tmp151 =
      _tmp147 * _tmp93 - _tmp148 * _tmp91 + _tmp98 * (_tmp110 + _tmp127 + _tmp134);

  // Output terms (4)
  if (res != nullptr) {
    Eigen::Matrix<Scalar, 3, 1>& _res = (*res);

    _res(0, 0) = _tmp32;
    _res(1, 0) = _tmp34;
    _res(2, 0) = _tmp37;
  }

  if (jacobian != nullptr) {
    Eigen::Matrix<Scalar, 3, 6>& _jacobian = (*jacobian);

    _jacobian(0, 0) = _tmp81;
    _jacobian(1, 0) = _tmp90;
    _jacobian(2, 0) = _tmp99;
    _jacobian(0, 1) = _tmp108;
    _jacobian(1, 1) = _tmp112;
    _jacobian(2, 1) = _tmp116;
    _jacobian(0, 2) = _tmp123;
    _jacobian(1, 2) = _tmp125;
    _jacobian(2, 2) = _tmp128;
    _jacobian(0, 3) = _tmp136;
    _jacobian(1, 3) = _tmp138;
    _jacobian(2, 3) = _tmp141;
    _jacobian(0, 4) = _tmp144;
    _jacobian(1, 4) = _tmp145;
    _jacobian(2, 4) = _tmp146;
    _jacobian(0, 5) = _tmp149;
    _jacobian(1, 5) = _tmp150;
    _jacobian(2, 5) = _tmp151;
  }

  if (hessian != nullptr) {
    Eigen::Matrix<Scalar, 6, 6>& _hessian = (*hessian);

    _hessian(0, 0) =
        std::pow(_tmp81, Scalar(2)) + std::pow(_tmp90, Scalar(2)) + std::pow(_tmp99, Scalar(2));
    _hessian(1, 0) = _tmp108 * _tmp81 + _tmp112 * _tmp90 + _tmp116 * _tmp99;
    _hessian(2, 0) = _tmp123 * _tmp81 + _tmp125 * _tmp90 + _tmp128 * _tmp99;
    _hessian(3, 0) = _tmp136 * _tmp81 + _tmp138 * _tmp90 + _tmp141 * _tmp99;
    _hessian(4, 0) = _tmp144 * _tmp81 + _tmp145 * _tmp90 + _tmp146 * _tmp99;
    _hessian(5, 0) = _tmp149 * _tmp81 + _tmp150 * _tmp90 + _tmp151 * _tmp99;
    _hessian(0, 1) = 0;
    _hessian(1, 1) =
        std::pow(_tmp108, Scalar(2)) + std::pow(_tmp112, Scalar(2)) + std::pow(_tmp116, Scalar(2));
    _hessian(2, 1) = _tmp108 * _tmp123 + _tmp112 * _tmp125 + _tmp116 * _tmp128;
    _hessian(3, 1) = _tmp108 * _tmp136 + _tmp112 * _tmp138 + _tmp116 * _tmp141;
    _hessian(4, 1) = _tmp108 * _tmp144 + _tmp112 * _tmp145 + _tmp116 * _tmp146;
    _hessian(5, 1) = _tmp108 * _tmp149 + _tmp112 * _tmp150 + _tmp116 * _tmp151;
    _hessian(0, 2) = 0;
    _hessian(1, 2) = 0;
    _hessian(2, 2) =
        std::pow(_tmp123, Scalar(2)) + std::pow(_tmp125, Scalar(2)) + std::pow(_tmp128, Scalar(2));
    _hessian(3, 2) = _tmp123 * _tmp136 + _tmp125 * _tmp138 + _tmp128 * _tmp141;
    _hessian(4, 2) = _tmp123 * _tmp144 + _tmp125 * _tmp145 + _tmp128 * _tmp146;
    _hessian(5, 2) = _tmp123 * _tmp149 + _tmp125 * _tmp150 + _tmp128 * _tmp151;
    _hessian(0, 3) = 0;
    _hessian(1, 3) = 0;
    _hessian(2, 3) = 0;
    _hessian(3, 3) =
        std::pow(_tmp136, Scalar(2)) + std::pow(_tmp138, Scalar(2)) + std::pow(_tmp141, Scalar(2));
    _hessian(4, 3) = _tmp136 * _tmp144 + _tmp138 * _tmp145 + _tmp141 * _tmp146;
    _hessian(5, 3) = _tmp136 * _tmp149 + _tmp138 * _tmp150 + _tmp141 * _tmp151;
    _hessian(0, 4) = 0;
    _hessian(1, 4) = 0;
    _hessian(2, 4) = 0;
    _hessian(3, 4) = 0;
    _hessian(4, 4) =
        std::pow(_tmp144, Scalar(2)) + std::pow(_tmp145, Scalar(2)) + std::pow(_tmp146, Scalar(2));
    _hessian(5, 4) = _tmp144 * _tmp149 + _tmp145 * _tmp150 + _tmp146 * _tmp151;
    _hessian(0, 5) = 0;
    _hessian(1, 5) = 0;
    _hessian(2, 5) = 0;
    _hessian(3, 5) = 0;
    _hessian(4, 5) = 0;
    _hessian(5, 5) =
        std::pow(_tmp149, Scalar(2)) + std::pow(_tmp150, Scalar(2)) + std::pow(_tmp151, Scalar(2));
  }

  if (rhs != nullptr) {
    Eigen::Matrix<Scalar, 6, 1>& _rhs = (*rhs);

    _rhs(0, 0) = _tmp32 * _tmp81 + _tmp34 * _tmp90 + _tmp37 * _tmp99;
    _rhs(1, 0) = _tmp108 * _tmp32 + _tmp112 * _tmp34 + _tmp116 * _tmp37;
    _rhs(2, 0) = _tmp123 * _tmp32 + _tmp125 * _tmp34 + _tmp128 * _tmp37;
    _rhs(3, 0) = _tmp136 * _tmp32 + _tmp138 * _tmp34 + _tmp141 * _tmp37;
    _rhs(4, 0) = _tmp144 * _tmp32 + _tmp145 * _tmp34 + _tmp146 * _tmp37;
    _rhs(5, 0) = _tmp149 * _tmp32 + _tmp150 * _tmp34 + _tmp151 * _tmp37;
  }
}  // NOLINT(readability/fn_size)

// NOLINTNEXTLINE(readability/fn_size)
}  // namespace sym
