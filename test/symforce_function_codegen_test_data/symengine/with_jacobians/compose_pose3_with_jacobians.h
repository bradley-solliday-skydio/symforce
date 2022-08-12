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
 * Composition of two elements in the group.
 *
 * Returns:
 *     Element: a @ b
 *     res_D_a: (6x6) jacobian of res (6) wrt arg a (6)
 *     res_D_b: (6x6) jacobian of res (6) wrt arg b (6)
 */
template <typename Scalar>
sym::Pose3<Scalar> ComposePose3WithJacobians(const sym::Pose3<Scalar>& a,
                                             const sym::Pose3<Scalar>& b,
                                             Eigen::Matrix<Scalar, 6, 6>* const res_D_a = nullptr,
                                             Eigen::Matrix<Scalar, 6, 6>* const res_D_b = nullptr) {
  // Total ops: 270

  // Input arrays
  const Eigen::Matrix<Scalar, 7, 1>& _a = a.Data();
  const Eigen::Matrix<Scalar, 7, 1>& _b = b.Data();

  // Intermediate terms (94)
  const Scalar _tmp0 = _a[1] * _b[2];
  const Scalar _tmp1 = _a[2] * _b[1];
  const Scalar _tmp2 = _a[0] * _b[3];
  const Scalar _tmp3 = _a[3] * _b[0];
  const Scalar _tmp4 = _tmp2 + _tmp3;
  const Scalar _tmp5 = _a[1] * _b[3];
  const Scalar _tmp6 = _a[2] * _b[0];
  const Scalar _tmp7 = _a[3] * _b[1];
  const Scalar _tmp8 = _a[0] * _b[2];
  const Scalar _tmp9 = _tmp7 - _tmp8;
  const Scalar _tmp10 = _tmp5 + _tmp6 + _tmp9;
  const Scalar _tmp11 = _a[1] * _b[0];
  const Scalar _tmp12 = -_tmp11;
  const Scalar _tmp13 = _a[2] * _b[3];
  const Scalar _tmp14 = _a[0] * _b[1];
  const Scalar _tmp15 = _a[3] * _b[2];
  const Scalar _tmp16 = _tmp14 + _tmp15;
  const Scalar _tmp17 = _tmp12 + _tmp13 + _tmp16;
  const Scalar _tmp18 = _a[0] * _b[0];
  const Scalar _tmp19 = _a[1] * _b[1];
  const Scalar _tmp20 = _a[2] * _b[2];
  const Scalar _tmp21 = _tmp19 + _tmp20;
  const Scalar _tmp22 = _tmp18 + _tmp21;
  const Scalar _tmp23 = _a[3] * _b[3];
  const Scalar _tmp24 = std::pow(_a[1], Scalar(2));
  const Scalar _tmp25 = -2 * _tmp24;
  const Scalar _tmp26 = std::pow(_a[2], Scalar(2));
  const Scalar _tmp27 = 1 - 2 * _tmp26;
  const Scalar _tmp28 = _a[0] * _a[2];
  const Scalar _tmp29 = 2 * _tmp28;
  const Scalar _tmp30 = _a[1] * _a[3];
  const Scalar _tmp31 = 2 * _tmp30;
  const Scalar _tmp32 = _a[0] * _a[1];
  const Scalar _tmp33 = 2 * _tmp32;
  const Scalar _tmp34 = _a[2] * _a[3];
  const Scalar _tmp35 = 2 * _tmp34;
  const Scalar _tmp36 = std::pow(_a[0], Scalar(2));
  const Scalar _tmp37 = -2 * _tmp36;
  const Scalar _tmp38 = _a[1] * _a[2];
  const Scalar _tmp39 = 2 * _tmp38;
  const Scalar _tmp40 = _a[0] * _a[3];
  const Scalar _tmp41 = 2 * _tmp40;
  const Scalar _tmp42 = -_tmp5;
  const Scalar _tmp43 = -_tmp6;
  const Scalar _tmp44 = _tmp42 + _tmp43 + _tmp9;
  const Scalar _tmp45 = -_tmp13;
  const Scalar _tmp46 = _tmp11 + _tmp16 + _tmp45;
  const Scalar _tmp47 = -_tmp2;
  const Scalar _tmp48 = -_tmp3;
  const Scalar _tmp49 = -_tmp0 + _tmp1;
  const Scalar _tmp50 = _tmp47 + _tmp48 + _tmp49;
  const Scalar _tmp51 = _tmp4 + _tmp49;
  const Scalar _tmp52 = -_tmp18;
  const Scalar _tmp53 = _tmp21 + _tmp23 + _tmp52;
  const Scalar _tmp54 = -_tmp23;
  const Scalar _tmp55 = _tmp22 + _tmp54;
  const Scalar _tmp56 = _tmp28 + _tmp30;
  const Scalar _tmp57 = _tmp32 - _tmp34;
  const Scalar _tmp58 = std::pow(_a[3], Scalar(2));
  const Scalar _tmp59 = -_tmp58;
  const Scalar _tmp60 = -_tmp24 + _tmp26;
  const Scalar _tmp61 = _tmp36 + _tmp59 + _tmp60;
  const Scalar _tmp62 = _tmp38 - _tmp40;
  const Scalar _tmp63 = 2 * _tmp62;
  const Scalar _tmp64 = -_tmp36;
  const Scalar _tmp65 = _tmp58 + _tmp60 + _tmp64;
  const Scalar _tmp66 = _tmp38 + _tmp40;
  const Scalar _tmp67 = 2 * _tmp66;
  const Scalar _tmp68 = _tmp19 - _tmp20;
  const Scalar _tmp69 = _tmp52 + _tmp54 + _tmp68;
  const Scalar _tmp70 = _tmp0 + _tmp1;
  const Scalar _tmp71 = _tmp3 + _tmp47 + _tmp70;
  const Scalar _tmp72 = _tmp7 + _tmp8;
  const Scalar _tmp73 = _tmp43 + _tmp5 + _tmp72;
  const Scalar _tmp74 = _tmp14 - _tmp15;
  const Scalar _tmp75 = _tmp11 + _tmp13 + _tmp74;
  const Scalar _tmp76 = _tmp24 + _tmp26;
  const Scalar _tmp77 = _tmp59 + _tmp64 + _tmp76;
  const Scalar _tmp78 = 2 * _tmp56;
  const Scalar _tmp79 = _tmp32 + _tmp34;
  const Scalar _tmp80 = _tmp28 - _tmp30;
  const Scalar _tmp81 = 2 * _tmp80;
  const Scalar _tmp82 = _tmp18 + _tmp23 + _tmp68;
  const Scalar _tmp83 = _tmp12 + _tmp45 + _tmp74;
  const Scalar _tmp84 = _tmp2 + _tmp48 + _tmp70;
  const Scalar _tmp85 = _tmp42 + _tmp6 + _tmp72;
  const Scalar _tmp86 = 2 * _tmp57;
  const Scalar _tmp87 = 2 * _tmp79;
  const Scalar _tmp88 = _tmp22 + _tmp54;
  const Scalar _tmp89 = std::pow(_tmp10, Scalar(2)) + std::pow(_tmp17, Scalar(2)) +
                        std::pow(_tmp50, Scalar(2)) + _tmp55 * _tmp88;
  const Scalar _tmp90 = -_tmp17 * _tmp55 + _tmp17 * _tmp88;
  const Scalar _tmp91 = -_tmp10 * _tmp55 + _tmp10 * _tmp88;
  const Scalar _tmp92 = -_tmp50 * _tmp55 + _tmp50 * _tmp88;
  const Scalar _tmp93 = _tmp36 + Scalar(-1) / Scalar(2);

  // Output terms (3)
  Eigen::Matrix<Scalar, 7, 1> _res;

  _res[0] = _tmp0 - _tmp1 + _tmp4;
  _res[1] = _tmp10;
  _res[2] = _tmp17;
  _res[3] = -_tmp22 + _tmp23;
  _res[4] =
      _a[4] + _b[4] * (_tmp25 + _tmp27) + _b[5] * (_tmp33 - _tmp35) + _b[6] * (_tmp29 + _tmp31);
  _res[5] =
      _a[5] + _b[4] * (_tmp33 + _tmp35) + _b[5] * (_tmp27 + _tmp37) + _b[6] * (_tmp39 - _tmp41);
  _res[6] =
      _a[6] + _b[4] * (_tmp29 - _tmp31) + _b[5] * (_tmp39 + _tmp41) + _b[6] * (_tmp25 + _tmp37 + 1);

  if (res_D_a != nullptr) {
    Eigen::Matrix<Scalar, 6, 6>& _res_D_a = (*res_D_a);

    _res_D_a(0, 0) = -_tmp10 * _tmp44 - _tmp17 * _tmp46 - _tmp50 * _tmp51 - _tmp53 * _tmp55;
    _res_D_a(1, 0) = _tmp10 * _tmp51 - _tmp17 * _tmp53 - _tmp44 * _tmp50 + _tmp46 * _tmp55;
    _res_D_a(2, 0) = _tmp10 * _tmp53 + _tmp17 * _tmp51 - _tmp44 * _tmp55 - _tmp46 * _tmp50;
    _res_D_a(3, 0) = 2 * _b[5] * _tmp56 - 2 * _b[6] * _tmp57;
    _res_D_a(4, 0) = _b[5] * _tmp63 + _b[6] * _tmp61;
    _res_D_a(5, 0) = _b[5] * _tmp65 - _b[6] * _tmp67;
    _res_D_a(0, 1) = _tmp10 * _tmp71 - _tmp17 * _tmp69 - _tmp50 * _tmp73 + _tmp55 * _tmp75;
    _res_D_a(1, 1) = _tmp10 * _tmp73 + _tmp17 * _tmp75 + _tmp50 * _tmp71 + _tmp55 * _tmp69;
    _res_D_a(2, 1) = -_tmp10 * _tmp75 + _tmp17 * _tmp73 - _tmp50 * _tmp69 + _tmp55 * _tmp71;
    _res_D_a(3, 1) = -_b[4] * _tmp78 - _b[6] * _tmp77;
    _res_D_a(4, 1) = -2 * _b[4] * _tmp62 + 2 * _b[6] * _tmp79;
    _res_D_a(5, 1) = -_b[4] * _tmp65 + _b[6] * _tmp81;
    _res_D_a(0, 2) = -_tmp10 * _tmp82 - _tmp17 * _tmp84 + _tmp50 * _tmp83 + _tmp55 * _tmp85;
    _res_D_a(1, 2) = -_tmp10 * _tmp83 + _tmp17 * _tmp85 - _tmp50 * _tmp82 + _tmp55 * _tmp84;
    _res_D_a(2, 2) = -_tmp10 * _tmp85 - _tmp17 * _tmp83 - _tmp50 * _tmp84 - _tmp55 * _tmp82;
    _res_D_a(3, 2) = _b[4] * _tmp86 + _b[5] * _tmp77;
    _res_D_a(4, 2) = -_b[4] * _tmp61 - _b[5] * _tmp87;
    _res_D_a(5, 2) = 2 * _b[4] * _tmp66 - 2 * _b[5] * _tmp80;
    _res_D_a(0, 3) = 0;
    _res_D_a(1, 3) = 0;
    _res_D_a(2, 3) = 0;
    _res_D_a(3, 3) = 1;
    _res_D_a(4, 3) = 0;
    _res_D_a(5, 3) = 0;
    _res_D_a(0, 4) = 0;
    _res_D_a(1, 4) = 0;
    _res_D_a(2, 4) = 0;
    _res_D_a(3, 4) = 0;
    _res_D_a(4, 4) = 1;
    _res_D_a(5, 4) = 0;
    _res_D_a(0, 5) = 0;
    _res_D_a(1, 5) = 0;
    _res_D_a(2, 5) = 0;
    _res_D_a(3, 5) = 0;
    _res_D_a(4, 5) = 0;
    _res_D_a(5, 5) = 1;
  }

  if (res_D_b != nullptr) {
    Eigen::Matrix<Scalar, 6, 6>& _res_D_b = (*res_D_b);

    _res_D_b(0, 0) = _tmp89;
    _res_D_b(1, 0) = _tmp90;
    _res_D_b(2, 0) = -_tmp91;
    _res_D_b(3, 0) = 0;
    _res_D_b(4, 0) = 0;
    _res_D_b(5, 0) = 0;
    _res_D_b(0, 1) = -_tmp90;
    _res_D_b(1, 1) = _tmp89;
    _res_D_b(2, 1) = -_tmp92;
    _res_D_b(3, 1) = 0;
    _res_D_b(4, 1) = 0;
    _res_D_b(5, 1) = 0;
    _res_D_b(0, 2) = _tmp91;
    _res_D_b(1, 2) = _tmp92;
    _res_D_b(2, 2) = _tmp89;
    _res_D_b(3, 2) = 0;
    _res_D_b(4, 2) = 0;
    _res_D_b(5, 2) = 0;
    _res_D_b(0, 3) = 0;
    _res_D_b(1, 3) = 0;
    _res_D_b(2, 3) = 0;
    _res_D_b(3, 3) = 1 - 2 * _tmp76;
    _res_D_b(4, 3) = _tmp87;
    _res_D_b(5, 3) = _tmp81;
    _res_D_b(0, 4) = 0;
    _res_D_b(1, 4) = 0;
    _res_D_b(2, 4) = 0;
    _res_D_b(3, 4) = _tmp86;
    _res_D_b(4, 4) = -2 * _tmp26 - 2 * _tmp93;
    _res_D_b(5, 4) = _tmp67;
    _res_D_b(0, 5) = 0;
    _res_D_b(1, 5) = 0;
    _res_D_b(2, 5) = 0;
    _res_D_b(3, 5) = _tmp78;
    _res_D_b(4, 5) = _tmp63;
    _res_D_b(5, 5) = -2 * _tmp24 - 2 * _tmp93;
  }

  return sym::Pose3<Scalar>(_res);
}  // NOLINT(readability/fn_size)

// NOLINTNEXTLINE(readability/fn_size)
}  // namespace sym
