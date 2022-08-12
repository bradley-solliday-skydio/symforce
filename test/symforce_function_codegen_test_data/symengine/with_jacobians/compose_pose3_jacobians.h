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
 *     res_D_a: (6x6) jacobian of res (6) wrt arg a (6)
 *     res_D_b: (6x6) jacobian of res (6) wrt arg b (6)
 */
template <typename Scalar>
void ComposePose3Jacobians(const sym::Pose3<Scalar>& a, const sym::Pose3<Scalar>& b,
                           Eigen::Matrix<Scalar, 6, 6>* const res_D_a = nullptr,
                           Eigen::Matrix<Scalar, 6, 6>* const res_D_b = nullptr) {
  // Total ops: 227

  // Input arrays
  const Eigen::Matrix<Scalar, 7, 1>& _a = a.Data();
  const Eigen::Matrix<Scalar, 7, 1>& _b = b.Data();

  // Intermediate terms (84)
  const Scalar _tmp0 = _a[0] * _b[2];
  const Scalar _tmp1 = -_tmp0;
  const Scalar _tmp2 = _a[1] * _b[3];
  const Scalar _tmp3 = -_tmp2;
  const Scalar _tmp4 = _a[3] * _b[1];
  const Scalar _tmp5 = _a[2] * _b[0];
  const Scalar _tmp6 = _tmp4 - _tmp5;
  const Scalar _tmp7 = _tmp1 + _tmp3 + _tmp6;
  const Scalar _tmp8 = _tmp4 + _tmp5;
  const Scalar _tmp9 = _tmp1 + _tmp2 + _tmp8;
  const Scalar _tmp10 = _a[2] * _b[3];
  const Scalar _tmp11 = _a[3] * _b[2];
  const Scalar _tmp12 = _a[0] * _b[1];
  const Scalar _tmp13 = _a[1] * _b[0];
  const Scalar _tmp14 = _tmp12 - _tmp13;
  const Scalar _tmp15 = _tmp10 + _tmp11 + _tmp14;
  const Scalar _tmp16 = -_tmp10;
  const Scalar _tmp17 = _tmp12 + _tmp13;
  const Scalar _tmp18 = _tmp11 + _tmp16 + _tmp17;
  const Scalar _tmp19 = _a[3] * _b[0];
  const Scalar _tmp20 = -_tmp19;
  const Scalar _tmp21 = _a[0] * _b[3];
  const Scalar _tmp22 = -_tmp21;
  const Scalar _tmp23 = _a[2] * _b[1];
  const Scalar _tmp24 = _a[1] * _b[2];
  const Scalar _tmp25 = _tmp23 - _tmp24;
  const Scalar _tmp26 = _tmp20 + _tmp22 + _tmp25;
  const Scalar _tmp27 = _tmp19 + _tmp21 + _tmp25;
  const Scalar _tmp28 = _a[3] * _b[3];
  const Scalar _tmp29 = _a[0] * _b[0];
  const Scalar _tmp30 = -_tmp29;
  const Scalar _tmp31 = _a[1] * _b[1];
  const Scalar _tmp32 = _a[2] * _b[2];
  const Scalar _tmp33 = _tmp31 + _tmp32;
  const Scalar _tmp34 = _tmp28 + _tmp30 + _tmp33;
  const Scalar _tmp35 = _tmp29 + _tmp33;
  const Scalar _tmp36 = -_tmp28;
  const Scalar _tmp37 = _tmp35 + _tmp36;
  const Scalar _tmp38 = _a[0] * _a[2];
  const Scalar _tmp39 = _a[1] * _a[3];
  const Scalar _tmp40 = _tmp38 + _tmp39;
  const Scalar _tmp41 = _a[0] * _a[1];
  const Scalar _tmp42 = _a[2] * _a[3];
  const Scalar _tmp43 = _tmp41 - _tmp42;
  const Scalar _tmp44 = std::pow(_a[0], Scalar(2));
  const Scalar _tmp45 = std::pow(_a[3], Scalar(2));
  const Scalar _tmp46 = -_tmp45;
  const Scalar _tmp47 = std::pow(_a[2], Scalar(2));
  const Scalar _tmp48 = std::pow(_a[1], Scalar(2));
  const Scalar _tmp49 = _tmp47 - _tmp48;
  const Scalar _tmp50 = _tmp44 + _tmp46 + _tmp49;
  const Scalar _tmp51 = _a[1] * _a[2];
  const Scalar _tmp52 = _a[0] * _a[3];
  const Scalar _tmp53 = _tmp51 - _tmp52;
  const Scalar _tmp54 = 2 * _tmp53;
  const Scalar _tmp55 = -_tmp44;
  const Scalar _tmp56 = _tmp45 + _tmp49 + _tmp55;
  const Scalar _tmp57 = _tmp51 + _tmp52;
  const Scalar _tmp58 = 2 * _tmp57;
  const Scalar _tmp59 = _tmp31 - _tmp32;
  const Scalar _tmp60 = _tmp30 + _tmp36 + _tmp59;
  const Scalar _tmp61 = _tmp23 + _tmp24;
  const Scalar _tmp62 = _tmp19 + _tmp22 + _tmp61;
  const Scalar _tmp63 = _tmp0 + _tmp2 + _tmp6;
  const Scalar _tmp64 = -_tmp11;
  const Scalar _tmp65 = _tmp10 + _tmp17 + _tmp64;
  const Scalar _tmp66 = _tmp47 + _tmp48;
  const Scalar _tmp67 = _tmp46 + _tmp55 + _tmp66;
  const Scalar _tmp68 = 2 * _tmp40;
  const Scalar _tmp69 = _tmp41 + _tmp42;
  const Scalar _tmp70 = _tmp38 - _tmp39;
  const Scalar _tmp71 = 2 * _tmp70;
  const Scalar _tmp72 = _tmp28 + _tmp29 + _tmp59;
  const Scalar _tmp73 = _tmp14 + _tmp16 + _tmp64;
  const Scalar _tmp74 = _tmp20 + _tmp21 + _tmp61;
  const Scalar _tmp75 = _tmp0 + _tmp3 + _tmp8;
  const Scalar _tmp76 = 2 * _tmp43;
  const Scalar _tmp77 = 2 * _tmp69;
  const Scalar _tmp78 = _tmp35 + _tmp36;
  const Scalar _tmp79 = std::pow(_tmp15, Scalar(2)) + std::pow(_tmp26, Scalar(2)) +
                        _tmp37 * _tmp78 + std::pow(_tmp9, Scalar(2));
  const Scalar _tmp80 = -_tmp15 * _tmp37 + _tmp15 * _tmp78;
  const Scalar _tmp81 = -_tmp37 * _tmp9 + _tmp78 * _tmp9;
  const Scalar _tmp82 = -_tmp26 * _tmp37 + _tmp26 * _tmp78;
  const Scalar _tmp83 = _tmp44 + Scalar(-1) / Scalar(2);

  // Output terms (2)
  if (res_D_a != nullptr) {
    Eigen::Matrix<Scalar, 6, 6>& _res_D_a = (*res_D_a);

    _res_D_a(0, 0) = -_tmp15 * _tmp18 - _tmp26 * _tmp27 - _tmp34 * _tmp37 - _tmp7 * _tmp9;
    _res_D_a(1, 0) = -_tmp15 * _tmp34 + _tmp18 * _tmp37 - _tmp26 * _tmp7 + _tmp27 * _tmp9;
    _res_D_a(2, 0) = _tmp15 * _tmp27 - _tmp18 * _tmp26 + _tmp34 * _tmp9 - _tmp37 * _tmp7;
    _res_D_a(3, 0) = 2 * _b[5] * _tmp40 - 2 * _b[6] * _tmp43;
    _res_D_a(4, 0) = _b[5] * _tmp54 + _b[6] * _tmp50;
    _res_D_a(5, 0) = _b[5] * _tmp56 - _b[6] * _tmp58;
    _res_D_a(0, 1) = -_tmp15 * _tmp60 - _tmp26 * _tmp63 + _tmp37 * _tmp65 + _tmp62 * _tmp9;
    _res_D_a(1, 1) = _tmp15 * _tmp65 + _tmp26 * _tmp62 + _tmp37 * _tmp60 + _tmp63 * _tmp9;
    _res_D_a(2, 1) = _tmp15 * _tmp63 - _tmp26 * _tmp60 + _tmp37 * _tmp62 - _tmp65 * _tmp9;
    _res_D_a(3, 1) = -_b[4] * _tmp68 - _b[6] * _tmp67;
    _res_D_a(4, 1) = -2 * _b[4] * _tmp53 + 2 * _b[6] * _tmp69;
    _res_D_a(5, 1) = -_b[4] * _tmp56 + _b[6] * _tmp71;
    _res_D_a(0, 2) = -_tmp15 * _tmp74 + _tmp26 * _tmp73 + _tmp37 * _tmp75 - _tmp72 * _tmp9;
    _res_D_a(1, 2) = _tmp15 * _tmp75 - _tmp26 * _tmp72 + _tmp37 * _tmp74 - _tmp73 * _tmp9;
    _res_D_a(2, 2) = -_tmp15 * _tmp73 - _tmp26 * _tmp74 - _tmp37 * _tmp72 - _tmp75 * _tmp9;
    _res_D_a(3, 2) = _b[4] * _tmp76 + _b[5] * _tmp67;
    _res_D_a(4, 2) = -_b[4] * _tmp50 - _b[5] * _tmp77;
    _res_D_a(5, 2) = 2 * _b[4] * _tmp57 - 2 * _b[5] * _tmp70;
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

    _res_D_b(0, 0) = _tmp79;
    _res_D_b(1, 0) = _tmp80;
    _res_D_b(2, 0) = -_tmp81;
    _res_D_b(3, 0) = 0;
    _res_D_b(4, 0) = 0;
    _res_D_b(5, 0) = 0;
    _res_D_b(0, 1) = -_tmp80;
    _res_D_b(1, 1) = _tmp79;
    _res_D_b(2, 1) = -_tmp82;
    _res_D_b(3, 1) = 0;
    _res_D_b(4, 1) = 0;
    _res_D_b(5, 1) = 0;
    _res_D_b(0, 2) = _tmp81;
    _res_D_b(1, 2) = _tmp82;
    _res_D_b(2, 2) = _tmp79;
    _res_D_b(3, 2) = 0;
    _res_D_b(4, 2) = 0;
    _res_D_b(5, 2) = 0;
    _res_D_b(0, 3) = 0;
    _res_D_b(1, 3) = 0;
    _res_D_b(2, 3) = 0;
    _res_D_b(3, 3) = 1 - 2 * _tmp66;
    _res_D_b(4, 3) = _tmp77;
    _res_D_b(5, 3) = _tmp71;
    _res_D_b(0, 4) = 0;
    _res_D_b(1, 4) = 0;
    _res_D_b(2, 4) = 0;
    _res_D_b(3, 4) = _tmp76;
    _res_D_b(4, 4) = -2 * _tmp47 - 2 * _tmp83;
    _res_D_b(5, 4) = _tmp58;
    _res_D_b(0, 5) = 0;
    _res_D_b(1, 5) = 0;
    _res_D_b(2, 5) = 0;
    _res_D_b(3, 5) = _tmp68;
    _res_D_b(4, 5) = _tmp54;
    _res_D_b(5, 5) = -2 * _tmp48 - 2 * _tmp83;
  }
}  // NOLINT(readability/fn_size)

// NOLINTNEXTLINE(readability/fn_size)
}  // namespace sym
