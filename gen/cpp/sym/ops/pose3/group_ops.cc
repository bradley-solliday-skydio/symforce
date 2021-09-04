// -----------------------------------------------------------------------------
// This file was autogenerated by symforce from template:
//     cpp_templates/geo_package/ops/CLASS/group_ops.cc.jinja
// Do NOT modify by hand.
// -----------------------------------------------------------------------------

#include "./group_ops.h"

namespace sym {

/**
 *
 * This function was autogenerated from a symbolic function. Do not modify by hand.
 *
 * Symbolic function: <lambda>
 *
 * Args:
 *
 * Outputs:
 *     res: Pose3
 *
 */
template <typename Scalar>
sym::Pose3<Scalar> GroupOps<Pose3<Scalar>>::Identity() {
  // Total ops: 0

  // Input arrays

  // Intermediate terms (0)

  // Output terms (1)
  Eigen::Matrix<Scalar, 7, 1> _res;

  _res[0] = 0;
  _res[1] = 0;
  _res[2] = 0;
  _res[3] = 1;
  _res[4] = 0;
  _res[5] = 0;
  _res[6] = 0;

  return sym::Pose3<Scalar>(_res);
}

/**
 *
 * Inverse of the element a.
 *
 * Returns:
 *     Element: b such that a @ b = identity
 *
 */
template <typename Scalar>
sym::Pose3<Scalar> GroupOps<Pose3<Scalar>>::Inverse(const sym::Pose3<Scalar>& a) {
  // Total ops: 50

  // Input arrays
  const Eigen::Matrix<Scalar, 7, 1>& _a = a.Data();

  // Intermediate terms (11)
  const Scalar _tmp0 = -2 * std::pow(_a[1], Scalar(2));
  const Scalar _tmp1 = 1 - 2 * std::pow(_a[2], Scalar(2));
  const Scalar _tmp2 = 2 * _a[0];
  const Scalar _tmp3 = _a[1] * _tmp2;
  const Scalar _tmp4 = 2 * _a[2];
  const Scalar _tmp5 = _a[3] * _tmp4;
  const Scalar _tmp6 = 2 * _a[1] * _a[3];
  const Scalar _tmp7 = _a[2] * _tmp2;
  const Scalar _tmp8 = -2 * std::pow(_a[0], Scalar(2));
  const Scalar _tmp9 = _a[3] * _tmp2;
  const Scalar _tmp10 = _a[1] * _tmp4;

  // Output terms (1)
  Eigen::Matrix<Scalar, 7, 1> _res;

  _res[0] = -_a[0];
  _res[1] = -_a[1];
  _res[2] = -_a[2];
  _res[3] = _a[3];
  _res[4] = -_a[4] * (_tmp0 + _tmp1) - _a[5] * (_tmp3 + _tmp5) - _a[6] * (-_tmp6 + _tmp7);
  _res[5] = -_a[4] * (_tmp3 - _tmp5) - _a[5] * (_tmp1 + _tmp8) - _a[6] * (_tmp10 + _tmp9);
  _res[6] = -_a[4] * (_tmp6 + _tmp7) - _a[5] * (_tmp10 - _tmp9) - _a[6] * (_tmp0 + _tmp8 + 1);

  return sym::Pose3<Scalar>(_res);
}

/**
 *
 * Composition of two elements in the group.
 *
 * Returns:
 *     Element: a @ b
 *
 */
template <typename Scalar>
sym::Pose3<Scalar> GroupOps<Pose3<Scalar>>::Compose(const sym::Pose3<Scalar>& a,
                                                    const sym::Pose3<Scalar>& b) {
  // Total ops: 79

  // Input arrays
  const Eigen::Matrix<Scalar, 7, 1>& _a = a.Data();
  const Eigen::Matrix<Scalar, 7, 1>& _b = b.Data();

  // Intermediate terms (11)
  const Scalar _tmp0 = -2 * std::pow(_a[1], Scalar(2));
  const Scalar _tmp1 = -2 * std::pow(_a[2], Scalar(2));
  const Scalar _tmp2 = 2 * _a[0] * _a[1];
  const Scalar _tmp3 = 2 * _a[3];
  const Scalar _tmp4 = _a[2] * _tmp3;
  const Scalar _tmp5 = _a[1] * _tmp3;
  const Scalar _tmp6 = 2 * _a[2];
  const Scalar _tmp7 = _a[0] * _tmp6;
  const Scalar _tmp8 = 1 - 2 * std::pow(_a[0], Scalar(2));
  const Scalar _tmp9 = _a[0] * _tmp3;
  const Scalar _tmp10 = _a[1] * _tmp6;

  // Output terms (1)
  Eigen::Matrix<Scalar, 7, 1> _res;

  _res[0] = _a[0] * _b[3] + _a[1] * _b[2] - _a[2] * _b[1] + _a[3] * _b[0];
  _res[1] = -_a[0] * _b[2] + _a[1] * _b[3] + _a[2] * _b[0] + _a[3] * _b[1];
  _res[2] = _a[0] * _b[1] - _a[1] * _b[0] + _a[2] * _b[3] + _a[3] * _b[2];
  _res[3] = -_a[0] * _b[0] - _a[1] * _b[1] - _a[2] * _b[2] + _a[3] * _b[3];
  _res[4] = _a[4] + _b[4] * (_tmp0 + _tmp1 + 1) + _b[5] * (_tmp2 - _tmp4) + _b[6] * (_tmp5 + _tmp7);
  _res[5] = _a[5] + _b[4] * (_tmp2 + _tmp4) + _b[5] * (_tmp1 + _tmp8) + _b[6] * (_tmp10 - _tmp9);
  _res[6] = _a[6] + _b[4] * (-_tmp5 + _tmp7) + _b[5] * (_tmp10 + _tmp9) + _b[6] * (_tmp0 + _tmp8);

  return sym::Pose3<Scalar>(_res);
}

/**
 *
 * Returns the element that when composed with a produces b. For vector spaces it is b - a.
 *
 * Implementation is simply `compose(inverse(a), b)`.
 *
 * Returns:
 *     Element: c such that a @ c = b
 *
 */
template <typename Scalar>
sym::Pose3<Scalar> GroupOps<Pose3<Scalar>>::Between(const sym::Pose3<Scalar>& a,
                                                    const sym::Pose3<Scalar>& b) {
  // Total ops: 103

  // Input arrays
  const Eigen::Matrix<Scalar, 7, 1>& _a = a.Data();
  const Eigen::Matrix<Scalar, 7, 1>& _b = b.Data();

  // Intermediate terms (20)
  const Scalar _tmp0 = -2 * std::pow(_a[1], Scalar(2));
  const Scalar _tmp1 = -2 * std::pow(_a[2], Scalar(2));
  const Scalar _tmp2 = _tmp0 + _tmp1 + 1;
  const Scalar _tmp3 = 2 * _a[0];
  const Scalar _tmp4 = _a[1] * _tmp3;
  const Scalar _tmp5 = 2 * _a[2];
  const Scalar _tmp6 = _a[3] * _tmp5;
  const Scalar _tmp7 = _tmp4 + _tmp6;
  const Scalar _tmp8 = 2 * _a[1] * _a[3];
  const Scalar _tmp9 = _a[2] * _tmp3;
  const Scalar _tmp10 = -_tmp8 + _tmp9;
  const Scalar _tmp11 = 1 - 2 * std::pow(_a[0], Scalar(2));
  const Scalar _tmp12 = _tmp1 + _tmp11;
  const Scalar _tmp13 = _tmp4 - _tmp6;
  const Scalar _tmp14 = _a[3] * _tmp3;
  const Scalar _tmp15 = _a[1] * _tmp5;
  const Scalar _tmp16 = _tmp14 + _tmp15;
  const Scalar _tmp17 = _tmp0 + _tmp11;
  const Scalar _tmp18 = -_tmp14 + _tmp15;
  const Scalar _tmp19 = _tmp8 + _tmp9;

  // Output terms (1)
  Eigen::Matrix<Scalar, 7, 1> _res;

  _res[0] = -_a[0] * _b[3] - _a[1] * _b[2] + _a[2] * _b[1] + _a[3] * _b[0];
  _res[1] = _a[0] * _b[2] - _a[1] * _b[3] - _a[2] * _b[0] + _a[3] * _b[1];
  _res[2] = -_a[0] * _b[1] + _a[1] * _b[0] - _a[2] * _b[3] + _a[3] * _b[2];
  _res[3] = _a[0] * _b[0] + _a[1] * _b[1] + _a[2] * _b[2] + _a[3] * _b[3];
  _res[4] = -_a[4] * _tmp2 - _a[5] * _tmp7 - _a[6] * _tmp10 + _b[4] * _tmp2 + _b[5] * _tmp7 +
            _b[6] * _tmp10;
  _res[5] = -_a[4] * _tmp13 - _a[5] * _tmp12 - _a[6] * _tmp16 + _b[4] * _tmp13 + _b[5] * _tmp12 +
            _b[6] * _tmp16;
  _res[6] = -_a[4] * _tmp19 - _a[5] * _tmp18 - _a[6] * _tmp17 + _b[4] * _tmp19 + _b[5] * _tmp18 +
            _b[6] * _tmp17;

  return sym::Pose3<Scalar>(_res);
}

/**
 *
 * Inverse of the element a.
 *
 * Returns:
 *     Element: b such that a @ b = identity
 *     res_D_a: (6x6) jacobian of res (6) wrt arg a (6)
 */
template <typename Scalar>
sym::Pose3<Scalar> GroupOps<Pose3<Scalar>>::InverseWithJacobian(const sym::Pose3<Scalar>& a,
                                                                SelfJacobian* const res_D_a) {
  // Total ops: 222

  // Input arrays
  const Eigen::Matrix<Scalar, 7, 1>& _a = a.Data();

  // Intermediate terms (67)
  const Scalar _tmp0 = std::pow(_a[1], Scalar(2));
  const Scalar _tmp1 = -2 * _tmp0;
  const Scalar _tmp2 = std::pow(_a[2], Scalar(2));
  const Scalar _tmp3 = -2 * _tmp2;
  const Scalar _tmp4 = _tmp1 + _tmp3 + 1;
  const Scalar _tmp5 = 2 * _a[0];
  const Scalar _tmp6 = _a[1] * _tmp5;
  const Scalar _tmp7 = 2 * _a[2];
  const Scalar _tmp8 = _a[3] * _tmp7;
  const Scalar _tmp9 = _tmp6 + _tmp8;
  const Scalar _tmp10 = 2 * _a[1];
  const Scalar _tmp11 = _a[3] * _tmp10;
  const Scalar _tmp12 = -_tmp11;
  const Scalar _tmp13 = _a[0] * _tmp7;
  const Scalar _tmp14 = _tmp12 + _tmp13;
  const Scalar _tmp15 = std::pow(_a[0], Scalar(2));
  const Scalar _tmp16 = 1 - 2 * _tmp15;
  const Scalar _tmp17 = _tmp16 + _tmp3;
  const Scalar _tmp18 = -_tmp8;
  const Scalar _tmp19 = _tmp18 + _tmp6;
  const Scalar _tmp20 = _a[3] * _tmp5;
  const Scalar _tmp21 = _a[1] * _tmp7;
  const Scalar _tmp22 = _tmp20 + _tmp21;
  const Scalar _tmp23 = _tmp1 + _tmp16;
  const Scalar _tmp24 = -_tmp20;
  const Scalar _tmp25 = _tmp21 + _tmp24;
  const Scalar _tmp26 = _tmp11 + _tmp13;
  const Scalar _tmp27 = -std::pow(_a[3], Scalar(2));
  const Scalar _tmp28 = -_tmp6;
  const Scalar _tmp29 = -_tmp13;
  const Scalar _tmp30 = _tmp15 + _tmp27;
  const Scalar _tmp31 = -_tmp21;
  const Scalar _tmp32 = _a[5] * _tmp10;
  const Scalar _tmp33 = _a[6] * _tmp7;
  const Scalar _tmp34 = _tmp32 + _tmp33;
  const Scalar _tmp35 = (Scalar(1) / Scalar(2)) * _a[3];
  const Scalar _tmp36 = _a[6] * _tmp10;
  const Scalar _tmp37 = _a[5] * _tmp7;
  const Scalar _tmp38 = -_tmp36 + _tmp37;
  const Scalar _tmp39 = (Scalar(1) / Scalar(2)) * _a[0];
  const Scalar _tmp40 = _a[6] * _tmp5;
  const Scalar _tmp41 = 2 * _a[3];
  const Scalar _tmp42 = _a[5] * _tmp41;
  const Scalar _tmp43 = 4 * _a[4];
  const Scalar _tmp44 = -_a[2] * _tmp43 + _tmp40 + _tmp42;
  const Scalar _tmp45 = (Scalar(1) / Scalar(2)) * _a[1];
  const Scalar _tmp46 = _a[5] * _tmp5;
  const Scalar _tmp47 = _a[6] * _tmp41;
  const Scalar _tmp48 = -_a[1] * _tmp43 + _tmp46 - _tmp47;
  const Scalar _tmp49 = (Scalar(1) / Scalar(2)) * _a[2];
  const Scalar _tmp50 = (Scalar(1) / Scalar(2)) * _tmp44;
  const Scalar _tmp51 = _a[4] * _tmp7;
  const Scalar _tmp52 = _tmp40 - _tmp51;
  const Scalar _tmp53 = _a[4] * _tmp5;
  const Scalar _tmp54 = _tmp33 + _tmp53;
  const Scalar _tmp55 = _a[4] * _tmp41;
  const Scalar _tmp56 = 4 * _a[5];
  const Scalar _tmp57 = -_a[2] * _tmp56 + _tmp36 - _tmp55;
  const Scalar _tmp58 = _a[4] * _tmp10;
  const Scalar _tmp59 = -_a[0] * _tmp56 + _tmp47 + _tmp58;
  const Scalar _tmp60 = (Scalar(1) / Scalar(2)) * _tmp54;
  const Scalar _tmp61 = -_tmp46 + _tmp58;
  const Scalar _tmp62 = (Scalar(1) / Scalar(2)) * _tmp61;
  const Scalar _tmp63 = (Scalar(1) / Scalar(2)) * _tmp32 + (Scalar(1) / Scalar(2)) * _tmp53;
  const Scalar _tmp64 = 4 * _a[6];
  const Scalar _tmp65 = -_a[0] * _tmp64 - _tmp42 + _tmp51;
  const Scalar _tmp66 = -_a[1] * _tmp64 + _tmp37 + _tmp55;

  // Output terms (2)
  Eigen::Matrix<Scalar, 7, 1> _res;

  _res[0] = -_a[0];
  _res[1] = -_a[1];
  _res[2] = -_a[2];
  _res[3] = _a[3];
  _res[4] = -_a[4] * _tmp4 - _a[5] * _tmp9 - _a[6] * _tmp14;
  _res[5] = -_a[4] * _tmp19 - _a[5] * _tmp17 - _a[6] * _tmp22;
  _res[6] = -_a[4] * _tmp26 - _a[5] * _tmp25 - _a[6] * _tmp23;

  if (res_D_a != nullptr) {
    Eigen::Matrix<Scalar, 6, 6>& _res_D_a = (*res_D_a);

    _res_D_a(0, 0) = _tmp0 - _tmp15 + _tmp2 + _tmp27;
    _res_D_a(0, 1) = _tmp28 + _tmp8;
    _res_D_a(0, 2) = _tmp12 + _tmp29;
    _res_D_a(0, 3) = 0;
    _res_D_a(0, 4) = 0;
    _res_D_a(0, 5) = 0;
    _res_D_a(1, 0) = _tmp18 + _tmp28;
    _res_D_a(1, 1) = -_tmp0 + _tmp2 + _tmp30;
    _res_D_a(1, 2) = _tmp20 + _tmp31;
    _res_D_a(1, 3) = 0;
    _res_D_a(1, 4) = 0;
    _res_D_a(1, 5) = 0;
    _res_D_a(2, 0) = _tmp11 + _tmp29;
    _res_D_a(2, 1) = _tmp24 + _tmp31;
    _res_D_a(2, 2) = _tmp0 - _tmp2 + _tmp30;
    _res_D_a(2, 3) = 0;
    _res_D_a(2, 4) = 0;
    _res_D_a(2, 5) = 0;
    _res_D_a(3, 0) = -_tmp34 * _tmp35 + _tmp38 * _tmp39 + _tmp44 * _tmp45 - _tmp48 * _tmp49;
    _res_D_a(3, 1) = -_a[0] * _tmp50 + _tmp34 * _tmp49 - _tmp35 * _tmp48 + _tmp38 * _tmp45;
    _res_D_a(3, 2) = -_a[3] * _tmp50 - _tmp34 * _tmp45 + _tmp38 * _tmp49 + _tmp39 * _tmp48;
    _res_D_a(3, 3) = -_tmp4;
    _res_D_a(3, 4) = -_tmp9;
    _res_D_a(3, 5) = -_tmp14;
    _res_D_a(4, 0) = -_tmp35 * _tmp59 + _tmp39 * _tmp52 + _tmp45 * _tmp57 - _tmp49 * _tmp54;
    _res_D_a(4, 1) = -_a[3] * _tmp60 - _tmp39 * _tmp57 + _tmp45 * _tmp52 + _tmp49 * _tmp59;
    _res_D_a(4, 2) = _a[0] * _tmp60 - _tmp35 * _tmp57 - _tmp45 * _tmp59 + _tmp49 * _tmp52;
    _res_D_a(4, 3) = -_tmp19;
    _res_D_a(4, 4) = -_tmp17;
    _res_D_a(4, 5) = -_tmp22;
    _res_D_a(5, 0) = _a[0] * _tmp62 + _a[1] * _tmp63 - _tmp35 * _tmp65 - _tmp49 * _tmp66;
    _res_D_a(5, 1) = -_a[0] * _tmp63 + _a[1] * _tmp62 - _tmp35 * _tmp66 + _tmp49 * _tmp65;
    _res_D_a(5, 2) = -_a[3] * _tmp63 + _tmp39 * _tmp66 - _tmp45 * _tmp65 + _tmp49 * _tmp61;
    _res_D_a(5, 3) = -_tmp26;
    _res_D_a(5, 4) = -_tmp25;
    _res_D_a(5, 5) = -_tmp23;
  }

  return sym::Pose3<Scalar>(_res);
}

/**
 *
 * Composition of two elements in the group.
 *
 * Returns:
 *     Element: a @ b
 *     res_D_a: (6x6) jacobian of res (6) wrt arg a (6)
 *     res_D_b: (6x6) jacobian of res (6) wrt arg b (6)
 */
template <typename Scalar>
sym::Pose3<Scalar> GroupOps<Pose3<Scalar>>::ComposeWithJacobians(const sym::Pose3<Scalar>& a,
                                                                 const sym::Pose3<Scalar>& b,
                                                                 SelfJacobian* const res_D_a,
                                                                 SelfJacobian* const res_D_b) {
  // Total ops: 484

  // Input arrays
  const Eigen::Matrix<Scalar, 7, 1>& _a = a.Data();
  const Eigen::Matrix<Scalar, 7, 1>& _b = b.Data();

  // Intermediate terms (133)
  const Scalar _tmp0 = _a[0] * _b[3] + _a[1] * _b[2] - _a[2] * _b[1] + _a[3] * _b[0];
  const Scalar _tmp1 = -_a[0] * _b[2] + _a[1] * _b[3] + _a[2] * _b[0] + _a[3] * _b[1];
  const Scalar _tmp2 = _a[0] * _b[1] - _a[1] * _b[0] + _a[2] * _b[3] + _a[3] * _b[2];
  const Scalar _tmp3 = -_a[0] * _b[0] - _a[1] * _b[1] - _a[2] * _b[2] + _a[3] * _b[3];
  const Scalar _tmp4 = -2 * std::pow(_a[1], Scalar(2));
  const Scalar _tmp5 = 1 - 2 * std::pow(_a[2], Scalar(2));
  const Scalar _tmp6 = _tmp4 + _tmp5;
  const Scalar _tmp7 = 2 * _a[0];
  const Scalar _tmp8 = _a[1] * _tmp7;
  const Scalar _tmp9 = 2 * _a[3];
  const Scalar _tmp10 = _a[2] * _tmp9;
  const Scalar _tmp11 = -_tmp10 + _tmp8;
  const Scalar _tmp12 = _a[1] * _tmp9;
  const Scalar _tmp13 = 2 * _a[2];
  const Scalar _tmp14 = _a[0] * _tmp13;
  const Scalar _tmp15 = _tmp12 + _tmp14;
  const Scalar _tmp16 = -2 * std::pow(_a[0], Scalar(2));
  const Scalar _tmp17 = _tmp16 + _tmp5;
  const Scalar _tmp18 = _tmp10 + _tmp8;
  const Scalar _tmp19 = _a[0] * _tmp9;
  const Scalar _tmp20 = _a[1] * _tmp13;
  const Scalar _tmp21 = -_tmp19 + _tmp20;
  const Scalar _tmp22 = _tmp16 + _tmp4 + 1;
  const Scalar _tmp23 = _tmp19 + _tmp20;
  const Scalar _tmp24 = -_tmp12 + _tmp14;
  const Scalar _tmp25 = 2 * _tmp2;
  const Scalar _tmp26 = _b[0] * _tmp25;
  const Scalar _tmp27 = 2 * _tmp3;
  const Scalar _tmp28 = _b[1] * _tmp27;
  const Scalar _tmp29 = 2 * _tmp1;
  const Scalar _tmp30 = _b[3] * _tmp29;
  const Scalar _tmp31 = 2 * _tmp0;
  const Scalar _tmp32 = _b[2] * _tmp31;
  const Scalar _tmp33 = -_tmp30 + _tmp32;
  const Scalar _tmp34 = _tmp26 - _tmp28 + _tmp33;
  const Scalar _tmp35 = (Scalar(1) / Scalar(2)) * _a[1];
  const Scalar _tmp36 = _b[2] * _tmp29;
  const Scalar _tmp37 = _b[3] * _tmp31;
  const Scalar _tmp38 = -_tmp37;
  const Scalar _tmp39 = _b[1] * _tmp25;
  const Scalar _tmp40 = _b[0] * _tmp27;
  const Scalar _tmp41 = _tmp39 + _tmp40;
  const Scalar _tmp42 = -_tmp36 + _tmp38 + _tmp41;
  const Scalar _tmp43 = (Scalar(1) / Scalar(2)) * _a[0];
  const Scalar _tmp44 = _b[1] * _tmp29;
  const Scalar _tmp45 = -_tmp44;
  const Scalar _tmp46 = _b[0] * _tmp31;
  const Scalar _tmp47 = _b[2] * _tmp25;
  const Scalar _tmp48 = _b[3] * _tmp27;
  const Scalar _tmp49 = -_tmp47 + _tmp48;
  const Scalar _tmp50 = _tmp45 + _tmp46 + _tmp49;
  const Scalar _tmp51 = (Scalar(1) / Scalar(2)) * _a[3];
  const Scalar _tmp52 = _b[3] * _tmp25;
  const Scalar _tmp53 = _b[2] * _tmp27;
  const Scalar _tmp54 = _b[0] * _tmp29;
  const Scalar _tmp55 = _b[1] * _tmp31;
  const Scalar _tmp56 = _tmp54 + _tmp55;
  const Scalar _tmp57 = _tmp52 + _tmp53 + _tmp56;
  const Scalar _tmp58 = (Scalar(1) / Scalar(2)) * _a[2];
  const Scalar _tmp59 = -_tmp26 + _tmp28 + _tmp33;
  const Scalar _tmp60 = _tmp36 + _tmp37 + _tmp41;
  const Scalar _tmp61 = -_tmp46;
  const Scalar _tmp62 = _tmp44 + _tmp49 + _tmp61;
  const Scalar _tmp63 = -_tmp52;
  const Scalar _tmp64 = -_tmp53 + _tmp56 + _tmp63;
  const Scalar _tmp65 = _tmp26 + _tmp28 + _tmp30 + _tmp32;
  const Scalar _tmp66 = _tmp36 + _tmp38 + _tmp39 - _tmp40;
  const Scalar _tmp67 = _tmp45 + _tmp47 + _tmp48 + _tmp61;
  const Scalar _tmp68 = _tmp53 + _tmp54 - _tmp55 + _tmp63;
  const Scalar _tmp69 = 2 * _a[1];
  const Scalar _tmp70 = _b[6] * _tmp69;
  const Scalar _tmp71 = _b[5] * _tmp13;
  const Scalar _tmp72 = _tmp70 - _tmp71;
  const Scalar _tmp73 = _b[5] * _tmp69;
  const Scalar _tmp74 = _b[6] * _tmp13;
  const Scalar _tmp75 = _tmp73 + _tmp74;
  const Scalar _tmp76 = _a[0] * _b[6];
  const Scalar _tmp77 = 2 * _tmp76;
  const Scalar _tmp78 = _b[5] * _tmp9;
  const Scalar _tmp79 = -4 * _a[2] * _b[4] + _tmp77 - _tmp78;
  const Scalar _tmp80 = 4 * _a[1];
  const Scalar _tmp81 = _b[5] * _tmp7;
  const Scalar _tmp82 = _b[6] * _tmp9;
  const Scalar _tmp83 = -_b[4] * _tmp80 + _tmp81 + _tmp82;
  const Scalar _tmp84 = _b[4] * _tmp13;
  const Scalar _tmp85 = -_tmp77 + _tmp84;
  const Scalar _tmp86 = _b[4] * _tmp7;
  const Scalar _tmp87 = _tmp74 + _tmp86;
  const Scalar _tmp88 = _b[4] * _tmp9;
  const Scalar _tmp89 = 4 * _b[5];
  const Scalar _tmp90 = -_a[2] * _tmp89 + _tmp70 + _tmp88;
  const Scalar _tmp91 = _b[4] * _tmp69;
  const Scalar _tmp92 = -_a[0] * _tmp89 - _tmp82 + _tmp91;
  const Scalar _tmp93 = _tmp81 - _tmp91;
  const Scalar _tmp94 = (Scalar(1) / Scalar(2)) * _tmp93;
  const Scalar _tmp95 = _tmp73 + _tmp86;
  const Scalar _tmp96 = -_b[6] * _tmp80 + _tmp71 - _tmp88;
  const Scalar _tmp97 = -4 * _tmp76 + _tmp78 + _tmp84;
  const Scalar _tmp98 = _a[0] * _tmp25;
  const Scalar _tmp99 = _tmp1 * _tmp9;
  const Scalar _tmp100 = _a[2] * _tmp31;
  const Scalar _tmp101 = _a[1] * _tmp27;
  const Scalar _tmp102 = _tmp100 + _tmp101 - _tmp98 - _tmp99;
  const Scalar _tmp103 = (Scalar(1) / Scalar(2)) * _b[1];
  const Scalar _tmp104 = -_tmp102 * _tmp103;
  const Scalar _tmp105 = _tmp2 * _tmp9;
  const Scalar _tmp106 = _a[0] * _tmp29;
  const Scalar _tmp107 = _a[1] * _tmp31;
  const Scalar _tmp108 = _a[2] * _tmp27;
  const Scalar _tmp109 = _tmp105 - _tmp106 + _tmp107 - _tmp108;
  const Scalar _tmp110 = (Scalar(1) / Scalar(2)) * _b[2];
  const Scalar _tmp111 = _a[0] * _tmp31 + _a[1] * _tmp29 + _a[2] * _tmp25 + _a[3] * _tmp27;
  const Scalar _tmp112 = (Scalar(1) / Scalar(2)) * _b[3];
  const Scalar _tmp113 = _tmp111 * _tmp112;
  const Scalar _tmp114 = _a[1] * _tmp25;
  const Scalar _tmp115 = _a[2] * _tmp29;
  const Scalar _tmp116 = _a[3] * _tmp31;
  const Scalar _tmp117 = _a[0] * _tmp27;
  const Scalar _tmp118 = (Scalar(1) / Scalar(2)) * _tmp114 - Scalar(1) / Scalar(2) * _tmp115 -
                         Scalar(1) / Scalar(2) * _tmp116 + (Scalar(1) / Scalar(2)) * _tmp117;
  const Scalar _tmp119 = -_b[0] * _tmp118 + _tmp113;
  const Scalar _tmp120 = _tmp110 * _tmp111;
  const Scalar _tmp121 = (Scalar(1) / Scalar(2)) * _b[0];
  const Scalar _tmp122 = _tmp102 * _tmp121;
  const Scalar _tmp123 = (Scalar(1) / Scalar(2)) * _tmp109;
  const Scalar _tmp124 = _tmp103 * _tmp111;
  const Scalar _tmp125 = _b[2] * _tmp118;
  const Scalar _tmp126 = -Scalar(1) / Scalar(2) * _tmp114 + (Scalar(1) / Scalar(2)) * _tmp115 +
                         (Scalar(1) / Scalar(2)) * _tmp116 - Scalar(1) / Scalar(2) * _tmp117;
  const Scalar _tmp127 = -_tmp105 + _tmp106 - _tmp107 + _tmp108;
  const Scalar _tmp128 = (Scalar(1) / Scalar(2)) * _tmp127;
  const Scalar _tmp129 = -_tmp110 * _tmp127;
  const Scalar _tmp130 = _tmp111 * _tmp121;
  const Scalar _tmp131 = _b[1] * _tmp128;
  const Scalar _tmp132 = -_tmp100 - _tmp101 + _tmp98 + _tmp99;

  // Output terms (3)
  Eigen::Matrix<Scalar, 7, 1> _res;

  _res[0] = _tmp0;
  _res[1] = _tmp1;
  _res[2] = _tmp2;
  _res[3] = _tmp3;
  _res[4] = _a[4] + _b[4] * _tmp6 + _b[5] * _tmp11 + _b[6] * _tmp15;
  _res[5] = _a[5] + _b[4] * _tmp18 + _b[5] * _tmp17 + _b[6] * _tmp21;
  _res[6] = _a[6] + _b[4] * _tmp24 + _b[5] * _tmp23 + _b[6] * _tmp22;

  if (res_D_a != nullptr) {
    Eigen::Matrix<Scalar, 6, 6>& _res_D_a = (*res_D_a);

    _res_D_a(0, 0) = -_tmp34 * _tmp35 - _tmp42 * _tmp43 + _tmp50 * _tmp51 + _tmp57 * _tmp58;
    _res_D_a(0, 1) = _tmp34 * _tmp43 - _tmp35 * _tmp42 - _tmp50 * _tmp58 + _tmp51 * _tmp57;
    _res_D_a(0, 2) = _tmp34 * _tmp51 + _tmp35 * _tmp50 - _tmp42 * _tmp58 - _tmp43 * _tmp57;
    _res_D_a(0, 3) = 0;
    _res_D_a(0, 4) = 0;
    _res_D_a(0, 5) = 0;
    _res_D_a(1, 0) = -_tmp35 * _tmp60 - _tmp43 * _tmp59 + _tmp51 * _tmp64 + _tmp58 * _tmp62;
    _res_D_a(1, 1) = -_tmp35 * _tmp59 + _tmp43 * _tmp60 + _tmp51 * _tmp62 - _tmp58 * _tmp64;
    _res_D_a(1, 2) = _tmp35 * _tmp64 - _tmp43 * _tmp62 + _tmp51 * _tmp60 - _tmp58 * _tmp59;
    _res_D_a(1, 3) = 0;
    _res_D_a(1, 4) = 0;
    _res_D_a(1, 5) = 0;
    _res_D_a(2, 0) = -_tmp35 * _tmp67 - _tmp43 * _tmp68 + _tmp51 * _tmp65 + _tmp58 * _tmp66;
    _res_D_a(2, 1) = -_tmp35 * _tmp68 + _tmp43 * _tmp67 + _tmp51 * _tmp66 - _tmp58 * _tmp65;
    _res_D_a(2, 2) = _tmp35 * _tmp65 - _tmp43 * _tmp66 + _tmp51 * _tmp67 - _tmp58 * _tmp68;
    _res_D_a(2, 3) = 0;
    _res_D_a(2, 4) = 0;
    _res_D_a(2, 5) = 0;
    _res_D_a(3, 0) = -_tmp35 * _tmp79 - _tmp43 * _tmp72 + _tmp51 * _tmp75 + _tmp58 * _tmp83;
    _res_D_a(3, 1) = -_tmp35 * _tmp72 + _tmp43 * _tmp79 + _tmp51 * _tmp83 - _tmp58 * _tmp75;
    _res_D_a(3, 2) = _tmp35 * _tmp75 - _tmp43 * _tmp83 + _tmp51 * _tmp79 - _tmp58 * _tmp72;
    _res_D_a(3, 3) = 1;
    _res_D_a(3, 4) = 0;
    _res_D_a(3, 5) = 0;
    _res_D_a(4, 0) = -_tmp35 * _tmp90 - _tmp43 * _tmp85 + _tmp51 * _tmp92 + _tmp58 * _tmp87;
    _res_D_a(4, 1) = -_tmp35 * _tmp85 + _tmp43 * _tmp90 + _tmp51 * _tmp87 - _tmp58 * _tmp92;
    _res_D_a(4, 2) = _tmp35 * _tmp92 - _tmp43 * _tmp87 + _tmp51 * _tmp90 - _tmp58 * _tmp85;
    _res_D_a(4, 3) = 0;
    _res_D_a(4, 4) = 1;
    _res_D_a(4, 5) = 0;
    _res_D_a(5, 0) = -_a[0] * _tmp94 - _tmp35 * _tmp95 + _tmp51 * _tmp97 + _tmp58 * _tmp96;
    _res_D_a(5, 1) = -_tmp35 * _tmp93 + _tmp43 * _tmp95 + _tmp51 * _tmp96 - _tmp58 * _tmp97;
    _res_D_a(5, 2) = -_a[2] * _tmp94 + _tmp35 * _tmp97 - _tmp43 * _tmp96 + _tmp51 * _tmp95;
    _res_D_a(5, 3) = 0;
    _res_D_a(5, 4) = 0;
    _res_D_a(5, 5) = 1;
  }

  if (res_D_b != nullptr) {
    Eigen::Matrix<Scalar, 6, 6>& _res_D_b = (*res_D_b);

    _res_D_b(0, 0) = _tmp104 + _tmp109 * _tmp110 + _tmp119;
    _res_D_b(0, 1) = -_b[1] * _tmp118 + _b[3] * _tmp123 - _tmp120 + _tmp122;
    _res_D_b(0, 2) = -_b[0] * _tmp123 + _tmp102 * _tmp112 + _tmp124 - _tmp125;
    _res_D_b(0, 3) = 0;
    _res_D_b(0, 4) = 0;
    _res_D_b(0, 5) = 0;
    _res_D_b(1, 0) = -_b[1] * _tmp126 + _b[3] * _tmp128 + _tmp120 - _tmp122;
    _res_D_b(1, 1) = _b[0] * _tmp126 + _tmp104 + _tmp113 + _tmp129;
    _res_D_b(1, 2) = _b[3] * _tmp126 - _tmp102 * _tmp110 - _tmp130 + _tmp131;
    _res_D_b(1, 3) = 0;
    _res_D_b(1, 4) = 0;
    _res_D_b(1, 5) = 0;
    _res_D_b(2, 0) = -_b[0] * _tmp128 + _tmp112 * _tmp132 - _tmp124 + _tmp125;
    _res_D_b(2, 1) = _b[3] * _tmp118 - _tmp110 * _tmp132 + _tmp130 - _tmp131;
    _res_D_b(2, 2) = _tmp103 * _tmp132 + _tmp119 + _tmp129;
    _res_D_b(2, 3) = 0;
    _res_D_b(2, 4) = 0;
    _res_D_b(2, 5) = 0;
    _res_D_b(3, 0) = 0;
    _res_D_b(3, 1) = 0;
    _res_D_b(3, 2) = 0;
    _res_D_b(3, 3) = _tmp6;
    _res_D_b(3, 4) = _tmp11;
    _res_D_b(3, 5) = _tmp15;
    _res_D_b(4, 0) = 0;
    _res_D_b(4, 1) = 0;
    _res_D_b(4, 2) = 0;
    _res_D_b(4, 3) = _tmp18;
    _res_D_b(4, 4) = _tmp17;
    _res_D_b(4, 5) = _tmp21;
    _res_D_b(5, 0) = 0;
    _res_D_b(5, 1) = 0;
    _res_D_b(5, 2) = 0;
    _res_D_b(5, 3) = _tmp24;
    _res_D_b(5, 4) = _tmp23;
    _res_D_b(5, 5) = _tmp22;
  }

  return sym::Pose3<Scalar>(_res);
}

/**
 *
 * Returns the element that when composed with a produces b. For vector spaces it is b - a.
 *
 * Implementation is simply `compose(inverse(a), b)`.
 *
 * Returns:
 *     Element: c such that a @ c = b
 *     res_D_a: (6x6) jacobian of res (6) wrt arg a (6)
 *     res_D_b: (6x6) jacobian of res (6) wrt arg b (6)
 */
template <typename Scalar>
sym::Pose3<Scalar> GroupOps<Pose3<Scalar>>::BetweenWithJacobians(const sym::Pose3<Scalar>& a,
                                                                 const sym::Pose3<Scalar>& b,
                                                                 SelfJacobian* const res_D_a,
                                                                 SelfJacobian* const res_D_b) {
  // Total ops: 586

  // Input arrays
  const Eigen::Matrix<Scalar, 7, 1>& _a = a.Data();
  const Eigen::Matrix<Scalar, 7, 1>& _b = b.Data();

  // Intermediate terms (149)
  const Scalar _tmp0 = -_a[0] * _b[3] - _a[1] * _b[2] + _a[2] * _b[1] + _a[3] * _b[0];
  const Scalar _tmp1 = _a[0] * _b[2] - _a[1] * _b[3] - _a[2] * _b[0] + _a[3] * _b[1];
  const Scalar _tmp2 = -_a[0] * _b[1] + _a[1] * _b[0] - _a[2] * _b[3] + _a[3] * _b[2];
  const Scalar _tmp3 = _a[0] * _b[0] + _a[1] * _b[1] + _a[2] * _b[2] + _a[3] * _b[3];
  const Scalar _tmp4 = 2 * std::pow(_a[1], Scalar(2));
  const Scalar _tmp5 = -_tmp4;
  const Scalar _tmp6 = 2 * std::pow(_a[2], Scalar(2));
  const Scalar _tmp7 = 1 - _tmp6;
  const Scalar _tmp8 = _tmp5 + _tmp7;
  const Scalar _tmp9 = 2 * _a[0];
  const Scalar _tmp10 = _a[1] * _tmp9;
  const Scalar _tmp11 = 2 * _a[2];
  const Scalar _tmp12 = _a[3] * _tmp11;
  const Scalar _tmp13 = _tmp10 + _tmp12;
  const Scalar _tmp14 = 2 * _a[3];
  const Scalar _tmp15 = _a[1] * _tmp14;
  const Scalar _tmp16 = -_tmp15;
  const Scalar _tmp17 = _a[0] * _tmp11;
  const Scalar _tmp18 = _tmp16 + _tmp17;
  const Scalar _tmp19 = 2 * std::pow(_a[0], Scalar(2));
  const Scalar _tmp20 = -_tmp19;
  const Scalar _tmp21 = _tmp20 + _tmp7;
  const Scalar _tmp22 = -_tmp12;
  const Scalar _tmp23 = _tmp10 + _tmp22;
  const Scalar _tmp24 = _a[0] * _tmp14;
  const Scalar _tmp25 = _a[1] * _tmp11;
  const Scalar _tmp26 = _tmp24 + _tmp25;
  const Scalar _tmp27 = _tmp20 + _tmp5 + 1;
  const Scalar _tmp28 = -_tmp24;
  const Scalar _tmp29 = _tmp25 + _tmp28;
  const Scalar _tmp30 = _tmp15 + _tmp17;
  const Scalar _tmp31 = 2 * _tmp1;
  const Scalar _tmp32 = _b[0] * _tmp31;
  const Scalar _tmp33 = -_tmp32;
  const Scalar _tmp34 = 2 * _tmp3;
  const Scalar _tmp35 = _b[2] * _tmp34;
  const Scalar _tmp36 = 2 * _tmp2;
  const Scalar _tmp37 = _b[3] * _tmp36;
  const Scalar _tmp38 = 2 * _tmp0;
  const Scalar _tmp39 = -_b[1] * _tmp38;
  const Scalar _tmp40 = -_tmp37 + _tmp39;
  const Scalar _tmp41 = _tmp33 - _tmp35 + _tmp40;
  const Scalar _tmp42 = (Scalar(1) / Scalar(2)) * _a[2];
  const Scalar _tmp43 = _b[3] * _tmp31;
  const Scalar _tmp44 = _b[1] * _tmp34;
  const Scalar _tmp45 = -_b[0] * _tmp36;
  const Scalar _tmp46 = _b[2] * _tmp38;
  const Scalar _tmp47 = _tmp45 - _tmp46;
  const Scalar _tmp48 = _tmp43 + _tmp44 + _tmp47;
  const Scalar _tmp49 = (Scalar(1) / Scalar(2)) * _a[1];
  const Scalar _tmp50 = _b[1] * _tmp31;
  const Scalar _tmp51 = _b[0] * _tmp38;
  const Scalar _tmp52 = _b[2] * _tmp36;
  const Scalar _tmp53 = -_b[3] * _tmp34;
  const Scalar _tmp54 = _tmp52 + _tmp53;
  const Scalar _tmp55 = _tmp50 - _tmp51 + _tmp54;
  const Scalar _tmp56 = (Scalar(1) / Scalar(2)) * _a[3];
  const Scalar _tmp57 = _b[1] * _tmp36;
  const Scalar _tmp58 = _b[3] * _tmp38;
  const Scalar _tmp59 = -_tmp58;
  const Scalar _tmp60 = -_b[2] * _tmp31;
  const Scalar _tmp61 = _b[0] * _tmp34;
  const Scalar _tmp62 = _tmp60 + _tmp61;
  const Scalar _tmp63 = _tmp57 + _tmp59 + _tmp62;
  const Scalar _tmp64 = (Scalar(1) / Scalar(2)) * _a[0];
  const Scalar _tmp65 = _tmp33 + _tmp35 + _tmp37 + _tmp39;
  const Scalar _tmp66 = -_tmp43;
  const Scalar _tmp67 = _tmp44 + _tmp45 + _tmp46 + _tmp66;
  const Scalar _tmp68 = -_tmp50 + _tmp51 + _tmp54;
  const Scalar _tmp69 = -_tmp57;
  const Scalar _tmp70 = _tmp59 + _tmp60 - _tmp61 + _tmp69;
  const Scalar _tmp71 = _tmp32 + _tmp35 + _tmp40;
  const Scalar _tmp72 = -_tmp44 + _tmp47 + _tmp66;
  const Scalar _tmp73 = _tmp50 + _tmp51 - _tmp52 + _tmp53;
  const Scalar _tmp74 = _tmp58 + _tmp62 + _tmp69;
  const Scalar _tmp75 = 2 * _a[1];
  const Scalar _tmp76 = _b[6] * _tmp75;
  const Scalar _tmp77 = _a[6] * _tmp75;
  const Scalar _tmp78 = -_a[5] * _tmp11 + _b[5] * _tmp11;
  const Scalar _tmp79 = -_tmp76 + _tmp77 + _tmp78;
  const Scalar _tmp80 = 4 * _a[1];
  const Scalar _tmp81 = 2 * _a[5];
  const Scalar _tmp82 = _a[0] * _tmp81;
  const Scalar _tmp83 = _b[5] * _tmp9;
  const Scalar _tmp84 = _b[6] * _tmp14;
  const Scalar _tmp85 = _a[6] * _tmp14;
  const Scalar _tmp86 = (Scalar(1) / Scalar(2)) * _a[4] * _tmp80 -
                        Scalar(1) / Scalar(2) * _b[4] * _tmp80 - Scalar(1) / Scalar(2) * _tmp82 +
                        (Scalar(1) / Scalar(2)) * _tmp83 - Scalar(1) / Scalar(2) * _tmp84 +
                        (Scalar(1) / Scalar(2)) * _tmp85;
  const Scalar _tmp87 = -_a[1] * _tmp81 + _b[5] * _tmp75;
  const Scalar _tmp88 = -_a[6] * _tmp11 + _b[6] * _tmp11;
  const Scalar _tmp89 = _tmp87 + _tmp88;
  const Scalar _tmp90 = _b[5] * _tmp14;
  const Scalar _tmp91 = _a[5] * _tmp14;
  const Scalar _tmp92 = 4 * _a[2];
  const Scalar _tmp93 = -_a[6] * _tmp9 + _b[6] * _tmp9;
  const Scalar _tmp94 = _a[4] * _tmp92 - _b[4] * _tmp92 + _tmp90 - _tmp91 + _tmp93;
  const Scalar _tmp95 = _tmp6 - 1;
  const Scalar _tmp96 = -_tmp10;
  const Scalar _tmp97 = -_tmp17;
  const Scalar _tmp98 = _b[4] * _tmp14;
  const Scalar _tmp99 = _a[4] * _tmp14;
  const Scalar _tmp100 = _a[5] * _tmp92 - _b[5] * _tmp92 + _tmp76 - _tmp77 - _tmp98 + _tmp99;
  const Scalar _tmp101 = 4 * _a[0];
  const Scalar _tmp102 = 2 * _a[4];
  const Scalar _tmp103 = -_a[1] * _tmp102 + _b[4] * _tmp75;
  const Scalar _tmp104 = _a[5] * _tmp101 - _b[5] * _tmp101 + _tmp103 + _tmp84 - _tmp85;
  const Scalar _tmp105 = -_a[0] * _tmp102 + _b[4] * _tmp9;
  const Scalar _tmp106 = _tmp105 + _tmp88;
  const Scalar _tmp107 = _b[4] * _tmp11;
  const Scalar _tmp108 = _a[4] * _tmp11;
  const Scalar _tmp109 = -_tmp107 + _tmp108 + _tmp93;
  const Scalar _tmp110 = -_tmp25;
  const Scalar _tmp111 = _a[6] * _tmp80 - _b[6] * _tmp80 + _tmp78 + _tmp98 - _tmp99;
  const Scalar _tmp112 = _tmp103 + _tmp82 - _tmp83;
  const Scalar _tmp113 = _tmp105 + _tmp87;
  const Scalar _tmp114 = _a[6] * _tmp101 - _b[6] * _tmp101 + _tmp107 - _tmp108 - _tmp90 + _tmp91;
  const Scalar _tmp115 = _a[0] * _tmp36;
  const Scalar _tmp116 = _a[3] * _tmp31;
  const Scalar _tmp117 = _a[2] * _tmp38;
  const Scalar _tmp118 = _a[1] * _tmp34;
  const Scalar _tmp119 = _tmp115 - _tmp116 - _tmp117 - _tmp118;
  const Scalar _tmp120 = (Scalar(1) / Scalar(2)) * _b[1];
  const Scalar _tmp121 = -_tmp119 * _tmp120;
  const Scalar _tmp122 = _a[3] * _tmp36;
  const Scalar _tmp123 = _a[0] * _tmp31;
  const Scalar _tmp124 = _a[1] * _tmp38;
  const Scalar _tmp125 = _a[2] * _tmp34;
  const Scalar _tmp126 = _tmp122 + _tmp123 - _tmp124 + _tmp125;
  const Scalar _tmp127 = (Scalar(1) / Scalar(2)) * _b[2];
  const Scalar _tmp128 = -_a[0] * _tmp38 - _a[1] * _tmp31 + _a[3] * _tmp34 - _tmp11 * _tmp2;
  const Scalar _tmp129 = (Scalar(1) / Scalar(2)) * _b[3];
  const Scalar _tmp130 = _tmp128 * _tmp129;
  const Scalar _tmp131 = _a[1] * _tmp36;
  const Scalar _tmp132 = _tmp1 * _tmp11;
  const Scalar _tmp133 = _a[3] * _tmp38;
  const Scalar _tmp134 = _a[0] * _tmp34;
  const Scalar _tmp135 = -_tmp131 + _tmp132 - _tmp133 - _tmp134;
  const Scalar _tmp136 = (Scalar(1) / Scalar(2)) * _b[0];
  const Scalar _tmp137 = _tmp130 - _tmp135 * _tmp136;
  const Scalar _tmp138 = _tmp119 * _tmp136;
  const Scalar _tmp139 = _tmp127 * _tmp128;
  const Scalar _tmp140 = _tmp120 * _tmp128;
  const Scalar _tmp141 = _tmp127 * _tmp135;
  const Scalar _tmp142 = _tmp131 - _tmp132 + _tmp133 + _tmp134;
  const Scalar _tmp143 = -_tmp122 - _tmp123 + _tmp124 - _tmp125;
  const Scalar _tmp144 = (Scalar(1) / Scalar(2)) * _tmp143;
  const Scalar _tmp145 = -_b[2] * _tmp144;
  const Scalar _tmp146 = _tmp128 * _tmp136;
  const Scalar _tmp147 = _b[1] * _tmp144;
  const Scalar _tmp148 = -_tmp115 + _tmp116 + _tmp117 + _tmp118;

  // Output terms (3)
  Eigen::Matrix<Scalar, 7, 1> _res;

  _res[0] = _tmp0;
  _res[1] = _tmp1;
  _res[2] = _tmp2;
  _res[3] = _tmp3;
  _res[4] = -_a[4] * _tmp8 - _a[5] * _tmp13 - _a[6] * _tmp18 + _b[4] * _tmp8 + _b[5] * _tmp13 +
            _b[6] * _tmp18;
  _res[5] = -_a[4] * _tmp23 - _a[5] * _tmp21 - _a[6] * _tmp26 + _b[4] * _tmp23 + _b[5] * _tmp21 +
            _b[6] * _tmp26;
  _res[6] = -_a[4] * _tmp30 - _a[5] * _tmp29 - _a[6] * _tmp27 + _b[4] * _tmp30 + _b[5] * _tmp29 +
            _b[6] * _tmp27;

  if (res_D_a != nullptr) {
    Eigen::Matrix<Scalar, 6, 6>& _res_D_a = (*res_D_a);

    _res_D_a(0, 0) = _tmp41 * _tmp42 - _tmp48 * _tmp49 + _tmp55 * _tmp56 - _tmp63 * _tmp64;
    _res_D_a(0, 1) = _tmp41 * _tmp56 - _tmp42 * _tmp55 + _tmp48 * _tmp64 - _tmp49 * _tmp63;
    _res_D_a(0, 2) = -_tmp41 * _tmp64 - _tmp42 * _tmp63 + _tmp48 * _tmp56 + _tmp49 * _tmp55;
    _res_D_a(0, 3) = 0;
    _res_D_a(0, 4) = 0;
    _res_D_a(0, 5) = 0;
    _res_D_a(1, 0) = _tmp42 * _tmp68 - _tmp49 * _tmp70 + _tmp56 * _tmp65 - _tmp64 * _tmp67;
    _res_D_a(1, 1) = -_tmp42 * _tmp65 - _tmp49 * _tmp67 + _tmp56 * _tmp68 + _tmp64 * _tmp70;
    _res_D_a(1, 2) = -_tmp42 * _tmp67 + _tmp49 * _tmp65 + _tmp56 * _tmp70 - _tmp64 * _tmp68;
    _res_D_a(1, 3) = 0;
    _res_D_a(1, 4) = 0;
    _res_D_a(1, 5) = 0;
    _res_D_a(2, 0) = _tmp42 * _tmp74 - _tmp49 * _tmp73 + _tmp56 * _tmp72 - _tmp64 * _tmp71;
    _res_D_a(2, 1) = -_tmp42 * _tmp72 - _tmp49 * _tmp71 + _tmp56 * _tmp74 + _tmp64 * _tmp73;
    _res_D_a(2, 2) = -_tmp42 * _tmp71 + _tmp49 * _tmp72 + _tmp56 * _tmp73 - _tmp64 * _tmp74;
    _res_D_a(2, 3) = 0;
    _res_D_a(2, 4) = 0;
    _res_D_a(2, 5) = 0;
    _res_D_a(3, 0) = _a[2] * _tmp86 - _tmp49 * _tmp94 + _tmp56 * _tmp89 - _tmp64 * _tmp79;
    _res_D_a(3, 1) = _a[3] * _tmp86 - _tmp42 * _tmp89 - _tmp49 * _tmp79 + _tmp64 * _tmp94;
    _res_D_a(3, 2) = -_a[0] * _tmp86 - _tmp42 * _tmp79 + _tmp49 * _tmp89 + _tmp56 * _tmp94;
    _res_D_a(3, 3) = _tmp4 + _tmp95;
    _res_D_a(3, 4) = _tmp22 + _tmp96;
    _res_D_a(3, 5) = _tmp15 + _tmp97;
    _res_D_a(4, 0) = -_tmp100 * _tmp49 + _tmp104 * _tmp56 + _tmp106 * _tmp42 - _tmp109 * _tmp64;
    _res_D_a(4, 1) = _tmp100 * _tmp64 - _tmp104 * _tmp42 + _tmp106 * _tmp56 - _tmp109 * _tmp49;
    _res_D_a(4, 2) = _tmp100 * _tmp56 + _tmp104 * _tmp49 - _tmp106 * _tmp64 - _tmp109 * _tmp42;
    _res_D_a(4, 3) = _tmp12 + _tmp96;
    _res_D_a(4, 4) = _tmp19 + _tmp95;
    _res_D_a(4, 5) = _tmp110 + _tmp28;
    _res_D_a(5, 0) = _tmp111 * _tmp42 - _tmp112 * _tmp64 - _tmp113 * _tmp49 + _tmp114 * _tmp56;
    _res_D_a(5, 1) = _tmp111 * _tmp56 - _tmp112 * _tmp49 + _tmp113 * _tmp64 - _tmp114 * _tmp42;
    _res_D_a(5, 2) = -_tmp111 * _tmp64 - _tmp112 * _tmp42 + _tmp113 * _tmp56 + _tmp114 * _tmp49;
    _res_D_a(5, 3) = _tmp16 + _tmp97;
    _res_D_a(5, 4) = _tmp110 + _tmp24;
    _res_D_a(5, 5) = _tmp19 + _tmp4 - 1;
  }

  if (res_D_b != nullptr) {
    Eigen::Matrix<Scalar, 6, 6>& _res_D_b = (*res_D_b);

    _res_D_b(0, 0) = _tmp121 + _tmp126 * _tmp127 + _tmp137;
    _res_D_b(0, 1) = -_tmp120 * _tmp135 + _tmp126 * _tmp129 + _tmp138 - _tmp139;
    _res_D_b(0, 2) = _tmp119 * _tmp129 - _tmp126 * _tmp136 + _tmp140 - _tmp141;
    _res_D_b(0, 3) = 0;
    _res_D_b(0, 4) = 0;
    _res_D_b(0, 5) = 0;
    _res_D_b(1, 0) = -_tmp120 * _tmp142 + _tmp129 * _tmp143 - _tmp138 + _tmp139;
    _res_D_b(1, 1) = _tmp121 + _tmp130 + _tmp136 * _tmp142 + _tmp145;
    _res_D_b(1, 2) = -_tmp119 * _tmp127 + _tmp129 * _tmp142 - _tmp146 + _tmp147;
    _res_D_b(1, 3) = 0;
    _res_D_b(1, 4) = 0;
    _res_D_b(1, 5) = 0;
    _res_D_b(2, 0) = _tmp129 * _tmp148 - _tmp136 * _tmp143 - _tmp140 + _tmp141;
    _res_D_b(2, 1) = -_tmp127 * _tmp148 + _tmp129 * _tmp135 + _tmp146 - _tmp147;
    _res_D_b(2, 2) = _tmp120 * _tmp148 + _tmp137 + _tmp145;
    _res_D_b(2, 3) = 0;
    _res_D_b(2, 4) = 0;
    _res_D_b(2, 5) = 0;
    _res_D_b(3, 0) = 0;
    _res_D_b(3, 1) = 0;
    _res_D_b(3, 2) = 0;
    _res_D_b(3, 3) = _tmp8;
    _res_D_b(3, 4) = _tmp13;
    _res_D_b(3, 5) = _tmp18;
    _res_D_b(4, 0) = 0;
    _res_D_b(4, 1) = 0;
    _res_D_b(4, 2) = 0;
    _res_D_b(4, 3) = _tmp23;
    _res_D_b(4, 4) = _tmp21;
    _res_D_b(4, 5) = _tmp26;
    _res_D_b(5, 0) = 0;
    _res_D_b(5, 1) = 0;
    _res_D_b(5, 2) = 0;
    _res_D_b(5, 3) = _tmp30;
    _res_D_b(5, 4) = _tmp29;
    _res_D_b(5, 5) = _tmp27;
  }

  return sym::Pose3<Scalar>(_res);
}

}  // namespace sym

// Explicit instantiation
template struct sym::GroupOps<sym::Pose3<double>>;
template struct sym::GroupOps<sym::Pose3<float>>;
