// -----------------------------------------------------------------------------
// This file was autogenerated by symforce from template:
//     backends/cpp/templates/function/FUNCTION.h.jinja
// Do NOT modify by hand.
// -----------------------------------------------------------------------------

#pragma once

#include <Eigen/Dense>

#include <sym/pose3.h>

namespace sym {

/**
 * Residual from a relative translation mesurement of a 3D pose to a landmark.
 *
 * Args:
 *     world_T_body: 3D pose of the robot in the world frame
 *     world_t_landmark: World location of the landmark
 *     body_t_landmark: Measured body-frame location of the landmark
 *     sigma: Isotropic standard deviation of the measurement [m]
 *     jacobian: (3x6) jacobian of res wrt arg world_T_body (6)
 *     hessian: (6x6) Gauss-Newton hessian for arg world_T_body (6)
 *     rhs: (6x1) Gauss-Newton rhs for arg world_T_body (6)
 */
template <typename Scalar>
void MatchingFactor(const sym::Pose3<Scalar>& world_T_body,
                    const Eigen::Matrix<Scalar, 3, 1>& world_t_landmark,
                    const Eigen::Matrix<Scalar, 3, 1>& body_t_landmark, const Scalar sigma,
                    Eigen::Matrix<Scalar, 3, 1>* const res = nullptr,
                    Eigen::Matrix<Scalar, 3, 6>* const jacobian = nullptr,
                    Eigen::Matrix<Scalar, 6, 6>* const hessian = nullptr,
                    Eigen::Matrix<Scalar, 6, 1>* const rhs = nullptr) {
  // Total ops: 307

  // Input arrays
  const Eigen::Matrix<Scalar, 7, 1>& _world_T_body = world_T_body.Data();

  // Intermediate terms (103)
  const Scalar _tmp0 = Scalar(1.0) / (sigma);
  const Scalar _tmp1 = std::pow(_world_T_body[2], Scalar(2));
  const Scalar _tmp2 = -2 * _tmp1;
  const Scalar _tmp3 = std::pow(_world_T_body[1], Scalar(2));
  const Scalar _tmp4 = 1 - 2 * _tmp3;
  const Scalar _tmp5 = _tmp2 + _tmp4;
  const Scalar _tmp6 = _world_T_body[2] * _world_T_body[3];
  const Scalar _tmp7 = 2 * _tmp6;
  const Scalar _tmp8 = _world_T_body[0] * _world_T_body[1];
  const Scalar _tmp9 = 2 * _tmp8;
  const Scalar _tmp10 = _tmp7 + _tmp9;
  const Scalar _tmp11 = _world_T_body[0] * _world_T_body[2];
  const Scalar _tmp12 = 2 * _tmp11;
  const Scalar _tmp13 = _world_T_body[1] * _world_T_body[3];
  const Scalar _tmp14 = 2 * _tmp13;
  const Scalar _tmp15 = _tmp12 - _tmp14;
  const Scalar _tmp16 = -_tmp10 * _world_T_body[5] + _tmp10 * world_t_landmark(1, 0) -
                        _tmp15 * _world_T_body[6] + _tmp15 * world_t_landmark(2, 0) -
                        _tmp5 * _world_T_body[4] + _tmp5 * world_t_landmark(0, 0) -
                        body_t_landmark(0, 0);
  const Scalar _tmp17 = std::pow(_world_T_body[0], Scalar(2));
  const Scalar _tmp18 = -2 * _tmp17;
  const Scalar _tmp19 = _tmp18 + _tmp2 + 1;
  const Scalar _tmp20 = -_tmp7 + _tmp9;
  const Scalar _tmp21 = _world_T_body[1] * _world_T_body[2];
  const Scalar _tmp22 = 2 * _tmp21;
  const Scalar _tmp23 = _world_T_body[0] * _world_T_body[3];
  const Scalar _tmp24 = 2 * _tmp23;
  const Scalar _tmp25 = _tmp22 + _tmp24;
  const Scalar _tmp26 = -_tmp19 * _world_T_body[5] + _tmp19 * world_t_landmark(1, 0) -
                        _tmp20 * _world_T_body[4] + _tmp20 * world_t_landmark(0, 0) -
                        _tmp25 * _world_T_body[6] + _tmp25 * world_t_landmark(2, 0) -
                        body_t_landmark(1, 0);
  const Scalar _tmp27 = _tmp18 + _tmp4;
  const Scalar _tmp28 = _tmp22 - _tmp24;
  const Scalar _tmp29 = _tmp12 + _tmp14;
  const Scalar _tmp30 = -_tmp27 * _world_T_body[6] + _tmp27 * world_t_landmark(2, 0) -
                        _tmp28 * _world_T_body[5] + _tmp28 * world_t_landmark(1, 0) -
                        _tmp29 * _world_T_body[4] + _tmp29 * world_t_landmark(0, 0) -
                        body_t_landmark(2, 0);
  const Scalar _tmp31 = _tmp11 + _tmp13;
  const Scalar _tmp32 = 2 * _world_T_body[4];
  const Scalar _tmp33 = _tmp31 * _tmp32;
  const Scalar _tmp34 = _tmp21 - _tmp23;
  const Scalar _tmp35 = 2 * _world_T_body[5];
  const Scalar _tmp36 = _tmp34 * _tmp35;
  const Scalar _tmp37 = -_tmp1;
  const Scalar _tmp38 = std::pow(_world_T_body[3], Scalar(2));
  const Scalar _tmp39 = _tmp17 - _tmp38;
  const Scalar _tmp40 = _tmp3 + _tmp37 + _tmp39;
  const Scalar _tmp41 = _tmp40 * _world_T_body[6];
  const Scalar _tmp42 = _tmp40 * world_t_landmark(2, 0);
  const Scalar _tmp43 = 2 * world_t_landmark(1, 0);
  const Scalar _tmp44 = _tmp34 * _tmp43;
  const Scalar _tmp45 = 2 * world_t_landmark(0, 0);
  const Scalar _tmp46 = _tmp31 * _tmp45;
  const Scalar _tmp47 = -_tmp33 - _tmp36 + _tmp41 - _tmp42 + _tmp44 + _tmp46;
  const Scalar _tmp48 = _tmp21 + _tmp23;
  const Scalar _tmp49 = 2 * _world_T_body[6];
  const Scalar _tmp50 = _tmp48 * _tmp49;
  const Scalar _tmp51 = _tmp6 - _tmp8;
  const Scalar _tmp52 = _tmp32 * _tmp51;
  const Scalar _tmp53 = -_tmp3;
  const Scalar _tmp54 = _tmp1 + _tmp39 + _tmp53;
  const Scalar _tmp55 = _tmp54 * _world_T_body[5];
  const Scalar _tmp56 = _tmp54 * world_t_landmark(1, 0);
  const Scalar _tmp57 = _tmp45 * _tmp51;
  const Scalar _tmp58 = 2 * world_t_landmark(2, 0);
  const Scalar _tmp59 = _tmp48 * _tmp58;
  const Scalar _tmp60 = _tmp50 - _tmp52 - _tmp55 + _tmp56 + _tmp57 - _tmp59;
  const Scalar _tmp61 = _tmp33 + _tmp36 - _tmp41 + _tmp42 - _tmp44 - _tmp46;
  const Scalar _tmp62 = _tmp11 - _tmp13;
  const Scalar _tmp63 = _tmp49 * _tmp62;
  const Scalar _tmp64 = _tmp6 + _tmp8;
  const Scalar _tmp65 = _tmp35 * _tmp64;
  const Scalar _tmp66 = _tmp17 + _tmp37 + _tmp38 + _tmp53;
  const Scalar _tmp67 = _tmp66 * _world_T_body[4];
  const Scalar _tmp68 = _tmp66 * world_t_landmark(0, 0);
  const Scalar _tmp69 = _tmp43 * _tmp64;
  const Scalar _tmp70 = _tmp58 * _tmp62;
  const Scalar _tmp71 = -_tmp63 - _tmp65 - _tmp67 + _tmp68 + _tmp69 + _tmp70;
  const Scalar _tmp72 = -_tmp50 + _tmp52 + _tmp55 - _tmp56 - _tmp57 + _tmp59;
  const Scalar _tmp73 = _tmp63 + _tmp65 + _tmp67 - _tmp68 - _tmp69 - _tmp70;
  const Scalar _tmp74 = _tmp1 + _tmp3 + Scalar(-1) / Scalar(2);
  const Scalar _tmp75 = 2 * _tmp0;
  const Scalar _tmp76 = _tmp17 + Scalar(-1) / Scalar(2);
  const Scalar _tmp77 = _tmp1 + _tmp76;
  const Scalar _tmp78 = _tmp3 + _tmp76;
  const Scalar _tmp79 = std::pow(sigma, Scalar(-2));
  const Scalar _tmp80 = _tmp71 * _tmp79;
  const Scalar _tmp81 = _tmp73 * _tmp79;
  const Scalar _tmp82 = _tmp60 * _tmp79;
  const Scalar _tmp83 = 2 * _tmp31;
  const Scalar _tmp84 = 2 * _tmp47 * _tmp79;
  const Scalar _tmp85 = 2 * _tmp34;
  const Scalar _tmp86 = 2 * _tmp78;
  const Scalar _tmp87 = _tmp72 * _tmp79;
  const Scalar _tmp88 = 2 * _tmp74;
  const Scalar _tmp89 = _tmp61 * _tmp79;
  const Scalar _tmp90 = 2 * _tmp89;
  const Scalar _tmp91 = 2 * _tmp81;
  const Scalar _tmp92 = 2 * _tmp87;
  const Scalar _tmp93 = 4 * _tmp79;
  const Scalar _tmp94 = _tmp51 * _tmp93;
  const Scalar _tmp95 = _tmp34 * _tmp93;
  const Scalar _tmp96 = _tmp62 * _tmp93;
  const Scalar _tmp97 = _tmp30 * _tmp79;
  const Scalar _tmp98 = _tmp26 * _tmp79;
  const Scalar _tmp99 = _tmp16 * _tmp79;
  const Scalar _tmp100 = 2 * _tmp97;
  const Scalar _tmp101 = 2 * _tmp99;
  const Scalar _tmp102 = 2 * _tmp98;

  // Output terms (4)
  if (res != nullptr) {
    Eigen::Matrix<Scalar, 3, 1>& _res = (*res);

    _res(0, 0) = _tmp0 * _tmp16;
    _res(1, 0) = _tmp0 * _tmp26;
    _res(2, 0) = _tmp0 * _tmp30;
  }

  if (jacobian != nullptr) {
    Eigen::Matrix<Scalar, 3, 6>& _jacobian = (*jacobian);

    _jacobian(0, 0) = 0;
    _jacobian(1, 0) = _tmp0 * _tmp47;
    _jacobian(2, 0) = _tmp0 * _tmp60;
    _jacobian(0, 1) = _tmp0 * _tmp61;
    _jacobian(1, 1) = 0;
    _jacobian(2, 1) = _tmp0 * _tmp71;
    _jacobian(0, 2) = _tmp0 * _tmp72;
    _jacobian(1, 2) = _tmp0 * _tmp73;
    _jacobian(2, 2) = 0;
    _jacobian(0, 3) = _tmp74 * _tmp75;
    _jacobian(1, 3) = _tmp51 * _tmp75;
    _jacobian(2, 3) = -_tmp31 * _tmp75;
    _jacobian(0, 4) = -_tmp64 * _tmp75;
    _jacobian(1, 4) = _tmp75 * _tmp77;
    _jacobian(2, 4) = -_tmp34 * _tmp75;
    _jacobian(0, 5) = -_tmp62 * _tmp75;
    _jacobian(1, 5) = -_tmp48 * _tmp75;
    _jacobian(2, 5) = _tmp75 * _tmp78;
  }

  if (hessian != nullptr) {
    Eigen::Matrix<Scalar, 6, 6>& _hessian = (*hessian);

    _hessian(0, 0) = std::pow(_tmp47, Scalar(2)) * _tmp79 + std::pow(_tmp60, Scalar(2)) * _tmp79;
    _hessian(1, 0) = _tmp60 * _tmp80;
    _hessian(2, 0) = _tmp47 * _tmp81;
    _hessian(3, 0) = _tmp51 * _tmp84 - _tmp82 * _tmp83;
    _hessian(4, 0) = _tmp77 * _tmp84 - _tmp82 * _tmp85;
    _hessian(5, 0) = -_tmp48 * _tmp84 + _tmp82 * _tmp86;
    _hessian(0, 1) = 0;
    _hessian(1, 1) = std::pow(_tmp61, Scalar(2)) * _tmp79 + std::pow(_tmp71, Scalar(2)) * _tmp79;
    _hessian(2, 1) = _tmp61 * _tmp87;
    _hessian(3, 1) = -_tmp80 * _tmp83 + _tmp88 * _tmp89;
    _hessian(4, 1) = -_tmp64 * _tmp90 - _tmp80 * _tmp85;
    _hessian(5, 1) = -_tmp62 * _tmp90 + _tmp80 * _tmp86;
    _hessian(0, 2) = 0;
    _hessian(1, 2) = 0;
    _hessian(2, 2) = std::pow(_tmp72, Scalar(2)) * _tmp79 + std::pow(_tmp73, Scalar(2)) * _tmp79;
    _hessian(3, 2) = _tmp51 * _tmp91 + _tmp87 * _tmp88;
    _hessian(4, 2) = -_tmp64 * _tmp92 + _tmp77 * _tmp91;
    _hessian(5, 2) = -_tmp48 * _tmp91 - _tmp62 * _tmp92;
    _hessian(0, 3) = 0;
    _hessian(1, 3) = 0;
    _hessian(2, 3) = 0;
    _hessian(3, 3) = std::pow(_tmp31, Scalar(2)) * _tmp93 + std::pow(_tmp51, Scalar(2)) * _tmp93 +
                     std::pow(_tmp74, Scalar(2)) * _tmp93;
    _hessian(4, 3) = _tmp31 * _tmp95 - _tmp64 * _tmp74 * _tmp93 + _tmp77 * _tmp94;
    _hessian(5, 3) = -_tmp31 * _tmp78 * _tmp93 - _tmp48 * _tmp94 - _tmp74 * _tmp96;
    _hessian(0, 4) = 0;
    _hessian(1, 4) = 0;
    _hessian(2, 4) = 0;
    _hessian(3, 4) = 0;
    _hessian(4, 4) = std::pow(_tmp34, Scalar(2)) * _tmp93 + std::pow(_tmp64, Scalar(2)) * _tmp93 +
                     std::pow(_tmp77, Scalar(2)) * _tmp93;
    _hessian(5, 4) = -_tmp48 * _tmp77 * _tmp93 + _tmp64 * _tmp96 - _tmp78 * _tmp95;
    _hessian(0, 5) = 0;
    _hessian(1, 5) = 0;
    _hessian(2, 5) = 0;
    _hessian(3, 5) = 0;
    _hessian(4, 5) = 0;
    _hessian(5, 5) = std::pow(_tmp48, Scalar(2)) * _tmp93 + std::pow(_tmp62, Scalar(2)) * _tmp93 +
                     std::pow(_tmp78, Scalar(2)) * _tmp93;
  }

  if (rhs != nullptr) {
    Eigen::Matrix<Scalar, 6, 1>& _rhs = (*rhs);

    _rhs(0, 0) = _tmp47 * _tmp98 + _tmp60 * _tmp97;
    _rhs(1, 0) = _tmp30 * _tmp80 + _tmp61 * _tmp99;
    _rhs(2, 0) = _tmp26 * _tmp81 + _tmp72 * _tmp99;
    _rhs(3, 0) = -_tmp100 * _tmp31 + _tmp101 * _tmp74 + _tmp102 * _tmp51;
    _rhs(4, 0) = -_tmp100 * _tmp34 - _tmp101 * _tmp64 + _tmp102 * _tmp77;
    _rhs(5, 0) = _tmp100 * _tmp78 - _tmp101 * _tmp62 - _tmp102 * _tmp48;
  }
}  // NOLINT(readability/fn_size)

// NOLINTNEXTLINE(readability/fn_size)
}  // namespace sym
