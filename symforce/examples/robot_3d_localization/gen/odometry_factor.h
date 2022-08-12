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
 * Residual on the relative pose between two timesteps of the robot.
 *
 * Args:
 *     world_T_a: First pose in the world frame
 *     world_T_b: Second pose in the world frame
 *     a_T_b: Relative pose measurement between the poses
 *     diagonal_sigmas: Diagonal standard deviation of the tangent-space error
 *     epsilon: Small number for singularity handling
 *     jacobian: (6x12) jacobian of res wrt args world_T_a (6), world_T_b (6)
 *     hessian: (12x12) Gauss-Newton hessian for args world_T_a (6), world_T_b (6)
 *     rhs: (12x1) Gauss-Newton rhs for args world_T_a (6), world_T_b (6)
 */
template <typename Scalar>
void OdometryFactor(const sym::Pose3<Scalar>& world_T_a, const sym::Pose3<Scalar>& world_T_b,
                    const sym::Pose3<Scalar>& a_T_b,
                    const Eigen::Matrix<Scalar, 6, 1>& diagonal_sigmas, const Scalar epsilon,
                    Eigen::Matrix<Scalar, 6, 1>* const res = nullptr,
                    Eigen::Matrix<Scalar, 6, 12>* const jacobian = nullptr,
                    Eigen::Matrix<Scalar, 12, 12>* const hessian = nullptr,
                    Eigen::Matrix<Scalar, 12, 1>* const rhs = nullptr) {
  // Total ops: 762

  // Input arrays
  const Eigen::Matrix<Scalar, 7, 1>& _world_T_a = world_T_a.Data();
  const Eigen::Matrix<Scalar, 7, 1>& _world_T_b = world_T_b.Data();
  const Eigen::Matrix<Scalar, 7, 1>& _a_T_b = a_T_b.Data();

  // Intermediate terms (282)
  const Scalar _tmp0 = Scalar(1.0) / (diagonal_sigmas(0, 0));
  const Scalar _tmp1 = _world_T_a[3] * _world_T_b[3];
  const Scalar _tmp2 = _world_T_a[0] * _world_T_b[0];
  const Scalar _tmp3 = _world_T_a[1] * _world_T_b[1];
  const Scalar _tmp4 = _world_T_a[2] * _world_T_b[2];
  const Scalar _tmp5 = _tmp1 + _tmp2 + _tmp3 + _tmp4;
  const Scalar _tmp6 = -_world_T_a[0] * _world_T_b[3] - _world_T_a[1] * _world_T_b[2] +
                       _world_T_a[2] * _world_T_b[1] + _world_T_a[3] * _world_T_b[0];
  const Scalar _tmp7 = _a_T_b[3] * _tmp6;
  const Scalar _tmp8 = -_tmp7;
  const Scalar _tmp9 = -_world_T_a[0] * _world_T_b[1] + _world_T_a[1] * _world_T_b[0] -
                       _world_T_a[2] * _world_T_b[3] + _world_T_a[3] * _world_T_b[2];
  const Scalar _tmp10 = _a_T_b[1] * _tmp9;
  const Scalar _tmp11 = _world_T_a[0] * _world_T_b[2] - _world_T_a[1] * _world_T_b[3] -
                        _world_T_a[2] * _world_T_b[0] + _world_T_a[3] * _world_T_b[1];
  const Scalar _tmp12 = _a_T_b[2] * _tmp11;
  const Scalar _tmp13 = _tmp10 - _tmp12;
  const Scalar _tmp14 = _tmp13 + _tmp8;
  const Scalar _tmp15 = _a_T_b[2] * _tmp9;
  const Scalar _tmp16 = _a_T_b[0] * _tmp6;
  const Scalar _tmp17 = -_tmp16;
  const Scalar _tmp18 = _a_T_b[1] * _tmp11;
  const Scalar _tmp19 = -_tmp18;
  const Scalar _tmp20 = _tmp17 + _tmp19;
  const Scalar _tmp21 = -_tmp15 + _tmp20;
  const Scalar _tmp22 = _a_T_b[3] * _tmp5;
  const Scalar _tmp23 = std::min<Scalar>(1 - epsilon, std::fabs(_tmp21 - _tmp22));
  const Scalar _tmp24 =
      2 * (2 * std::min<Scalar>(0, (((-_tmp21 + _tmp22) > 0) - ((-_tmp21 + _tmp22) < 0))) + 1) *
      std::acos(_tmp23) / std::sqrt(Scalar(1 - std::pow(_tmp23, Scalar(2))));
  const Scalar _tmp25 = _tmp0 * _tmp24 * (_a_T_b[0] * _tmp5 + _tmp14);
  const Scalar _tmp26 = Scalar(1.0) / (diagonal_sigmas(1, 0));
  const Scalar _tmp27 = _a_T_b[0] * _tmp9;
  const Scalar _tmp28 = _a_T_b[3] * _tmp11;
  const Scalar _tmp29 = -_tmp28;
  const Scalar _tmp30 = _a_T_b[2] * _tmp6;
  const Scalar _tmp31 = _tmp29 + _tmp30;
  const Scalar _tmp32 = _tmp24 * _tmp26 * (_a_T_b[1] * _tmp5 - _tmp27 + _tmp31);
  const Scalar _tmp33 = Scalar(1.0) / (diagonal_sigmas(2, 0));
  const Scalar _tmp34 = _a_T_b[3] * _tmp9;
  const Scalar _tmp35 = _a_T_b[1] * _tmp6;
  const Scalar _tmp36 = -_tmp35;
  const Scalar _tmp37 = _a_T_b[0] * _tmp11;
  const Scalar _tmp38 = _tmp36 + _tmp37;
  const Scalar _tmp39 = _tmp24 * _tmp33 * (_a_T_b[2] * _tmp5 - _tmp34 + _tmp38);
  const Scalar _tmp40 = Scalar(1.0) / (diagonal_sigmas(3, 0));
  const Scalar _tmp41 = _world_T_a[0] * _world_T_a[1];
  const Scalar _tmp42 = 2 * _tmp41;
  const Scalar _tmp43 = _world_T_a[2] * _world_T_a[3];
  const Scalar _tmp44 = 2 * _tmp43;
  const Scalar _tmp45 = _tmp42 + _tmp44;
  const Scalar _tmp46 = _world_T_a[1] * _world_T_a[3];
  const Scalar _tmp47 = 2 * _tmp46;
  const Scalar _tmp48 = _world_T_a[0] * _world_T_a[2];
  const Scalar _tmp49 = 2 * _tmp48;
  const Scalar _tmp50 = -_tmp47 + _tmp49;
  const Scalar _tmp51 = std::pow(_world_T_a[1], Scalar(2));
  const Scalar _tmp52 = -2 * _tmp51;
  const Scalar _tmp53 = std::pow(_world_T_a[2], Scalar(2));
  const Scalar _tmp54 = -2 * _tmp53;
  const Scalar _tmp55 = _tmp52 + _tmp54 + 1;
  const Scalar _tmp56 = _a_T_b[4] + _tmp45 * _world_T_a[5] - _tmp45 * _world_T_b[5] +
                        _tmp50 * _world_T_a[6] - _tmp50 * _world_T_b[6] + _tmp55 * _world_T_a[4] -
                        _tmp55 * _world_T_b[4];
  const Scalar _tmp57 = Scalar(1.0) / (diagonal_sigmas(4, 0));
  const Scalar _tmp58 = _tmp42 - _tmp44;
  const Scalar _tmp59 = _world_T_a[0] * _world_T_a[3];
  const Scalar _tmp60 = 2 * _tmp59;
  const Scalar _tmp61 = _world_T_a[1] * _world_T_a[2];
  const Scalar _tmp62 = 2 * _tmp61;
  const Scalar _tmp63 = _tmp60 + _tmp62;
  const Scalar _tmp64 = std::pow(_world_T_a[0], Scalar(2));
  const Scalar _tmp65 = 1 - 2 * _tmp64;
  const Scalar _tmp66 = _tmp54 + _tmp65;
  const Scalar _tmp67 = _a_T_b[5] + _tmp58 * _world_T_a[4] - _tmp58 * _world_T_b[4] +
                        _tmp63 * _world_T_a[6] - _tmp63 * _world_T_b[6] + _tmp66 * _world_T_a[5] -
                        _tmp66 * _world_T_b[5];
  const Scalar _tmp68 = Scalar(1.0) / (diagonal_sigmas(5, 0));
  const Scalar _tmp69 = _tmp47 + _tmp49;
  const Scalar _tmp70 = -_tmp60 + _tmp62;
  const Scalar _tmp71 = _tmp52 + _tmp65;
  const Scalar _tmp72 = _a_T_b[6] + _tmp69 * _world_T_a[4] - _tmp69 * _world_T_b[4] +
                        _tmp70 * _world_T_a[5] - _tmp70 * _world_T_b[5] + _tmp71 * _world_T_a[6] -
                        _tmp71 * _world_T_b[6];
  const Scalar _tmp73 = _tmp1 + _tmp2 + _tmp3 + _tmp4;
  const Scalar _tmp74 = _a_T_b[3] * _tmp73;
  const Scalar _tmp75 = -_tmp74;
  const Scalar _tmp76 = _tmp15 + _tmp16;
  const Scalar _tmp77 = _tmp18 + _tmp74 + _tmp76;
  const Scalar _tmp78 = (((_tmp77) > 0) - ((_tmp77) < 0));
  const Scalar _tmp79 = std::min<Scalar>(0, _tmp78) + Scalar(1) / Scalar(2);
  const Scalar _tmp80 = std::fabs(_tmp77);
  const Scalar _tmp81 = epsilon - 1;
  const Scalar _tmp82 = std::min<Scalar>(_tmp80, -_tmp81);
  const Scalar _tmp83 = std::pow(_tmp82, Scalar(2)) - 1;
  const Scalar _tmp84 = -_tmp83;
  const Scalar _tmp85 = std::acos(_tmp82);
  const Scalar _tmp86 = 2 * _tmp85 / std::sqrt(_tmp84);
  const Scalar _tmp87 = _tmp79 * _tmp86;
  const Scalar _tmp88 = _tmp0 * _tmp87;
  const Scalar _tmp89 = _a_T_b[0] * _tmp73;
  const Scalar _tmp90 = -_tmp89;
  const Scalar _tmp91 = _tmp13 + _tmp7 + _tmp90;
  const Scalar _tmp92 = _tmp14 + _tmp89;
  const Scalar _tmp93 = _tmp78 * ((((_tmp80 + _tmp81) > 0) - ((_tmp80 + _tmp81) < 0)) - 1);
  const Scalar _tmp94 = _tmp93 / _tmp83;
  const Scalar _tmp95 = _tmp79 * _tmp94;
  const Scalar _tmp96 = _tmp0 * _tmp95;
  const Scalar _tmp97 = _tmp92 * _tmp96;
  const Scalar _tmp98 = _tmp82 * _tmp85 * _tmp93 / (_tmp84 * std::sqrt(_tmp84));
  const Scalar _tmp99 = _tmp79 * _tmp98;
  const Scalar _tmp100 = _tmp0 * _tmp99;
  const Scalar _tmp101 = _tmp100 * _tmp92;
  const Scalar _tmp102 =
      -_tmp101 * _tmp91 - _tmp88 * (_tmp15 + _tmp17 + _tmp18 + _tmp75) - _tmp91 * _tmp97;
  const Scalar _tmp103 = _a_T_b[1] * _tmp73;
  const Scalar _tmp104 = -_tmp103;
  const Scalar _tmp105 = _tmp27 - _tmp30;
  const Scalar _tmp106 = _tmp104 + _tmp105 + _tmp28;
  const Scalar _tmp107 = _tmp26 * _tmp79;
  const Scalar _tmp108 = _tmp107 * _tmp94;
  const Scalar _tmp109 = _tmp106 * _tmp108;
  const Scalar _tmp110 = -_tmp37;
  const Scalar _tmp111 = _a_T_b[2] * _tmp73;
  const Scalar _tmp112 = _tmp111 + _tmp34;
  const Scalar _tmp113 = _tmp107 * _tmp86;
  const Scalar _tmp114 = _tmp107 * _tmp98;
  const Scalar _tmp115 = _tmp106 * _tmp114;
  const Scalar _tmp116 =
      _tmp109 * _tmp91 - _tmp113 * (_tmp110 + _tmp112 + _tmp36) + _tmp115 * _tmp91;
  const Scalar _tmp117 = -_tmp111 + _tmp34;
  const Scalar _tmp118 = _tmp110 + _tmp117 + _tmp35;
  const Scalar _tmp119 = _tmp33 * _tmp99;
  const Scalar _tmp120 = _tmp118 * _tmp119;
  const Scalar _tmp121 = _tmp33 * _tmp95;
  const Scalar _tmp122 = _tmp118 * _tmp121;
  const Scalar _tmp123 = _tmp33 * _tmp87;
  const Scalar _tmp124 =
      _tmp120 * _tmp91 + _tmp122 * _tmp91 + _tmp123 * (_tmp103 + _tmp27 + _tmp28 + _tmp30);
  const Scalar _tmp125 = _tmp46 + _tmp48;
  const Scalar _tmp126 = 2 * _tmp125;
  const Scalar _tmp127 = _tmp126 * _world_T_b[4];
  const Scalar _tmp128 = _tmp126 * _world_T_a[4];
  const Scalar _tmp129 = _tmp59 - _tmp61;
  const Scalar _tmp130 = 2 * _tmp129;
  const Scalar _tmp131 = _tmp130 * _world_T_a[5];
  const Scalar _tmp132 = _tmp130 * _world_T_b[5];
  const Scalar _tmp133 = -_tmp51;
  const Scalar _tmp134 = std::pow(_world_T_a[3], Scalar(2));
  const Scalar _tmp135 = _tmp134 - _tmp64;
  const Scalar _tmp136 = _tmp133 + _tmp135 + _tmp53;
  const Scalar _tmp137 = _tmp136 * _world_T_b[6];
  const Scalar _tmp138 = _tmp136 * _world_T_a[6];
  const Scalar _tmp139 = -_tmp127 + _tmp128 - _tmp131 + _tmp132 - _tmp137 + _tmp138;
  const Scalar _tmp140 = _tmp41 - _tmp43;
  const Scalar _tmp141 = 2 * _tmp140;
  const Scalar _tmp142 = _tmp141 * _world_T_a[4];
  const Scalar _tmp143 = _tmp141 * _world_T_b[4];
  const Scalar _tmp144 = _tmp59 + _tmp61;
  const Scalar _tmp145 = 2 * _tmp144;
  const Scalar _tmp146 = _tmp145 * _world_T_a[6];
  const Scalar _tmp147 = _tmp145 * _world_T_b[6];
  const Scalar _tmp148 = -_tmp53;
  const Scalar _tmp149 = _tmp135 + _tmp148 + _tmp51;
  const Scalar _tmp150 = _tmp149 * _world_T_b[5];
  const Scalar _tmp151 = _tmp149 * _world_T_a[5];
  const Scalar _tmp152 = -_tmp142 + _tmp143 - _tmp146 + _tmp147 + _tmp150 - _tmp151;
  const Scalar _tmp153 = _tmp103 + _tmp105 + _tmp29;
  const Scalar _tmp154 =
      _tmp101 * _tmp153 + _tmp153 * _tmp97 + _tmp88 * (_tmp112 + _tmp35 + _tmp37);
  const Scalar _tmp155 =
      -_tmp109 * _tmp153 - _tmp113 * (_tmp19 + _tmp75 + _tmp76) - _tmp115 * _tmp153;
  const Scalar _tmp156 = _tmp10 + _tmp12;
  const Scalar _tmp157 =
      -_tmp120 * _tmp153 - _tmp122 * _tmp153 + _tmp123 * (_tmp156 + _tmp8 + _tmp90);
  const Scalar _tmp158 = _tmp127 - _tmp128 + _tmp131 - _tmp132 + _tmp137 - _tmp138;
  const Scalar _tmp159 = _tmp41 + _tmp43;
  const Scalar _tmp160 = 2 * _tmp159;
  const Scalar _tmp161 = _tmp160 * _world_T_b[5];
  const Scalar _tmp162 = _tmp160 * _world_T_a[5];
  const Scalar _tmp163 = _tmp46 - _tmp48;
  const Scalar _tmp164 = 2 * _tmp163;
  const Scalar _tmp165 = _tmp164 * _world_T_b[6];
  const Scalar _tmp166 = _tmp164 * _world_T_a[6];
  const Scalar _tmp167 = _tmp133 + _tmp134 + _tmp148 + _tmp64;
  const Scalar _tmp168 = _tmp167 * _world_T_b[4];
  const Scalar _tmp169 = _tmp167 * _world_T_a[4];
  const Scalar _tmp170 = -_tmp161 + _tmp162 + _tmp165 - _tmp166 - _tmp168 + _tmp169;
  const Scalar _tmp171 = _tmp117 + _tmp38;
  const Scalar _tmp172 =
      -_tmp101 * _tmp171 - _tmp171 * _tmp97 + _tmp88 * (_tmp104 + _tmp27 + _tmp31);
  const Scalar _tmp173 =
      _tmp109 * _tmp171 + _tmp113 * (_tmp156 + _tmp7 + _tmp89) + _tmp115 * _tmp171;
  const Scalar _tmp174 =
      _tmp120 * _tmp171 + _tmp122 * _tmp171 + _tmp123 * (_tmp15 + _tmp20 + _tmp74);
  const Scalar _tmp175 = _tmp142 - _tmp143 + _tmp146 - _tmp147 - _tmp150 + _tmp151;
  const Scalar _tmp176 = _tmp161 - _tmp162 - _tmp165 + _tmp166 + _tmp168 - _tmp169;
  const Scalar _tmp177 = _tmp53 + Scalar(-1) / Scalar(2);
  const Scalar _tmp178 = _tmp177 + _tmp51;
  const Scalar _tmp179 = 2 * _tmp40;
  const Scalar _tmp180 = _tmp178 * _tmp179;
  const Scalar _tmp181 = 2 * _tmp57;
  const Scalar _tmp182 = _tmp140 * _tmp181;
  const Scalar _tmp183 = 2 * _tmp68;
  const Scalar _tmp184 = _tmp125 * _tmp183;
  const Scalar _tmp185 = _tmp159 * _tmp179;
  const Scalar _tmp186 = _tmp177 + _tmp64;
  const Scalar _tmp187 = _tmp181 * _tmp186;
  const Scalar _tmp188 = _tmp129 * _tmp183;
  const Scalar _tmp189 = _tmp163 * _tmp179;
  const Scalar _tmp190 = _tmp144 * _tmp181;
  const Scalar _tmp191 = _tmp51 + _tmp64 + Scalar(-1) / Scalar(2);
  const Scalar _tmp192 = _tmp183 * _tmp191;
  const Scalar _tmp193 = std::pow(_tmp92, Scalar(2));
  const Scalar _tmp194 = -_tmp100 * _tmp193 - _tmp193 * _tmp96 - _tmp77 * _tmp88;
  const Scalar _tmp195 = _tmp107 * _tmp92;
  const Scalar _tmp196 = _tmp106 * _tmp195;
  const Scalar _tmp197 = -_tmp113 * _tmp118 + _tmp196 * _tmp94 + _tmp196 * _tmp98;
  const Scalar _tmp198 = _tmp106 * _tmp123 + _tmp120 * _tmp92 + _tmp122 * _tmp92;
  const Scalar _tmp199 = _tmp101 * _tmp106 + _tmp106 * _tmp97 + _tmp118 * _tmp88;
  const Scalar _tmp200 = std::pow(_tmp106, Scalar(2));
  const Scalar _tmp201 = -_tmp108 * _tmp200 - _tmp113 * _tmp77 - _tmp114 * _tmp200;
  const Scalar _tmp202 = -_tmp106 * _tmp120 - _tmp106 * _tmp122 + _tmp123 * _tmp92;
  const Scalar _tmp203 = _tmp101 * _tmp118 - _tmp106 * _tmp88 + _tmp118 * _tmp97;
  const Scalar _tmp204 = -_tmp109 * _tmp118 - _tmp115 * _tmp118 - _tmp195 * _tmp86;
  const Scalar _tmp205 = std::pow(_tmp118, Scalar(2));
  const Scalar _tmp206 = -_tmp119 * _tmp205 - _tmp121 * _tmp205 - _tmp123 * _tmp77;
  const Scalar _tmp207 = std::pow(diagonal_sigmas(5, 0), Scalar(-2));
  const Scalar _tmp208 = std::pow(diagonal_sigmas(4, 0), Scalar(-2));
  const Scalar _tmp209 = _tmp152 * _tmp207;
  const Scalar _tmp210 = _tmp139 * _tmp208;
  const Scalar _tmp211 = _tmp141 * _tmp210;
  const Scalar _tmp212 = _tmp126 * _tmp209;
  const Scalar _tmp213 = _tmp130 * _tmp209;
  const Scalar _tmp214 = 2 * _tmp186;
  const Scalar _tmp215 = _tmp210 * _tmp214;
  const Scalar _tmp216 = _tmp145 * _tmp210;
  const Scalar _tmp217 = 2 * _tmp191;
  const Scalar _tmp218 = _tmp209 * _tmp217;
  const Scalar _tmp219 = std::pow(diagonal_sigmas(3, 0), Scalar(-2));
  const Scalar _tmp220 = _tmp175 * _tmp219;
  const Scalar _tmp221 = _tmp170 * _tmp207;
  const Scalar _tmp222 = _tmp126 * _tmp221;
  const Scalar _tmp223 = _tmp158 * _tmp219;
  const Scalar _tmp224 = 2 * _tmp178;
  const Scalar _tmp225 = _tmp223 * _tmp224;
  const Scalar _tmp226 = _tmp160 * _tmp223;
  const Scalar _tmp227 = _tmp130 * _tmp221;
  const Scalar _tmp228 = _tmp164 * _tmp223;
  const Scalar _tmp229 = _tmp217 * _tmp221;
  const Scalar _tmp230 = _tmp176 * _tmp208;
  const Scalar _tmp231 = _tmp141 * _tmp230;
  const Scalar _tmp232 = _tmp220 * _tmp224;
  const Scalar _tmp233 = _tmp160 * _tmp220;
  const Scalar _tmp234 = _tmp214 * _tmp230;
  const Scalar _tmp235 = _tmp145 * _tmp230;
  const Scalar _tmp236 = _tmp164 * _tmp220;
  const Scalar _tmp237 = 4 * _tmp219;
  const Scalar _tmp238 = std::pow(_tmp178, Scalar(2)) * _tmp237;
  const Scalar _tmp239 = 4 * _tmp207;
  const Scalar _tmp240 = std::pow(_tmp125, Scalar(2)) * _tmp239;
  const Scalar _tmp241 = 4 * _tmp208;
  const Scalar _tmp242 = std::pow(_tmp140, Scalar(2)) * _tmp241;
  const Scalar _tmp243 = _tmp238 + _tmp240 + _tmp242;
  const Scalar _tmp244 = _tmp129 * _tmp239;
  const Scalar _tmp245 = _tmp125 * _tmp244;
  const Scalar _tmp246 = _tmp159 * _tmp237;
  const Scalar _tmp247 = _tmp178 * _tmp246;
  const Scalar _tmp248 = _tmp140 * _tmp186 * _tmp241;
  const Scalar _tmp249 = -_tmp245 - _tmp247 - _tmp248;
  const Scalar _tmp250 = _tmp163 * _tmp178 * _tmp237;
  const Scalar _tmp251 = _tmp125 * _tmp191 * _tmp239;
  const Scalar _tmp252 = _tmp144 * _tmp241;
  const Scalar _tmp253 = _tmp140 * _tmp252;
  const Scalar _tmp254 = _tmp250 - _tmp251 + _tmp253;
  const Scalar _tmp255 = _tmp245 + _tmp247 + _tmp248;
  const Scalar _tmp256 = -_tmp250 + _tmp251 - _tmp253;
  const Scalar _tmp257 = std::pow(_tmp186, Scalar(2)) * _tmp241;
  const Scalar _tmp258 = std::pow(_tmp129, Scalar(2)) * _tmp239;
  const Scalar _tmp259 = std::pow(_tmp159, Scalar(2)) * _tmp237;
  const Scalar _tmp260 = _tmp257 + _tmp258 + _tmp259;
  const Scalar _tmp261 = _tmp163 * _tmp246;
  const Scalar _tmp262 = _tmp186 * _tmp252;
  const Scalar _tmp263 = _tmp191 * _tmp244;
  const Scalar _tmp264 = -_tmp261 - _tmp262 + _tmp263;
  const Scalar _tmp265 = _tmp261 + _tmp262 - _tmp263;
  const Scalar _tmp266 = std::pow(_tmp191, Scalar(2)) * _tmp239;
  const Scalar _tmp267 = std::pow(_tmp144, Scalar(2)) * _tmp241;
  const Scalar _tmp268 = std::pow(_tmp163, Scalar(2)) * _tmp237;
  const Scalar _tmp269 = _tmp266 + _tmp267 + _tmp268;
  const Scalar _tmp270 = _tmp207 * _tmp72;
  const Scalar _tmp271 = _tmp126 * _tmp270;
  const Scalar _tmp272 = _tmp208 * _tmp67;
  const Scalar _tmp273 = _tmp141 * _tmp272;
  const Scalar _tmp274 = _tmp219 * _tmp56;
  const Scalar _tmp275 = _tmp224 * _tmp274;
  const Scalar _tmp276 = _tmp130 * _tmp270;
  const Scalar _tmp277 = _tmp160 * _tmp274;
  const Scalar _tmp278 = _tmp214 * _tmp272;
  const Scalar _tmp279 = _tmp145 * _tmp272;
  const Scalar _tmp280 = _tmp164 * _tmp274;
  const Scalar _tmp281 = _tmp217 * _tmp270;

  // Output terms (4)
  if (res != nullptr) {
    Eigen::Matrix<Scalar, 6, 1>& _res = (*res);

    _res(0, 0) = _tmp25;
    _res(1, 0) = _tmp32;
    _res(2, 0) = _tmp39;
    _res(3, 0) = _tmp40 * _tmp56;
    _res(4, 0) = _tmp57 * _tmp67;
    _res(5, 0) = _tmp68 * _tmp72;
  }

  if (jacobian != nullptr) {
    Eigen::Matrix<Scalar, 6, 12>& _jacobian = (*jacobian);

    _jacobian(0, 0) = _tmp102;
    _jacobian(1, 0) = _tmp116;
    _jacobian(2, 0) = _tmp124;
    _jacobian(3, 0) = 0;
    _jacobian(4, 0) = _tmp139 * _tmp57;
    _jacobian(5, 0) = _tmp152 * _tmp68;
    _jacobian(0, 1) = _tmp154;
    _jacobian(1, 1) = _tmp155;
    _jacobian(2, 1) = _tmp157;
    _jacobian(3, 1) = _tmp158 * _tmp40;
    _jacobian(4, 1) = 0;
    _jacobian(5, 1) = _tmp170 * _tmp68;
    _jacobian(0, 2) = _tmp172;
    _jacobian(1, 2) = _tmp173;
    _jacobian(2, 2) = _tmp174;
    _jacobian(3, 2) = _tmp175 * _tmp40;
    _jacobian(4, 2) = _tmp176 * _tmp57;
    _jacobian(5, 2) = 0;
    _jacobian(0, 3) = 0;
    _jacobian(1, 3) = 0;
    _jacobian(2, 3) = 0;
    _jacobian(3, 3) = -_tmp180;
    _jacobian(4, 3) = _tmp182;
    _jacobian(5, 3) = _tmp184;
    _jacobian(0, 4) = 0;
    _jacobian(1, 4) = 0;
    _jacobian(2, 4) = 0;
    _jacobian(3, 4) = _tmp185;
    _jacobian(4, 4) = -_tmp187;
    _jacobian(5, 4) = -_tmp188;
    _jacobian(0, 5) = 0;
    _jacobian(1, 5) = 0;
    _jacobian(2, 5) = 0;
    _jacobian(3, 5) = -_tmp189;
    _jacobian(4, 5) = _tmp190;
    _jacobian(5, 5) = -_tmp192;
    _jacobian(0, 6) = _tmp194;
    _jacobian(1, 6) = _tmp197;
    _jacobian(2, 6) = _tmp198;
    _jacobian(3, 6) = 0;
    _jacobian(4, 6) = 0;
    _jacobian(5, 6) = 0;
    _jacobian(0, 7) = _tmp199;
    _jacobian(1, 7) = _tmp201;
    _jacobian(2, 7) = _tmp202;
    _jacobian(3, 7) = 0;
    _jacobian(4, 7) = 0;
    _jacobian(5, 7) = 0;
    _jacobian(0, 8) = _tmp203;
    _jacobian(1, 8) = _tmp204;
    _jacobian(2, 8) = _tmp206;
    _jacobian(3, 8) = 0;
    _jacobian(4, 8) = 0;
    _jacobian(5, 8) = 0;
    _jacobian(0, 9) = 0;
    _jacobian(1, 9) = 0;
    _jacobian(2, 9) = 0;
    _jacobian(3, 9) = _tmp180;
    _jacobian(4, 9) = -_tmp182;
    _jacobian(5, 9) = -_tmp184;
    _jacobian(0, 10) = 0;
    _jacobian(1, 10) = 0;
    _jacobian(2, 10) = 0;
    _jacobian(3, 10) = -_tmp185;
    _jacobian(4, 10) = _tmp187;
    _jacobian(5, 10) = _tmp188;
    _jacobian(0, 11) = 0;
    _jacobian(1, 11) = 0;
    _jacobian(2, 11) = 0;
    _jacobian(3, 11) = _tmp189;
    _jacobian(4, 11) = -_tmp190;
    _jacobian(5, 11) = _tmp192;
  }

  if (hessian != nullptr) {
    Eigen::Matrix<Scalar, 12, 12>& _hessian = (*hessian);

    _hessian.setZero();

    _hessian(0, 0) = std::pow(_tmp102, Scalar(2)) + std::pow(_tmp116, Scalar(2)) +
                     std::pow(_tmp124, Scalar(2)) + std::pow(_tmp139, Scalar(2)) * _tmp208 +
                     std::pow(_tmp152, Scalar(2)) * _tmp207;
    _hessian(1, 0) = _tmp102 * _tmp154 + _tmp116 * _tmp155 + _tmp124 * _tmp157 + _tmp170 * _tmp209;
    _hessian(2, 0) = _tmp102 * _tmp172 + _tmp116 * _tmp173 + _tmp124 * _tmp174 + _tmp176 * _tmp210;
    _hessian(3, 0) = _tmp211 + _tmp212;
    _hessian(4, 0) = -_tmp213 - _tmp215;
    _hessian(5, 0) = _tmp216 - _tmp218;
    _hessian(6, 0) = _tmp102 * _tmp194 + _tmp116 * _tmp197 + _tmp124 * _tmp198;
    _hessian(7, 0) = _tmp102 * _tmp199 + _tmp116 * _tmp201 + _tmp124 * _tmp202;
    _hessian(8, 0) = _tmp102 * _tmp203 + _tmp116 * _tmp204 + _tmp124 * _tmp206;
    _hessian(9, 0) = -_tmp211 - _tmp212;
    _hessian(10, 0) = _tmp213 + _tmp215;
    _hessian(11, 0) = -_tmp216 + _tmp218;
    _hessian(1, 1) = std::pow(_tmp154, Scalar(2)) + std::pow(_tmp155, Scalar(2)) +
                     std::pow(_tmp157, Scalar(2)) + std::pow(_tmp158, Scalar(2)) * _tmp219 +
                     std::pow(_tmp170, Scalar(2)) * _tmp207;
    _hessian(2, 1) = _tmp154 * _tmp172 + _tmp155 * _tmp173 + _tmp157 * _tmp174 + _tmp158 * _tmp220;
    _hessian(3, 1) = _tmp222 - _tmp225;
    _hessian(4, 1) = _tmp226 - _tmp227;
    _hessian(5, 1) = -_tmp228 - _tmp229;
    _hessian(6, 1) = _tmp154 * _tmp194 + _tmp155 * _tmp197 + _tmp157 * _tmp198;
    _hessian(7, 1) = _tmp154 * _tmp199 + _tmp155 * _tmp201 + _tmp157 * _tmp202;
    _hessian(8, 1) = _tmp154 * _tmp203 + _tmp155 * _tmp204 + _tmp157 * _tmp206;
    _hessian(9, 1) = -_tmp222 + _tmp225;
    _hessian(10, 1) = -_tmp226 + _tmp227;
    _hessian(11, 1) = _tmp228 + _tmp229;
    _hessian(2, 2) = std::pow(_tmp172, Scalar(2)) + std::pow(_tmp173, Scalar(2)) +
                     std::pow(_tmp174, Scalar(2)) + std::pow(_tmp175, Scalar(2)) * _tmp219 +
                     std::pow(_tmp176, Scalar(2)) * _tmp208;
    _hessian(3, 2) = _tmp231 - _tmp232;
    _hessian(4, 2) = _tmp233 - _tmp234;
    _hessian(5, 2) = _tmp235 - _tmp236;
    _hessian(6, 2) = _tmp172 * _tmp194 + _tmp173 * _tmp197 + _tmp174 * _tmp198;
    _hessian(7, 2) = _tmp172 * _tmp199 + _tmp173 * _tmp201 + _tmp174 * _tmp202;
    _hessian(8, 2) = _tmp172 * _tmp203 + _tmp173 * _tmp204 + _tmp174 * _tmp206;
    _hessian(9, 2) = -_tmp231 + _tmp232;
    _hessian(10, 2) = -_tmp233 + _tmp234;
    _hessian(11, 2) = -_tmp235 + _tmp236;
    _hessian(3, 3) = _tmp243;
    _hessian(4, 3) = _tmp249;
    _hessian(5, 3) = _tmp254;
    _hessian(9, 3) = -_tmp238 - _tmp240 - _tmp242;
    _hessian(10, 3) = _tmp255;
    _hessian(11, 3) = _tmp256;
    _hessian(4, 4) = _tmp260;
    _hessian(5, 4) = _tmp264;
    _hessian(9, 4) = _tmp255;
    _hessian(10, 4) = -_tmp257 - _tmp258 - _tmp259;
    _hessian(11, 4) = _tmp265;
    _hessian(5, 5) = _tmp269;
    _hessian(9, 5) = _tmp256;
    _hessian(10, 5) = _tmp265;
    _hessian(11, 5) = -_tmp266 - _tmp267 - _tmp268;
    _hessian(6, 6) =
        std::pow(_tmp194, Scalar(2)) + std::pow(_tmp197, Scalar(2)) + std::pow(_tmp198, Scalar(2));
    _hessian(7, 6) = _tmp194 * _tmp199 + _tmp197 * _tmp201 + _tmp198 * _tmp202;
    _hessian(8, 6) = _tmp194 * _tmp203 + _tmp197 * _tmp204 + _tmp198 * _tmp206;
    _hessian(7, 7) =
        std::pow(_tmp199, Scalar(2)) + std::pow(_tmp201, Scalar(2)) + std::pow(_tmp202, Scalar(2));
    _hessian(8, 7) = _tmp199 * _tmp203 + _tmp201 * _tmp204 + _tmp202 * _tmp206;
    _hessian(8, 8) =
        std::pow(_tmp203, Scalar(2)) + std::pow(_tmp204, Scalar(2)) + std::pow(_tmp206, Scalar(2));
    _hessian(9, 9) = _tmp243;
    _hessian(10, 9) = _tmp249;
    _hessian(11, 9) = _tmp254;
    _hessian(10, 10) = _tmp260;
    _hessian(11, 10) = _tmp264;
    _hessian(11, 11) = _tmp269;
  }

  if (rhs != nullptr) {
    Eigen::Matrix<Scalar, 12, 1>& _rhs = (*rhs);

    _rhs(0, 0) = _tmp102 * _tmp25 + _tmp116 * _tmp32 + _tmp124 * _tmp39 + _tmp152 * _tmp270 +
                 _tmp210 * _tmp67;
    _rhs(1, 0) = _tmp154 * _tmp25 + _tmp155 * _tmp32 + _tmp157 * _tmp39 + _tmp170 * _tmp270 +
                 _tmp223 * _tmp56;
    _rhs(2, 0) = _tmp172 * _tmp25 + _tmp173 * _tmp32 + _tmp174 * _tmp39 + _tmp220 * _tmp56 +
                 _tmp230 * _tmp67;
    _rhs(3, 0) = _tmp271 + _tmp273 - _tmp275;
    _rhs(4, 0) = -_tmp276 + _tmp277 - _tmp278;
    _rhs(5, 0) = _tmp279 - _tmp280 - _tmp281;
    _rhs(6, 0) = _tmp194 * _tmp25 + _tmp197 * _tmp32 + _tmp198 * _tmp39;
    _rhs(7, 0) = _tmp199 * _tmp25 + _tmp201 * _tmp32 + _tmp202 * _tmp39;
    _rhs(8, 0) = _tmp203 * _tmp25 + _tmp204 * _tmp32 + _tmp206 * _tmp39;
    _rhs(9, 0) = -_tmp271 - _tmp273 + _tmp275;
    _rhs(10, 0) = _tmp276 - _tmp277 + _tmp278;
    _rhs(11, 0) = -_tmp279 + _tmp280 + _tmp281;
  }
}  // NOLINT(readability/fn_size)

// NOLINTNEXTLINE(readability/fn_size)
}  // namespace sym
