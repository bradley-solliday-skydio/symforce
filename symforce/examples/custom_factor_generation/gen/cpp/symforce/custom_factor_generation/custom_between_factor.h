// -----------------------------------------------------------------------------
// This file was autogenerated by symforce from template:
//     cpp_templates/function/FUNCTION.h.jinja
// Do NOT modify by hand.
// -----------------------------------------------------------------------------

#pragma once

#include <Eigen/Dense>

#include <sym/pose3.h>

namespace custom_factor_generation {

/**
 * Return the 6dof residual on the relative pose between the given two views. Compares
 * the relative pose between the optimized poses to the relative pose between the priors.
 *
 * This is similar to geo_factors_codegen.between_factor, but it uses a weight and diagonal
 * covariance instead of a sqrt information matrix
 *
 * Args:
 *     nav_T_src: Current pose of the src frame
 *     nav_T_target: Current pose of the target frame
 *     target_T_src_prior: Prior on the pose of src in the target frame
 *     prior_weight: The weight of the Gaussian prior
 *     prior_sigmas: The diagonal of the sqrt covariance matrix
 *     epsilon: Small positive value
 *
 * Outputs:
 *     res: The residual of the between factor
 *     jacobian: (6x12) jacobian of res wrt args nav_T_src (6), nav_T_target (6)
 *     hessian: (12x12) Gauss-Newton hessian for args nav_T_src (6), nav_T_target (6)
 *     rhs: (12x1) Gauss-Newton rhs for args nav_T_src (6), nav_T_target (6)
 */
template <typename Scalar>
void CustomBetweenFactor(const sym::Pose3<Scalar>& nav_T_src,
                         const sym::Pose3<Scalar>& nav_T_target,
                         const sym::Pose3<Scalar>& target_T_src_prior, const Scalar prior_weight,
                         const Eigen::Matrix<Scalar, 6, 1>& prior_sigmas, const Scalar epsilon,
                         Eigen::Matrix<Scalar, 6, 1>* const res = nullptr,
                         Eigen::Matrix<Scalar, 6, 12>* const jacobian = nullptr,
                         Eigen::Matrix<Scalar, 12, 12>* const hessian = nullptr,
                         Eigen::Matrix<Scalar, 12, 1>* const rhs = nullptr) {
  // Total ops: 944

  // Input arrays
  const Eigen::Matrix<Scalar, 7, 1>& _nav_T_src = nav_T_src.Data();
  const Eigen::Matrix<Scalar, 7, 1>& _nav_T_target = nav_T_target.Data();
  const Eigen::Matrix<Scalar, 7, 1>& _target_T_src_prior = target_T_src_prior.Data();

  // Intermediate terms (263)
  const Scalar _tmp0 = _nav_T_src[3] * _nav_T_target[2];
  const Scalar _tmp1 = _nav_T_src[2] * _nav_T_target[3];
  const Scalar _tmp2 = _nav_T_src[1] * _nav_T_target[0];
  const Scalar _tmp3 = _nav_T_src[0] * _nav_T_target[1];
  const Scalar _tmp4 = -_tmp0 + _tmp1 - _tmp2 + _tmp3;
  const Scalar _tmp5 = _nav_T_src[0] * _nav_T_target[2];
  const Scalar _tmp6 = _nav_T_src[1] * _nav_T_target[3];
  const Scalar _tmp7 = _nav_T_src[2] * _nav_T_target[0];
  const Scalar _tmp8 = _nav_T_src[3] * _nav_T_target[1];
  const Scalar _tmp9 = -_tmp5 + _tmp6 + _tmp7 - _tmp8;
  const Scalar _tmp10 = _nav_T_src[1] * _nav_T_target[2];
  const Scalar _tmp11 = _nav_T_src[0] * _nav_T_target[3];
  const Scalar _tmp12 = _nav_T_src[3] * _nav_T_target[0];
  const Scalar _tmp13 = _nav_T_src[2] * _nav_T_target[1];
  const Scalar _tmp14 = _tmp10 + _tmp11 - _tmp12 - _tmp13;
  const Scalar _tmp15 = _nav_T_src[2] * _nav_T_target[2];
  const Scalar _tmp16 = _nav_T_src[0] * _nav_T_target[0];
  const Scalar _tmp17 = _nav_T_src[1] * _nav_T_target[1];
  const Scalar _tmp18 = _nav_T_src[3] * _nav_T_target[3];
  const Scalar _tmp19 = _tmp15 + _tmp16 + _tmp17 + _tmp18;
  const Scalar _tmp20 = -_target_T_src_prior[0] * _tmp19 - _target_T_src_prior[1] * _tmp4 +
                        _target_T_src_prior[2] * _tmp9 + _target_T_src_prior[3] * _tmp14;
  const Scalar _tmp21 = Scalar(1.0) / (epsilon + prior_sigmas(0, 0));
  const Scalar _tmp22 = std::sqrt(prior_weight);
  const Scalar _tmp23 = _target_T_src_prior[2] * _tmp4;
  const Scalar _tmp24 = _target_T_src_prior[1] * _tmp9;
  const Scalar _tmp25 = _target_T_src_prior[0] * _tmp14;
  const Scalar _tmp26 = -_tmp23 - _tmp24 - _tmp25;
  const Scalar _tmp27 = _target_T_src_prior[3] * _tmp19;
  const Scalar _tmp28 =
      _tmp22 * (2 * std::min<Scalar>(0, (((-_tmp26 + _tmp27) > 0) - ((-_tmp26 + _tmp27) < 0))) + 1);
  const Scalar _tmp29 = 2 * _tmp28;
  const Scalar _tmp30 = _tmp21 * _tmp29;
  const Scalar _tmp31 = 1 - epsilon;
  const Scalar _tmp32 = std::min<Scalar>(_tmp31, std::fabs(_tmp26 - _tmp27));
  const Scalar _tmp33 = std::acos(_tmp32) / std::sqrt(Scalar(1 - std::pow(_tmp32, Scalar(2))));
  const Scalar _tmp34 = _tmp20 * _tmp30 * _tmp33;
  const Scalar _tmp35 = _tmp29 * _tmp33;
  const Scalar _tmp36 = Scalar(1.0) / (epsilon + prior_sigmas(1, 0));
  const Scalar _tmp37 = _tmp36 * (_target_T_src_prior[0] * _tmp4 - _target_T_src_prior[1] * _tmp19 -
                                  _target_T_src_prior[2] * _tmp14 + _target_T_src_prior[3] * _tmp9);
  const Scalar _tmp38 = _tmp35 * _tmp37;
  const Scalar _tmp39 = Scalar(1.0) / (epsilon + prior_sigmas(2, 0));
  const Scalar _tmp40 =
      _tmp39 * (-_target_T_src_prior[0] * _tmp9 + _target_T_src_prior[1] * _tmp14 -
                _target_T_src_prior[2] * _tmp19 + _target_T_src_prior[3] * _tmp4);
  const Scalar _tmp41 = _tmp35 * _tmp40;
  const Scalar _tmp42 = std::pow(_nav_T_target[1], Scalar(2));
  const Scalar _tmp43 = 2 * _tmp42;
  const Scalar _tmp44 = -_tmp43;
  const Scalar _tmp45 = std::pow(_nav_T_target[2], Scalar(2));
  const Scalar _tmp46 = 2 * _tmp45;
  const Scalar _tmp47 = -_tmp46;
  const Scalar _tmp48 = _tmp44 + _tmp47 + 1;
  const Scalar _tmp49 = 2 * _nav_T_target[0] * _nav_T_target[1];
  const Scalar _tmp50 = 2 * _nav_T_target[2];
  const Scalar _tmp51 = _nav_T_target[3] * _tmp50;
  const Scalar _tmp52 = _tmp49 + _tmp51;
  const Scalar _tmp53 = _nav_T_target[5] * _tmp52;
  const Scalar _tmp54 = _nav_T_target[0] * _tmp50;
  const Scalar _tmp55 = 2 * _nav_T_target[3];
  const Scalar _tmp56 = _nav_T_target[1] * _tmp55;
  const Scalar _tmp57 = -_tmp56;
  const Scalar _tmp58 = _tmp54 + _tmp57;
  const Scalar _tmp59 = _nav_T_target[6] * _tmp58;
  const Scalar _tmp60 = _nav_T_src[5] * _tmp52 + _nav_T_src[6] * _tmp58;
  const Scalar _tmp61 = _nav_T_src[4] * _tmp48 - _nav_T_target[4] * _tmp48 -
                        _target_T_src_prior[4] - _tmp53 - _tmp59 + _tmp60;
  const Scalar _tmp62 = epsilon + prior_sigmas(3, 0);
  const Scalar _tmp63 = _tmp22 / _tmp62;
  const Scalar _tmp64 = std::pow(_nav_T_target[0], Scalar(2));
  const Scalar _tmp65 = 2 * _tmp64;
  const Scalar _tmp66 = 1 - _tmp65;
  const Scalar _tmp67 = _tmp47 + _tmp66;
  const Scalar _tmp68 = _nav_T_target[1] * _tmp50;
  const Scalar _tmp69 = _nav_T_target[0] * _tmp55;
  const Scalar _tmp70 = _tmp68 + _tmp69;
  const Scalar _tmp71 = _nav_T_target[6] * _tmp70;
  const Scalar _tmp72 = -_tmp51;
  const Scalar _tmp73 = _tmp49 + _tmp72;
  const Scalar _tmp74 = _nav_T_target[4] * _tmp73;
  const Scalar _tmp75 = _nav_T_src[4] * _tmp73 + _nav_T_src[6] * _tmp70;
  const Scalar _tmp76 = _nav_T_src[5] * _tmp67 - _nav_T_target[5] * _tmp67 -
                        _target_T_src_prior[5] - _tmp71 - _tmp74 + _tmp75;
  const Scalar _tmp77 = epsilon + prior_sigmas(4, 0);
  const Scalar _tmp78 = _tmp22 / _tmp77;
  const Scalar _tmp79 = _tmp44 + _tmp66;
  const Scalar _tmp80 = -_tmp69;
  const Scalar _tmp81 = _tmp68 + _tmp80;
  const Scalar _tmp82 = _nav_T_target[5] * _tmp81;
  const Scalar _tmp83 = _tmp54 + _tmp56;
  const Scalar _tmp84 = _nav_T_target[4] * _tmp83;
  const Scalar _tmp85 = _nav_T_src[4] * _tmp83 + _nav_T_src[5] * _tmp81;
  const Scalar _tmp86 = _nav_T_src[6] * _tmp79 - _nav_T_target[6] * _tmp79 -
                        _target_T_src_prior[6] - _tmp82 - _tmp84 + _tmp85;
  const Scalar _tmp87 = epsilon + prior_sigmas(5, 0);
  const Scalar _tmp88 = _tmp22 / _tmp87;
  const Scalar _tmp89 = (Scalar(1) / Scalar(2)) * _tmp15;
  const Scalar _tmp90 = (Scalar(1) / Scalar(2)) * _tmp18;
  const Scalar _tmp91 = (Scalar(1) / Scalar(2)) * _tmp16;
  const Scalar _tmp92 = (Scalar(1) / Scalar(2)) * _tmp17;
  const Scalar _tmp93 = _tmp89 + _tmp90 + _tmp91 + _tmp92;
  const Scalar _tmp94 = _target_T_src_prior[0] * _tmp93;
  const Scalar _tmp95 = (Scalar(1) / Scalar(2)) * _tmp10;
  const Scalar _tmp96 = (Scalar(1) / Scalar(2)) * _tmp11;
  const Scalar _tmp97 = (Scalar(1) / Scalar(2)) * _tmp12;
  const Scalar _tmp98 = (Scalar(1) / Scalar(2)) * _tmp13;
  const Scalar _tmp99 = -_tmp95 - _tmp96 + _tmp97 + _tmp98;
  const Scalar _tmp100 = _target_T_src_prior[3] * _tmp99;
  const Scalar _tmp101 = (Scalar(1) / Scalar(2)) * _tmp0;
  const Scalar _tmp102 = (Scalar(1) / Scalar(2)) * _tmp1;
  const Scalar _tmp103 = (Scalar(1) / Scalar(2)) * _tmp2;
  const Scalar _tmp104 = (Scalar(1) / Scalar(2)) * _tmp3;
  const Scalar _tmp105 = -_tmp101 + _tmp102 - _tmp103 + _tmp104;
  const Scalar _tmp106 = _target_T_src_prior[1] * _tmp105;
  const Scalar _tmp107 = (Scalar(1) / Scalar(2)) * _tmp5;
  const Scalar _tmp108 = (Scalar(1) / Scalar(2)) * _tmp6;
  const Scalar _tmp109 = (Scalar(1) / Scalar(2)) * _tmp7;
  const Scalar _tmp110 = (Scalar(1) / Scalar(2)) * _tmp8;
  const Scalar _tmp111 = _tmp107 - _tmp108 - _tmp109 + _tmp110;
  const Scalar _tmp112 = _target_T_src_prior[2] * _tmp111;
  const Scalar _tmp113 = _tmp106 + _tmp112;
  const Scalar _tmp114 = _tmp100 + _tmp113 + _tmp94;
  const Scalar _tmp115 = _tmp20 * _tmp21;
  const Scalar _tmp116 = _tmp23 + _tmp24 + _tmp25 + _tmp27;
  const Scalar _tmp117 = std::fabs(_tmp116);
  const Scalar _tmp118 = std::min<Scalar>(_tmp117, _tmp31);
  const Scalar _tmp119 = std::acos(_tmp118);
  const Scalar _tmp120 = 1 - std::pow(_tmp118, Scalar(2));
  const Scalar _tmp121 = _tmp28 * ((((-_tmp117 + _tmp31) > 0) - ((-_tmp117 + _tmp31) < 0)) + 1) *
                         (((_tmp116) > 0) - ((_tmp116) < 0));
  const Scalar _tmp122 = _tmp118 * _tmp119 * _tmp121 / (_tmp120 * std::sqrt(_tmp120));
  const Scalar _tmp123 = _tmp115 * _tmp122;
  const Scalar _tmp124 = _tmp121 / _tmp120;
  const Scalar _tmp125 = _tmp114 * _tmp124;
  const Scalar _tmp126 = _target_T_src_prior[3] * _tmp93;
  const Scalar _tmp127 = -_target_T_src_prior[0] * _tmp99;
  const Scalar _tmp128 = -_target_T_src_prior[1] * _tmp111;
  const Scalar _tmp129 = _target_T_src_prior[2] * _tmp105;
  const Scalar _tmp130 = _tmp128 + _tmp129;
  const Scalar _tmp131 = _tmp119 / std::sqrt(_tmp120);
  const Scalar _tmp132 = _tmp131 * _tmp30;
  const Scalar _tmp133 =
      _tmp114 * _tmp123 - _tmp115 * _tmp125 + _tmp132 * (_tmp126 + _tmp127 + _tmp130);
  const Scalar _tmp134 = _tmp95 + _tmp96 - _tmp97 - _tmp98;
  const Scalar _tmp135 = -_target_T_src_prior[1] * _tmp134;
  const Scalar _tmp136 = _target_T_src_prior[0] * _tmp111;
  const Scalar _tmp137 = _target_T_src_prior[2] * _tmp93;
  const Scalar _tmp138 = _tmp101 - _tmp102 + _tmp103 - _tmp104;
  const Scalar _tmp139 = _target_T_src_prior[3] * _tmp138;
  const Scalar _tmp140 = _tmp137 + _tmp139;
  const Scalar _tmp141 = _target_T_src_prior[3] * _tmp111;
  const Scalar _tmp142 = _target_T_src_prior[1] * _tmp93;
  const Scalar _tmp143 = _target_T_src_prior[2] * _tmp134;
  const Scalar _tmp144 = _target_T_src_prior[0] * _tmp138;
  const Scalar _tmp145 = _tmp143 + _tmp144;
  const Scalar _tmp146 = _tmp141 + _tmp142 + _tmp145;
  const Scalar _tmp147 = _tmp124 * _tmp146;
  const Scalar _tmp148 =
      -_tmp115 * _tmp147 + _tmp123 * _tmp146 + _tmp132 * (_tmp135 - _tmp136 + _tmp140);
  const Scalar _tmp149 = _target_T_src_prior[1] * _tmp99;
  const Scalar _tmp150 = -_tmp107 + _tmp108 + _tmp109 - _tmp110;
  const Scalar _tmp151 = _target_T_src_prior[0] * _tmp150;
  const Scalar _tmp152 = _tmp149 + _tmp151;
  const Scalar _tmp153 = _tmp140 + _tmp152;
  const Scalar _tmp154 = _target_T_src_prior[2] * _tmp99;
  const Scalar _tmp155 = _target_T_src_prior[3] * _tmp150;
  const Scalar _tmp156 = _tmp154 + _tmp155;
  const Scalar _tmp157 = _tmp124 * _tmp153;
  const Scalar _tmp158 =
      -_tmp115 * _tmp157 + _tmp123 * _tmp153 + _tmp132 * (-_tmp142 - _tmp144 + _tmp156);
  const Scalar _tmp159 = _target_T_src_prior[3] * _tmp134;
  const Scalar _tmp160 = -_tmp89 - _tmp90 - _tmp91 - _tmp92;
  const Scalar _tmp161 = _target_T_src_prior[0] * _tmp160;
  const Scalar _tmp162 = _tmp113 + _tmp159 + _tmp161;
  const Scalar _tmp163 = _target_T_src_prior[3] * _tmp160;
  const Scalar _tmp164 = _target_T_src_prior[0] * _tmp134;
  const Scalar _tmp165 = _tmp124 * _tmp162;
  const Scalar _tmp166 =
      -_tmp115 * _tmp165 + _tmp123 * _tmp162 + _tmp132 * (_tmp130 + _tmp163 - _tmp164);
  const Scalar _tmp167 = _target_T_src_prior[1] * _tmp160;
  const Scalar _tmp168 = _tmp145 + _tmp155 + _tmp167;
  const Scalar _tmp169 = _tmp124 * _tmp168;
  const Scalar _tmp170 = _target_T_src_prior[2] * _tmp160;
  const Scalar _tmp171 =
      -_tmp115 * _tmp169 + _tmp123 * _tmp168 + _tmp132 * (_tmp135 + _tmp139 - _tmp151 + _tmp170);
  const Scalar _tmp172 = _target_T_src_prior[3] * _tmp105;
  const Scalar _tmp173 = _tmp152 + _tmp170 + _tmp172;
  const Scalar _tmp174 = -_target_T_src_prior[0] * _tmp105;
  const Scalar _tmp175 = _tmp124 * _tmp173;
  const Scalar _tmp176 =
      -_tmp115 * _tmp175 + _tmp123 * _tmp173 + _tmp132 * (_tmp156 - _tmp167 + _tmp174);
  const Scalar _tmp177 = _tmp136 + _tmp172;
  const Scalar _tmp178 = _tmp131 * _tmp29;
  const Scalar _tmp179 = _tmp178 * _tmp36;
  const Scalar _tmp180 = _tmp122 * _tmp37;
  const Scalar _tmp181 =
      _tmp114 * _tmp180 - _tmp125 * _tmp37 + _tmp179 * (-_tmp137 - _tmp149 + _tmp177);
  const Scalar _tmp182 = _tmp124 * _tmp37;
  const Scalar _tmp183 = -_target_T_src_prior[2] * _tmp138;
  const Scalar _tmp184 = _tmp126 + _tmp183;
  const Scalar _tmp185 =
      _tmp146 * _tmp180 - _tmp146 * _tmp182 + _tmp179 * (_tmp128 + _tmp164 + _tmp184);
  const Scalar _tmp186 = _target_T_src_prior[1] * _tmp138;
  const Scalar _tmp187 = -_target_T_src_prior[2] * _tmp150;
  const Scalar _tmp188 = _tmp100 + _tmp187;
  const Scalar _tmp189 =
      _tmp153 * _tmp180 - _tmp157 * _tmp37 + _tmp179 * (-_tmp186 + _tmp188 + _tmp94);
  const Scalar _tmp190 =
      _tmp162 * _tmp180 - _tmp165 * _tmp37 + _tmp179 * (_tmp135 - _tmp170 + _tmp177);
  const Scalar _tmp191 = _target_T_src_prior[1] * _tmp150;
  const Scalar _tmp192 =
      _tmp168 * _tmp180 - _tmp169 * _tmp37 + _tmp179 * (_tmp163 + _tmp164 + _tmp183 - _tmp191);
  const Scalar _tmp193 =
      _tmp173 * _tmp180 - _tmp173 * _tmp182 + _tmp179 * (-_tmp106 + _tmp161 + _tmp188);
  const Scalar _tmp194 = _tmp122 * _tmp40;
  const Scalar _tmp195 = _tmp141 + _tmp174;
  const Scalar _tmp196 = _tmp178 * _tmp39;
  const Scalar _tmp197 =
      _tmp114 * _tmp194 - _tmp125 * _tmp40 + _tmp196 * (_tmp142 - _tmp154 + _tmp195);
  const Scalar _tmp198 = _tmp159 + _tmp186;
  const Scalar _tmp199 =
      _tmp146 * _tmp194 - _tmp147 * _tmp40 + _tmp196 * (-_tmp112 + _tmp198 - _tmp94);
  const Scalar _tmp200 = _tmp127 + _tmp191;
  const Scalar _tmp201 = _tmp153 * _tmp194 - _tmp157 * _tmp40 + _tmp196 * (_tmp184 + _tmp200);
  const Scalar _tmp202 =
      _tmp162 * _tmp194 - _tmp165 * _tmp40 + _tmp196 * (-_tmp143 + _tmp167 + _tmp195);
  const Scalar _tmp203 =
      _tmp168 * _tmp194 - _tmp169 * _tmp40 + _tmp196 * (-_tmp161 + _tmp187 + _tmp198);
  const Scalar _tmp204 =
      _tmp173 * _tmp194 - _tmp175 * _tmp40 + _tmp196 * (-_tmp129 + _tmp163 + _tmp200);
  const Scalar _tmp205 = std::pow(_nav_T_target[3], Scalar(2));
  const Scalar _tmp206 = -_tmp205;
  const Scalar _tmp207 = -_tmp45;
  const Scalar _tmp208 = _tmp206 + _tmp207 + _tmp42 + _tmp64;
  const Scalar _tmp209 = -_tmp68;
  const Scalar _tmp210 = _tmp209 + _tmp69;
  const Scalar _tmp211 = -_tmp54;
  const Scalar _tmp212 = _tmp211 + _tmp57;
  const Scalar _tmp213 = _nav_T_src[4] * _tmp212 + _nav_T_src[5] * _tmp210 +
                         _nav_T_src[6] * _tmp208 - _nav_T_target[4] * _tmp212 -
                         _nav_T_target[5] * _tmp210 - _nav_T_target[6] * _tmp208;
  const Scalar _tmp214 = _tmp205 + _tmp207;
  const Scalar _tmp215 = -_tmp64;
  const Scalar _tmp216 = _tmp215 + _tmp42;
  const Scalar _tmp217 = _tmp214 + _tmp216;
  const Scalar _tmp218 =
      _nav_T_src[5] * _tmp217 - _nav_T_target[5] * _tmp217 - _tmp71 - _tmp74 + _tmp75;
  const Scalar _tmp219 = _tmp43 + _tmp46 - 1;
  const Scalar _tmp220 = -_tmp49;
  const Scalar _tmp221 = _tmp220 + _tmp72;
  const Scalar _tmp222 = _tmp211 + _tmp56;
  const Scalar _tmp223 = -_tmp42;
  const Scalar _tmp224 = _tmp205 + _tmp215 + _tmp223 + _tmp45;
  const Scalar _tmp225 =
      _nav_T_src[6] * _tmp224 - _nav_T_target[6] * _tmp224 - _tmp82 - _tmp84 + _tmp85;
  const Scalar _tmp226 = _tmp206 + _tmp45;
  const Scalar _tmp227 = _tmp216 + _tmp226;
  const Scalar _tmp228 = _nav_T_src[4] * _tmp227 + _nav_T_src[5] * _tmp221 +
                         _nav_T_src[6] * _tmp222 - _nav_T_target[4] * _tmp227 -
                         _nav_T_target[5] * _tmp221 - _nav_T_target[6] * _tmp222;
  const Scalar _tmp229 = _tmp220 + _tmp51;
  const Scalar _tmp230 = _tmp65 - 1;
  const Scalar _tmp231 = _tmp230 + _tmp46;
  const Scalar _tmp232 = _tmp209 + _tmp80;
  const Scalar _tmp233 = _tmp223 + _tmp64;
  const Scalar _tmp234 = _tmp226 + _tmp233;
  const Scalar _tmp235 = _nav_T_src[4] * _tmp229 + _nav_T_src[5] * _tmp234 +
                         _nav_T_src[6] * _tmp232 - _nav_T_target[4] * _tmp229 -
                         _nav_T_target[5] * _tmp234 - _nav_T_target[6] * _tmp232;
  const Scalar _tmp236 = _tmp214 + _tmp233;
  const Scalar _tmp237 =
      _nav_T_src[4] * _tmp236 - _nav_T_target[4] * _tmp236 - _tmp53 - _tmp59 + _tmp60;
  const Scalar _tmp238 = _tmp230 + _tmp43;
  const Scalar _tmp239 = prior_weight / std::pow(_tmp87, Scalar(2));
  const Scalar _tmp240 = prior_weight / std::pow(_tmp77, Scalar(2));
  const Scalar _tmp241 = prior_weight / std::pow(_tmp62, Scalar(2));
  const Scalar _tmp242 = _tmp239 * _tmp81;
  const Scalar _tmp243 = _tmp241 * _tmp52;
  const Scalar _tmp244 = _tmp240 * _tmp67;
  const Scalar _tmp245 = _tmp239 * _tmp79;
  const Scalar _tmp246 = _tmp241 * _tmp58;
  const Scalar _tmp247 = _tmp225 * _tmp240;
  const Scalar _tmp248 = _tmp235 * _tmp239;
  const Scalar _tmp249 = _tmp237 * _tmp239;
  const Scalar _tmp250 = _tmp213 * _tmp241;
  const Scalar _tmp251 = _tmp228 * _tmp240;
  const Scalar _tmp252 = _tmp218 * _tmp241;
  const Scalar _tmp253 = _tmp229 * _tmp240;
  const Scalar _tmp254 = _tmp219 * _tmp241;
  const Scalar _tmp255 = _tmp210 * _tmp239;
  const Scalar _tmp256 = _tmp221 * _tmp241;
  const Scalar _tmp257 = _tmp231 * _tmp240;
  const Scalar _tmp258 = _tmp238 * _tmp239;
  const Scalar _tmp259 = _tmp232 * _tmp240;
  const Scalar _tmp260 = _tmp239 * _tmp86;
  const Scalar _tmp261 = _tmp240 * _tmp76;
  const Scalar _tmp262 = _tmp241 * _tmp61;

  // Output terms (4)
  if (res != nullptr) {
    Eigen::Matrix<Scalar, 6, 1>& _res = (*res);

    _res(0, 0) = _tmp34;
    _res(1, 0) = _tmp38;
    _res(2, 0) = _tmp41;
    _res(3, 0) = _tmp61 * _tmp63;
    _res(4, 0) = _tmp76 * _tmp78;
    _res(5, 0) = _tmp86 * _tmp88;
  }

  if (jacobian != nullptr) {
    Eigen::Matrix<Scalar, 6, 12>& _jacobian = (*jacobian);

    _jacobian(0, 0) = _tmp133;
    _jacobian(0, 1) = _tmp148;
    _jacobian(0, 2) = _tmp158;
    _jacobian(0, 3) = 0;
    _jacobian(0, 4) = 0;
    _jacobian(0, 5) = 0;
    _jacobian(0, 6) = _tmp166;
    _jacobian(0, 7) = _tmp171;
    _jacobian(0, 8) = _tmp176;
    _jacobian(0, 9) = 0;
    _jacobian(0, 10) = 0;
    _jacobian(0, 11) = 0;
    _jacobian(1, 0) = _tmp181;
    _jacobian(1, 1) = _tmp185;
    _jacobian(1, 2) = _tmp189;
    _jacobian(1, 3) = 0;
    _jacobian(1, 4) = 0;
    _jacobian(1, 5) = 0;
    _jacobian(1, 6) = _tmp190;
    _jacobian(1, 7) = _tmp192;
    _jacobian(1, 8) = _tmp193;
    _jacobian(1, 9) = 0;
    _jacobian(1, 10) = 0;
    _jacobian(1, 11) = 0;
    _jacobian(2, 0) = _tmp197;
    _jacobian(2, 1) = _tmp199;
    _jacobian(2, 2) = _tmp201;
    _jacobian(2, 3) = 0;
    _jacobian(2, 4) = 0;
    _jacobian(2, 5) = 0;
    _jacobian(2, 6) = _tmp202;
    _jacobian(2, 7) = _tmp203;
    _jacobian(2, 8) = _tmp204;
    _jacobian(2, 9) = 0;
    _jacobian(2, 10) = 0;
    _jacobian(2, 11) = 0;
    _jacobian(3, 0) = 0;
    _jacobian(3, 1) = 0;
    _jacobian(3, 2) = 0;
    _jacobian(3, 3) = _tmp48 * _tmp63;
    _jacobian(3, 4) = _tmp52 * _tmp63;
    _jacobian(3, 5) = _tmp58 * _tmp63;
    _jacobian(3, 6) = 0;
    _jacobian(3, 7) = _tmp213 * _tmp63;
    _jacobian(3, 8) = _tmp218 * _tmp63;
    _jacobian(3, 9) = _tmp219 * _tmp63;
    _jacobian(3, 10) = _tmp221 * _tmp63;
    _jacobian(3, 11) = _tmp222 * _tmp63;
    _jacobian(4, 0) = 0;
    _jacobian(4, 1) = 0;
    _jacobian(4, 2) = 0;
    _jacobian(4, 3) = _tmp73 * _tmp78;
    _jacobian(4, 4) = _tmp67 * _tmp78;
    _jacobian(4, 5) = _tmp70 * _tmp78;
    _jacobian(4, 6) = _tmp225 * _tmp78;
    _jacobian(4, 7) = 0;
    _jacobian(4, 8) = _tmp228 * _tmp78;
    _jacobian(4, 9) = _tmp229 * _tmp78;
    _jacobian(4, 10) = _tmp231 * _tmp78;
    _jacobian(4, 11) = _tmp232 * _tmp78;
    _jacobian(5, 0) = 0;
    _jacobian(5, 1) = 0;
    _jacobian(5, 2) = 0;
    _jacobian(5, 3) = _tmp83 * _tmp88;
    _jacobian(5, 4) = _tmp81 * _tmp88;
    _jacobian(5, 5) = _tmp79 * _tmp88;
    _jacobian(5, 6) = _tmp235 * _tmp88;
    _jacobian(5, 7) = _tmp237 * _tmp88;
    _jacobian(5, 8) = 0;
    _jacobian(5, 9) = _tmp212 * _tmp88;
    _jacobian(5, 10) = _tmp210 * _tmp88;
    _jacobian(5, 11) = _tmp238 * _tmp88;
  }

  if (hessian != nullptr) {
    Eigen::Matrix<Scalar, 12, 12>& _hessian = (*hessian);

    _hessian(0, 0) =
        std::pow(_tmp133, Scalar(2)) + std::pow(_tmp181, Scalar(2)) + std::pow(_tmp197, Scalar(2));
    _hessian(0, 1) = 0;
    _hessian(0, 2) = 0;
    _hessian(0, 3) = 0;
    _hessian(0, 4) = 0;
    _hessian(0, 5) = 0;
    _hessian(0, 6) = 0;
    _hessian(0, 7) = 0;
    _hessian(0, 8) = 0;
    _hessian(0, 9) = 0;
    _hessian(0, 10) = 0;
    _hessian(0, 11) = 0;
    _hessian(1, 0) = _tmp133 * _tmp148 + _tmp181 * _tmp185 + _tmp197 * _tmp199;
    _hessian(1, 1) =
        std::pow(_tmp148, Scalar(2)) + std::pow(_tmp185, Scalar(2)) + std::pow(_tmp199, Scalar(2));
    _hessian(1, 2) = 0;
    _hessian(1, 3) = 0;
    _hessian(1, 4) = 0;
    _hessian(1, 5) = 0;
    _hessian(1, 6) = 0;
    _hessian(1, 7) = 0;
    _hessian(1, 8) = 0;
    _hessian(1, 9) = 0;
    _hessian(1, 10) = 0;
    _hessian(1, 11) = 0;
    _hessian(2, 0) = _tmp133 * _tmp158 + _tmp181 * _tmp189 + _tmp197 * _tmp201;
    _hessian(2, 1) = _tmp148 * _tmp158 + _tmp185 * _tmp189 + _tmp199 * _tmp201;
    _hessian(2, 2) =
        std::pow(_tmp158, Scalar(2)) + std::pow(_tmp189, Scalar(2)) + std::pow(_tmp201, Scalar(2));
    _hessian(2, 3) = 0;
    _hessian(2, 4) = 0;
    _hessian(2, 5) = 0;
    _hessian(2, 6) = 0;
    _hessian(2, 7) = 0;
    _hessian(2, 8) = 0;
    _hessian(2, 9) = 0;
    _hessian(2, 10) = 0;
    _hessian(2, 11) = 0;
    _hessian(3, 0) = 0;
    _hessian(3, 1) = 0;
    _hessian(3, 2) = 0;
    _hessian(3, 3) = _tmp239 * std::pow(_tmp83, Scalar(2)) + _tmp240 * std::pow(_tmp73, Scalar(2)) +
                     _tmp241 * std::pow(_tmp48, Scalar(2));
    _hessian(3, 4) = 0;
    _hessian(3, 5) = 0;
    _hessian(3, 6) = 0;
    _hessian(3, 7) = 0;
    _hessian(3, 8) = 0;
    _hessian(3, 9) = 0;
    _hessian(3, 10) = 0;
    _hessian(3, 11) = 0;
    _hessian(4, 0) = 0;
    _hessian(4, 1) = 0;
    _hessian(4, 2) = 0;
    _hessian(4, 3) = _tmp242 * _tmp83 + _tmp243 * _tmp48 + _tmp244 * _tmp73;
    _hessian(4, 4) = _tmp239 * std::pow(_tmp81, Scalar(2)) + _tmp240 * std::pow(_tmp67, Scalar(2)) +
                     _tmp241 * std::pow(_tmp52, Scalar(2));
    _hessian(4, 5) = 0;
    _hessian(4, 6) = 0;
    _hessian(4, 7) = 0;
    _hessian(4, 8) = 0;
    _hessian(4, 9) = 0;
    _hessian(4, 10) = 0;
    _hessian(4, 11) = 0;
    _hessian(5, 0) = 0;
    _hessian(5, 1) = 0;
    _hessian(5, 2) = 0;
    _hessian(5, 3) = _tmp240 * _tmp70 * _tmp73 + _tmp245 * _tmp83 + _tmp246 * _tmp48;
    _hessian(5, 4) = _tmp242 * _tmp79 + _tmp243 * _tmp58 + _tmp244 * _tmp70;
    _hessian(5, 5) = _tmp239 * std::pow(_tmp79, Scalar(2)) + _tmp240 * std::pow(_tmp70, Scalar(2)) +
                     _tmp241 * std::pow(_tmp58, Scalar(2));
    _hessian(5, 6) = 0;
    _hessian(5, 7) = 0;
    _hessian(5, 8) = 0;
    _hessian(5, 9) = 0;
    _hessian(5, 10) = 0;
    _hessian(5, 11) = 0;
    _hessian(6, 0) = _tmp133 * _tmp166 + _tmp181 * _tmp190 + _tmp197 * _tmp202;
    _hessian(6, 1) = _tmp148 * _tmp166 + _tmp185 * _tmp190 + _tmp199 * _tmp202;
    _hessian(6, 2) = _tmp158 * _tmp166 + _tmp189 * _tmp190 + _tmp201 * _tmp202;
    _hessian(6, 3) = _tmp247 * _tmp73 + _tmp248 * _tmp83;
    _hessian(6, 4) = _tmp225 * _tmp244 + _tmp235 * _tmp242;
    _hessian(6, 5) = _tmp247 * _tmp70 + _tmp248 * _tmp79;
    _hessian(6, 6) = std::pow(_tmp166, Scalar(2)) + std::pow(_tmp190, Scalar(2)) +
                     std::pow(_tmp202, Scalar(2)) + std::pow(_tmp225, Scalar(2)) * _tmp240 +
                     std::pow(_tmp235, Scalar(2)) * _tmp239;
    _hessian(6, 7) = 0;
    _hessian(6, 8) = 0;
    _hessian(6, 9) = 0;
    _hessian(6, 10) = 0;
    _hessian(6, 11) = 0;
    _hessian(7, 0) = _tmp133 * _tmp171 + _tmp181 * _tmp192 + _tmp197 * _tmp203;
    _hessian(7, 1) = _tmp148 * _tmp171 + _tmp185 * _tmp192 + _tmp199 * _tmp203;
    _hessian(7, 2) = _tmp158 * _tmp171 + _tmp189 * _tmp192 + _tmp201 * _tmp203;
    _hessian(7, 3) = _tmp249 * _tmp83 + _tmp250 * _tmp48;
    _hessian(7, 4) = _tmp237 * _tmp242 + _tmp250 * _tmp52;
    _hessian(7, 5) = _tmp249 * _tmp79 + _tmp250 * _tmp58;
    _hessian(7, 6) = _tmp166 * _tmp171 + _tmp190 * _tmp192 + _tmp202 * _tmp203 + _tmp237 * _tmp248;
    _hessian(7, 7) = std::pow(_tmp171, Scalar(2)) + std::pow(_tmp192, Scalar(2)) +
                     std::pow(_tmp203, Scalar(2)) + std::pow(_tmp213, Scalar(2)) * _tmp241 +
                     std::pow(_tmp237, Scalar(2)) * _tmp239;
    _hessian(7, 8) = 0;
    _hessian(7, 9) = 0;
    _hessian(7, 10) = 0;
    _hessian(7, 11) = 0;
    _hessian(8, 0) = _tmp133 * _tmp176 + _tmp181 * _tmp193 + _tmp197 * _tmp204;
    _hessian(8, 1) = _tmp148 * _tmp176 + _tmp185 * _tmp193 + _tmp199 * _tmp204;
    _hessian(8, 2) = _tmp158 * _tmp176 + _tmp189 * _tmp193 + _tmp201 * _tmp204;
    _hessian(8, 3) = _tmp251 * _tmp73 + _tmp252 * _tmp48;
    _hessian(8, 4) = _tmp228 * _tmp244 + _tmp252 * _tmp52;
    _hessian(8, 5) = _tmp251 * _tmp70 + _tmp252 * _tmp58;
    _hessian(8, 6) = _tmp166 * _tmp176 + _tmp190 * _tmp193 + _tmp202 * _tmp204 + _tmp225 * _tmp251;
    _hessian(8, 7) = _tmp171 * _tmp176 + _tmp192 * _tmp193 + _tmp203 * _tmp204 + _tmp218 * _tmp250;
    _hessian(8, 8) = std::pow(_tmp176, Scalar(2)) + std::pow(_tmp193, Scalar(2)) +
                     std::pow(_tmp204, Scalar(2)) + std::pow(_tmp218, Scalar(2)) * _tmp241 +
                     std::pow(_tmp228, Scalar(2)) * _tmp240;
    _hessian(8, 9) = 0;
    _hessian(8, 10) = 0;
    _hessian(8, 11) = 0;
    _hessian(9, 0) = 0;
    _hessian(9, 1) = 0;
    _hessian(9, 2) = 0;
    _hessian(9, 3) = _tmp212 * _tmp239 * _tmp83 + _tmp253 * _tmp73 + _tmp254 * _tmp48;
    _hessian(9, 4) = _tmp212 * _tmp242 + _tmp229 * _tmp244 + _tmp254 * _tmp52;
    _hessian(9, 5) = _tmp212 * _tmp245 + _tmp253 * _tmp70 + _tmp254 * _tmp58;
    _hessian(9, 6) = _tmp212 * _tmp248 + _tmp225 * _tmp253;
    _hessian(9, 7) = _tmp212 * _tmp249 + _tmp219 * _tmp250;
    _hessian(9, 8) = _tmp218 * _tmp254 + _tmp229 * _tmp251;
    _hessian(9, 9) = std::pow(_tmp212, Scalar(2)) * _tmp239 +
                     std::pow(_tmp219, Scalar(2)) * _tmp241 +
                     std::pow(_tmp229, Scalar(2)) * _tmp240;
    _hessian(9, 10) = 0;
    _hessian(9, 11) = 0;
    _hessian(10, 0) = 0;
    _hessian(10, 1) = 0;
    _hessian(10, 2) = 0;
    _hessian(10, 3) = _tmp255 * _tmp83 + _tmp256 * _tmp48 + _tmp257 * _tmp73;
    _hessian(10, 4) = _tmp255 * _tmp81 + _tmp256 * _tmp52 + _tmp257 * _tmp67;
    _hessian(10, 5) = _tmp255 * _tmp79 + _tmp256 * _tmp58 + _tmp257 * _tmp70;
    _hessian(10, 6) = _tmp225 * _tmp257 + _tmp235 * _tmp255;
    _hessian(10, 7) = _tmp221 * _tmp250 + _tmp237 * _tmp255;
    _hessian(10, 8) = _tmp221 * _tmp252 + _tmp228 * _tmp257;
    _hessian(10, 9) = _tmp212 * _tmp255 + _tmp221 * _tmp254 + _tmp229 * _tmp257;
    _hessian(10, 10) = std::pow(_tmp210, Scalar(2)) * _tmp239 +
                       std::pow(_tmp221, Scalar(2)) * _tmp241 +
                       std::pow(_tmp231, Scalar(2)) * _tmp240;
    _hessian(10, 11) = 0;
    _hessian(11, 0) = 0;
    _hessian(11, 1) = 0;
    _hessian(11, 2) = 0;
    _hessian(11, 3) = _tmp222 * _tmp241 * _tmp48 + _tmp258 * _tmp83 + _tmp259 * _tmp73;
    _hessian(11, 4) = _tmp222 * _tmp243 + _tmp238 * _tmp242 + _tmp259 * _tmp67;
    _hessian(11, 5) = _tmp222 * _tmp246 + _tmp258 * _tmp79 + _tmp259 * _tmp70;
    _hessian(11, 6) = _tmp225 * _tmp259 + _tmp238 * _tmp248;
    _hessian(11, 7) = _tmp222 * _tmp250 + _tmp238 * _tmp249;
    _hessian(11, 8) = _tmp222 * _tmp252 + _tmp228 * _tmp259;
    _hessian(11, 9) = _tmp212 * _tmp258 + _tmp222 * _tmp254 + _tmp229 * _tmp259;
    _hessian(11, 10) = _tmp222 * _tmp256 + _tmp232 * _tmp257 + _tmp238 * _tmp255;
    _hessian(11, 11) = std::pow(_tmp222, Scalar(2)) * _tmp241 +
                       std::pow(_tmp232, Scalar(2)) * _tmp240 +
                       std::pow(_tmp238, Scalar(2)) * _tmp239;
  }

  if (rhs != nullptr) {
    Eigen::Matrix<Scalar, 12, 1>& _rhs = (*rhs);

    _rhs(0, 0) = _tmp133 * _tmp34 + _tmp181 * _tmp38 + _tmp197 * _tmp41;
    _rhs(1, 0) = _tmp148 * _tmp34 + _tmp185 * _tmp38 + _tmp199 * _tmp41;
    _rhs(2, 0) = _tmp158 * _tmp34 + _tmp189 * _tmp38 + _tmp201 * _tmp41;
    _rhs(3, 0) = _tmp260 * _tmp83 + _tmp261 * _tmp73 + _tmp262 * _tmp48;
    _rhs(4, 0) = _tmp242 * _tmp86 + _tmp244 * _tmp76 + _tmp262 * _tmp52;
    _rhs(5, 0) = _tmp260 * _tmp79 + _tmp261 * _tmp70 + _tmp262 * _tmp58;
    _rhs(6, 0) = _tmp166 * _tmp34 + _tmp190 * _tmp38 + _tmp202 * _tmp41 + _tmp225 * _tmp261 +
                 _tmp248 * _tmp86;
    _rhs(7, 0) = _tmp171 * _tmp34 + _tmp192 * _tmp38 + _tmp203 * _tmp41 + _tmp249 * _tmp86 +
                 _tmp250 * _tmp61;
    _rhs(8, 0) = _tmp176 * _tmp34 + _tmp193 * _tmp38 + _tmp204 * _tmp41 + _tmp251 * _tmp76 +
                 _tmp252 * _tmp61;
    _rhs(9, 0) = _tmp212 * _tmp260 + _tmp229 * _tmp261 + _tmp254 * _tmp61;
    _rhs(10, 0) = _tmp221 * _tmp262 + _tmp255 * _tmp86 + _tmp257 * _tmp76;
    _rhs(11, 0) = _tmp222 * _tmp262 + _tmp238 * _tmp260 + _tmp259 * _tmp76;
  }
}  // NOLINT(readability/fn_size)

// NOLINTNEXTLINE(readability/fn_size)
}  // namespace custom_factor_generation
