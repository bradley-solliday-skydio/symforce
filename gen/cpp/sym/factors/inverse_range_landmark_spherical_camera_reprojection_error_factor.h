// -----------------------------------------------------------------------------
// This file was autogenerated by symforce from template:
//     cpp_templates/function/FUNCTION.h.jinja
// Do NOT modify by hand.
// -----------------------------------------------------------------------------

#pragma once

#include <Eigen/Dense>

#include <sym/pose3.h>

namespace sym {

/**
 * Return the 2dof residual of reprojecting the landmark ray into the target spherical camera and
 * comparing it against the correspondence.
 *
 * The landmark is specified as a camera point in the source camera with an inverse range; this
 * means the landmark is fixed in the source camera and always has residual 0 there (this 0 residual
 * is not returned, only the residual in the target camera is returned).
 *
 * The norm of the residual is whitened using the Barron noise model.  Whitening each component of
 * the reprojection error separately would result in rejecting individual components as outliers.
 * Instead, we minimize the whitened norm of the full reprojection error for each point.  See the
 * docstring for `NoiseModel.whiten_norm` for more information on this, and the docstring of
 * `BarronNoiseModel` for more information on the noise model.
 *
 * Args:
 *     source_pose: The pose of the source camera
 *     target_pose: The pose of the target camera
 *     target_calibration_storage: The storage vector of the target spherical camera calibration
 *     source_inverse_range: The inverse range of the landmark in the source camera
 *     p_camera_source: The location of the landmark in the source camera coordinate, will be
 * normalized target_pixel: The location of the correspondence in the target camera weight: The
 * weight of the factor gnc_mu: The mu convexity parameter for the Barron noise model gnc_scale: The
 * scale parameter for the Barron noise model epsilon: Small positive value
 *
 * Outputs:
 *     res: 2dof whiten residual of the reprojection
 *     jacobian: (2x13) jacobian of res wrt args source_pose (6), target_pose (6),
 *               source_inverse_range (1)
 *     hessian: (13x13) Gauss-Newton hessian for args source_pose (6), target_pose (6),
 *              source_inverse_range (1)
 *     rhs: (13x1) Gauss-Newton rhs for args source_pose (6), target_pose (6), source_inverse_range
 *          (1)
 */
template <typename Scalar>
void InverseRangeLandmarkSphericalCameraReprojectionErrorFactor(
    const sym::Pose3<Scalar>& source_pose, const sym::Pose3<Scalar>& target_pose,
    const Eigen::Matrix<Scalar, 9, 1>& target_calibration_storage,
    const Scalar source_inverse_range, const Eigen::Matrix<Scalar, 3, 1>& p_camera_source,
    const Eigen::Matrix<Scalar, 2, 1>& target_pixel, const Scalar weight, const Scalar gnc_mu,
    const Scalar gnc_scale, const Scalar epsilon, Eigen::Matrix<Scalar, 2, 1>* const res = nullptr,
    Eigen::Matrix<Scalar, 2, 13>* const jacobian = nullptr,
    Eigen::Matrix<Scalar, 13, 13>* const hessian = nullptr,
    Eigen::Matrix<Scalar, 13, 1>* const rhs = nullptr) {
  // Total ops: 1288

  // Input arrays
  const Eigen::Matrix<Scalar, 7, 1>& _source_pose = source_pose.Data();
  const Eigen::Matrix<Scalar, 7, 1>& _target_pose = target_pose.Data();

  // Intermediate terms (313)
  const Scalar _tmp0 = std::pow(_target_pose[1], Scalar(2));
  const Scalar _tmp1 = -2 * _tmp0;
  const Scalar _tmp2 = std::pow(_target_pose[2], Scalar(2));
  const Scalar _tmp3 = 1 - 2 * _tmp2;
  const Scalar _tmp4 = _tmp1 + _tmp3;
  const Scalar _tmp5 = 2 * _source_pose[1];
  const Scalar _tmp6 = _source_pose[3] * _tmp5;
  const Scalar _tmp7 = 2 * _source_pose[0] * _source_pose[2];
  const Scalar _tmp8 = _tmp6 + _tmp7;
  const Scalar _tmp9 = std::pow(Scalar(epsilon + std::pow(p_camera_source(0, 0), Scalar(2)) +
                                       std::pow(p_camera_source(1, 0), Scalar(2)) +
                                       std::pow(p_camera_source(2, 0), Scalar(2))),
                                Scalar(Scalar(-1) / Scalar(2)));
  const Scalar _tmp10 = _tmp9 * p_camera_source(2, 0);
  const Scalar _tmp11 = _source_pose[0] * _tmp5;
  const Scalar _tmp12 = 2 * _source_pose[3];
  const Scalar _tmp13 = _source_pose[2] * _tmp12;
  const Scalar _tmp14 = -_tmp13;
  const Scalar _tmp15 = _tmp11 + _tmp14;
  const Scalar _tmp16 = _tmp9 * p_camera_source(1, 0);
  const Scalar _tmp17 = std::pow(_source_pose[1], Scalar(2));
  const Scalar _tmp18 = -2 * _tmp17;
  const Scalar _tmp19 = std::pow(_source_pose[2], Scalar(2));
  const Scalar _tmp20 = 1 - 2 * _tmp19;
  const Scalar _tmp21 = _tmp9 * p_camera_source(0, 0);
  const Scalar _tmp22 = _source_pose[4] - _target_pose[4];
  const Scalar _tmp23 =
      _tmp10 * _tmp8 + _tmp15 * _tmp16 + _tmp21 * (_tmp18 + _tmp20) + _tmp22 * source_inverse_range;
  const Scalar _tmp24 = 2 * _target_pose[3];
  const Scalar _tmp25 = _target_pose[2] * _tmp24;
  const Scalar _tmp26 = 2 * _target_pose[0];
  const Scalar _tmp27 = _target_pose[1] * _tmp26;
  const Scalar _tmp28 = _tmp25 + _tmp27;
  const Scalar _tmp29 = _source_pose[2] * _tmp5;
  const Scalar _tmp30 = _source_pose[0] * _tmp12;
  const Scalar _tmp31 = -_tmp30;
  const Scalar _tmp32 = _tmp29 + _tmp31;
  const Scalar _tmp33 = _tmp11 + _tmp13;
  const Scalar _tmp34 = std::pow(_source_pose[0], Scalar(2));
  const Scalar _tmp35 = -2 * _tmp34;
  const Scalar _tmp36 = _source_pose[5] - _target_pose[5];
  const Scalar _tmp37 = _tmp10 * _tmp32 + _tmp16 * (_tmp20 + _tmp35) + _tmp21 * _tmp33 +
                        _tmp36 * source_inverse_range;
  const Scalar _tmp38 = _target_pose[2] * _tmp26;
  const Scalar _tmp39 = _target_pose[1] * _tmp24;
  const Scalar _tmp40 = -_tmp39;
  const Scalar _tmp41 = _tmp38 + _tmp40;
  const Scalar _tmp42 = _tmp29 + _tmp30;
  const Scalar _tmp43 = -_tmp6;
  const Scalar _tmp44 = _tmp43 + _tmp7;
  const Scalar _tmp45 = _source_pose[6] - _target_pose[6];
  const Scalar _tmp46 = _tmp10 * (_tmp18 + _tmp35 + 1) + _tmp16 * _tmp42 + _tmp21 * _tmp44 +
                        _tmp45 * source_inverse_range;
  const Scalar _tmp47 = _tmp28 * _tmp37 + _tmp41 * _tmp46;
  const Scalar _tmp48 = _tmp23 * _tmp4 + _tmp47;
  const Scalar _tmp49 = std::pow(_tmp48, Scalar(2));
  const Scalar _tmp50 = std::pow(_target_pose[0], Scalar(2));
  const Scalar _tmp51 = -2 * _tmp50;
  const Scalar _tmp52 = _tmp3 + _tmp51;
  const Scalar _tmp53 = -_tmp25;
  const Scalar _tmp54 = _tmp27 + _tmp53;
  const Scalar _tmp55 = 2 * _target_pose[1] * _target_pose[2];
  const Scalar _tmp56 = _target_pose[3] * _tmp26;
  const Scalar _tmp57 = _tmp55 + _tmp56;
  const Scalar _tmp58 = _tmp23 * _tmp54 + _tmp46 * _tmp57;
  const Scalar _tmp59 = _tmp37 * _tmp52 + _tmp58;
  const Scalar _tmp60 = std::pow(_tmp59, Scalar(2));
  const Scalar _tmp61 = _tmp49 + _tmp60 + epsilon;
  const Scalar _tmp62 = std::sqrt(_tmp61);
  const Scalar _tmp63 = _tmp1 + _tmp51 + 1;
  const Scalar _tmp64 = _tmp38 + _tmp39;
  const Scalar _tmp65 = -_tmp56;
  const Scalar _tmp66 = _tmp55 + _tmp65;
  const Scalar _tmp67 = _tmp23 * _tmp64 + _tmp37 * _tmp66;
  const Scalar _tmp68 = _tmp46 * _tmp63 + _tmp67;
  const Scalar _tmp69 = std::atan2(_tmp62, _tmp68);
  const Scalar _tmp70 = -epsilon + target_calibration_storage(4, 0);
  const Scalar _tmp71 = std::min<Scalar>(_tmp69, _tmp70);
  const Scalar _tmp72 = std::pow(_tmp71, Scalar(9)) * target_calibration_storage(8, 0) +
                        std::pow(_tmp71, Scalar(7)) * target_calibration_storage(7, 0) +
                        std::pow(_tmp71, Scalar(5)) * target_calibration_storage(6, 0) +
                        [&]() {
                          const Scalar base = _tmp71;
                          return base * base * base;
                        }() *
                            target_calibration_storage(5, 0) +
                        _tmp71;
  const Scalar _tmp73 = Scalar(1.0) / (_tmp62);
  const Scalar _tmp74 = _tmp73 * target_calibration_storage(0, 0);
  const Scalar _tmp75 = _tmp48 * _tmp74;
  const Scalar _tmp76 = _tmp72 * _tmp75 + target_calibration_storage(2, 0) - target_pixel(0, 0);
  const Scalar _tmp77 = _tmp73 * target_calibration_storage(1, 0);
  const Scalar _tmp78 = _tmp59 * _tmp77;
  const Scalar _tmp79 = _tmp72 * _tmp78 + target_calibration_storage(3, 0) - target_pixel(1, 0);
  const Scalar _tmp80 = std::pow(_tmp76, Scalar(2)) + std::pow(_tmp79, Scalar(2)) + epsilon;
  const Scalar _tmp81 = std::pow(_tmp80, Scalar(Scalar(-1) / Scalar(2)));
  const Scalar _tmp82 = Scalar(1.0) / (epsilon - gnc_mu + 1);
  const Scalar _tmp83 = epsilon + std::fabs(_tmp82);
  const Scalar _tmp84 = 2 - _tmp82;
  const Scalar _tmp85 =
      _tmp84 + epsilon * (2 * std::min<Scalar>(0, (((_tmp84) > 0) - ((_tmp84) < 0))) + 1);
  const Scalar _tmp86 = std::pow(gnc_scale, Scalar(-2));
  const Scalar _tmp87 = _tmp80 * _tmp86 / _tmp83 + 1;
  const Scalar _tmp88 = (Scalar(1) / Scalar(2)) * _tmp85;
  const Scalar _tmp89 = std::sqrt(2) * std::sqrt(_tmp83 * (std::pow(_tmp87, _tmp88) - 1) / _tmp85);
  const Scalar _tmp90 = -_tmp69;
  const Scalar _tmp91 =
      std::sqrt(weight) *
      std::sqrt(std::max<Scalar>(0, (((_tmp90 + target_calibration_storage(4, 0)) > 0) -
                                     ((_tmp90 + target_calibration_storage(4, 0)) < 0))));
  const Scalar _tmp92 = _tmp89 * _tmp91;
  const Scalar _tmp93 = _tmp81 * _tmp92;
  const Scalar _tmp94 = _tmp76 * _tmp93;
  const Scalar _tmp95 = _tmp79 * _tmp93;
  const Scalar _tmp96 = -_tmp11;
  const Scalar _tmp97 = _tmp10 * (_tmp13 + _tmp96) + _tmp16 * _tmp8;
  const Scalar _tmp98 = std::pow(_source_pose[3], Scalar(2));
  const Scalar _tmp99 = -_tmp98;
  const Scalar _tmp100 = _tmp34 + _tmp99;
  const Scalar _tmp101 = -_tmp17;
  const Scalar _tmp102 = _tmp101 + _tmp19;
  const Scalar _tmp103 = _tmp10 * (_tmp100 + _tmp102) + _tmp16 * _tmp32;
  const Scalar _tmp104 = -_tmp29;
  const Scalar _tmp105 = -_tmp34;
  const Scalar _tmp106 = _tmp105 + _tmp98;
  const Scalar _tmp107 = _tmp10 * (_tmp104 + _tmp31) + _tmp16 * (_tmp102 + _tmp106);
  const Scalar _tmp108 = _tmp103 * _tmp52 + _tmp107 * _tmp57 + _tmp54 * _tmp97;
  const Scalar _tmp109 = _tmp72 * _tmp77;
  const Scalar _tmp110 = std::pow(_tmp68, Scalar(2));
  const Scalar _tmp111 = _tmp62 / _tmp110;
  const Scalar _tmp112 = _tmp103 * _tmp28 + _tmp107 * _tmp41 + _tmp4 * _tmp97;
  const Scalar _tmp113 = 2 * _tmp48;
  const Scalar _tmp114 = 2 * _tmp59;
  const Scalar _tmp115 = _tmp108 * _tmp114 + _tmp112 * _tmp113;
  const Scalar _tmp116 = _tmp73 / _tmp68;
  const Scalar _tmp117 = (Scalar(1) / Scalar(2)) * _tmp116;
  const Scalar _tmp118 =
      -_tmp111 * (_tmp103 * _tmp66 + _tmp107 * _tmp63 + _tmp64 * _tmp97) + _tmp115 * _tmp117;
  const Scalar _tmp119 =
      ((((_tmp70 + _tmp90) > 0) - ((_tmp70 + _tmp90) < 0)) + 1) / (_tmp110 + _tmp61);
  const Scalar _tmp120 = _tmp110 * _tmp119;
  const Scalar _tmp121 = std::pow(_tmp71, Scalar(2)) * target_calibration_storage(5, 0);
  const Scalar _tmp122 = (Scalar(3) / Scalar(2)) * _tmp121;
  const Scalar _tmp123 = _tmp120 * _tmp122;
  const Scalar _tmp124 = std::pow(_tmp71, Scalar(8)) * target_calibration_storage(8, 0);
  const Scalar _tmp125 = (Scalar(9) / Scalar(2)) * _tmp124;
  const Scalar _tmp126 = _tmp120 * _tmp125;
  const Scalar _tmp127 = std::pow(_tmp71, Scalar(6)) * target_calibration_storage(7, 0);
  const Scalar _tmp128 = (Scalar(7) / Scalar(2)) * _tmp127;
  const Scalar _tmp129 = _tmp120 * _tmp128;
  const Scalar _tmp130 = (Scalar(1) / Scalar(2)) * _tmp120;
  const Scalar _tmp131 = std::pow(_tmp71, Scalar(4)) * target_calibration_storage(6, 0);
  const Scalar _tmp132 = (Scalar(5) / Scalar(2)) * _tmp131;
  const Scalar _tmp133 = _tmp120 * _tmp132;
  const Scalar _tmp134 = _tmp118 * _tmp123 + _tmp118 * _tmp126 + _tmp118 * _tmp129 +
                         _tmp118 * _tmp130 + _tmp118 * _tmp133;
  const Scalar _tmp135 = std::pow(_tmp61, Scalar(Scalar(-3) / Scalar(2)));
  const Scalar _tmp136 = _tmp135 * target_calibration_storage(1, 0);
  const Scalar _tmp137 = _tmp136 * _tmp72;
  const Scalar _tmp138 = _tmp137 * _tmp59;
  const Scalar _tmp139 = (Scalar(1) / Scalar(2)) * _tmp138;
  const Scalar _tmp140 = _tmp108 * _tmp109 - _tmp115 * _tmp139 + _tmp134 * _tmp78;
  const Scalar _tmp141 = 2 * _tmp79;
  const Scalar _tmp142 = _tmp72 * _tmp74;
  const Scalar _tmp143 = _tmp135 * target_calibration_storage(0, 0);
  const Scalar _tmp144 = _tmp143 * _tmp48 * _tmp72;
  const Scalar _tmp145 = (Scalar(1) / Scalar(2)) * _tmp144;
  const Scalar _tmp146 = _tmp112 * _tmp142 - _tmp115 * _tmp145 + _tmp134 * _tmp75;
  const Scalar _tmp147 = 2 * _tmp76;
  const Scalar _tmp148 = _tmp140 * _tmp141 + _tmp146 * _tmp147;
  const Scalar _tmp149 = (Scalar(1) / Scalar(2)) * _tmp76;
  const Scalar _tmp150 = _tmp148 * _tmp149;
  const Scalar _tmp151 = _tmp81 * _tmp86 * std::pow(_tmp87, Scalar(_tmp88 - 1)) * _tmp91 / _tmp89;
  const Scalar _tmp152 = _tmp92 / (_tmp80 * std::sqrt(_tmp80));
  const Scalar _tmp153 = _tmp146 * _tmp93 + _tmp150 * _tmp151 - _tmp150 * _tmp152;
  const Scalar _tmp154 = -_tmp7;
  const Scalar _tmp155 = -_tmp19;
  const Scalar _tmp156 =
      _tmp10 * (_tmp101 + _tmp155 + _tmp34 + _tmp98) + _tmp21 * (_tmp154 + _tmp43);
  const Scalar _tmp157 = _tmp155 + _tmp17;
  const Scalar _tmp158 = _tmp10 * _tmp44 + _tmp21 * (_tmp100 + _tmp157);
  const Scalar _tmp159 = _tmp10 * _tmp33 + _tmp21 * (_tmp104 + _tmp30);
  const Scalar _tmp160 = _tmp156 * _tmp54 + _tmp158 * _tmp57 + _tmp159 * _tmp52;
  const Scalar _tmp161 = _tmp156 * _tmp4 + _tmp158 * _tmp41 + _tmp159 * _tmp28;
  const Scalar _tmp162 = _tmp113 * _tmp161 + _tmp114 * _tmp160;
  const Scalar _tmp163 =
      -_tmp111 * (_tmp156 * _tmp64 + _tmp158 * _tmp63 + _tmp159 * _tmp66) + _tmp117 * _tmp162;
  const Scalar _tmp164 = _tmp123 * _tmp163 + _tmp126 * _tmp163 + _tmp129 * _tmp163 +
                         _tmp130 * _tmp163 + _tmp133 * _tmp163;
  const Scalar _tmp165 = _tmp142 * _tmp161 - _tmp145 * _tmp162 + _tmp164 * _tmp75;
  const Scalar _tmp166 = _tmp109 * _tmp160 - _tmp139 * _tmp162 + _tmp164 * _tmp78;
  const Scalar _tmp167 = _tmp141 * _tmp166 + _tmp147 * _tmp165;
  const Scalar _tmp168 = _tmp149 * _tmp151;
  const Scalar _tmp169 = _tmp149 * _tmp152;
  const Scalar _tmp170 = _tmp165 * _tmp93 + _tmp167 * _tmp168 - _tmp167 * _tmp169;
  const Scalar _tmp171 = _tmp15 * _tmp21 + _tmp16 * (_tmp105 + _tmp17 + _tmp19 + _tmp99);
  const Scalar _tmp172 = _tmp16 * (_tmp154 + _tmp6) + _tmp21 * _tmp42;
  const Scalar _tmp173 = _tmp16 * (_tmp14 + _tmp96) + _tmp21 * (_tmp106 + _tmp157);
  const Scalar _tmp174 = _tmp171 * _tmp4 + _tmp172 * _tmp41 + _tmp173 * _tmp28;
  const Scalar _tmp175 = _tmp171 * _tmp54 + _tmp172 * _tmp57 + _tmp173 * _tmp52;
  const Scalar _tmp176 = _tmp113 * _tmp174 + _tmp114 * _tmp175;
  const Scalar _tmp177 =
      -_tmp111 * (_tmp171 * _tmp64 + _tmp172 * _tmp63 + _tmp173 * _tmp66) + _tmp117 * _tmp176;
  const Scalar _tmp178 = _tmp123 * _tmp177 + _tmp126 * _tmp177 + _tmp129 * _tmp177 +
                         _tmp130 * _tmp177 + _tmp133 * _tmp177;
  const Scalar _tmp179 = _tmp142 * _tmp174 - _tmp145 * _tmp176 + _tmp178 * _tmp75;
  const Scalar _tmp180 = _tmp109 * _tmp175 - _tmp139 * _tmp176 + _tmp178 * _tmp78;
  const Scalar _tmp181 = _tmp141 * _tmp180 + _tmp147 * _tmp179;
  const Scalar _tmp182 = _tmp152 * _tmp181;
  const Scalar _tmp183 = -_tmp149 * _tmp182 + _tmp168 * _tmp181 + _tmp179 * _tmp93;
  const Scalar _tmp184 = _tmp114 * source_inverse_range;
  const Scalar _tmp185 = _tmp184 * _tmp54;
  const Scalar _tmp186 = _tmp113 * source_inverse_range;
  const Scalar _tmp187 = _tmp186 * _tmp4;
  const Scalar _tmp188 = _tmp185 + _tmp187;
  const Scalar _tmp189 = _tmp109 * source_inverse_range;
  const Scalar _tmp190 = _tmp189 * _tmp54;
  const Scalar _tmp191 = _tmp111 * source_inverse_range;
  const Scalar _tmp192 = _tmp191 * _tmp64;
  const Scalar _tmp193 = _tmp117 * _tmp188 - _tmp192;
  const Scalar _tmp194 = _tmp123 * _tmp193 + _tmp126 * _tmp193 + _tmp129 * _tmp193 +
                         _tmp130 * _tmp193 + _tmp133 * _tmp193;
  const Scalar _tmp195 = -_tmp139 * _tmp188 + _tmp190 + _tmp194 * _tmp78;
  const Scalar _tmp196 = _tmp142 * source_inverse_range;
  const Scalar _tmp197 = _tmp196 * _tmp4;
  const Scalar _tmp198 = -_tmp145 * _tmp188 + _tmp194 * _tmp75 + _tmp197;
  const Scalar _tmp199 = _tmp141 * _tmp195 + _tmp147 * _tmp198;
  const Scalar _tmp200 = _tmp168 * _tmp199 - _tmp169 * _tmp199 + _tmp198 * _tmp93;
  const Scalar _tmp201 = _tmp186 * _tmp28;
  const Scalar _tmp202 = _tmp184 * _tmp52;
  const Scalar _tmp203 = _tmp201 + _tmp202;
  const Scalar _tmp204 = _tmp191 * _tmp66;
  const Scalar _tmp205 = _tmp117 * _tmp203 - _tmp204;
  const Scalar _tmp206 = _tmp120 * _tmp205;
  const Scalar _tmp207 = _tmp122 * _tmp206 + _tmp125 * _tmp206 + _tmp128 * _tmp206 +
                         _tmp130 * _tmp205 + _tmp132 * _tmp206;
  const Scalar _tmp208 = _tmp189 * _tmp52;
  const Scalar _tmp209 = -_tmp139 * _tmp203 + _tmp207 * _tmp78 + _tmp208;
  const Scalar _tmp210 = _tmp196 * _tmp28;
  const Scalar _tmp211 = -_tmp145 * _tmp203 + _tmp207 * _tmp75 + _tmp210;
  const Scalar _tmp212 = _tmp141 * _tmp209 + _tmp147 * _tmp211;
  const Scalar _tmp213 = _tmp168 * _tmp212 - _tmp169 * _tmp212 + _tmp211 * _tmp93;
  const Scalar _tmp214 = _tmp189 * _tmp57;
  const Scalar _tmp215 = _tmp191 * _tmp63;
  const Scalar _tmp216 = _tmp186 * _tmp41;
  const Scalar _tmp217 = _tmp184 * _tmp57;
  const Scalar _tmp218 = (Scalar(1) / Scalar(2)) * _tmp216 + (Scalar(1) / Scalar(2)) * _tmp217;
  const Scalar _tmp219 = _tmp116 * _tmp218 - _tmp215;
  const Scalar _tmp220 = _tmp123 * _tmp219 + _tmp126 * _tmp219 + _tmp129 * _tmp219 +
                         _tmp130 * _tmp219 + _tmp133 * _tmp219;
  const Scalar _tmp221 = -_tmp138 * _tmp218 + _tmp214 + _tmp220 * _tmp78;
  const Scalar _tmp222 = _tmp196 * _tmp41;
  const Scalar _tmp223 = -_tmp144 * _tmp218 + _tmp220 * _tmp75 + _tmp222;
  const Scalar _tmp224 = _tmp141 * _tmp221 + _tmp147 * _tmp223;
  const Scalar _tmp225 = _tmp168 * _tmp224 - _tmp169 * _tmp224 + _tmp223 * _tmp93;
  const Scalar _tmp226 = std::pow(_target_pose[3], Scalar(2));
  const Scalar _tmp227 = -_tmp50;
  const Scalar _tmp228 = -_tmp0;
  const Scalar _tmp229 = _tmp46 * (_tmp2 + _tmp226 + _tmp227 + _tmp228) + _tmp67;
  const Scalar _tmp230 = _tmp229 * _tmp59;
  const Scalar _tmp231 = -_tmp27;
  const Scalar _tmp232 = -_tmp226;
  const Scalar _tmp233 = _tmp2 + _tmp232;
  const Scalar _tmp234 = _tmp228 + _tmp50;
  const Scalar _tmp235 = -_tmp55;
  const Scalar _tmp236 = -_tmp111 * (_tmp23 * (_tmp231 + _tmp25) + _tmp37 * (_tmp233 + _tmp234) +
                                     _tmp46 * (_tmp235 + _tmp65)) +
                         _tmp116 * _tmp230;
  const Scalar _tmp237 = _tmp120 * _tmp236;
  const Scalar _tmp238 = _tmp122 * _tmp237 + _tmp125 * _tmp237 + _tmp128 * _tmp237 +
                         _tmp130 * _tmp236 + _tmp132 * _tmp237;
  const Scalar _tmp239 = _tmp109 * _tmp229 - _tmp137 * _tmp229 * _tmp60 + _tmp238 * _tmp78;
  const Scalar _tmp240 = -_tmp144 * _tmp230 + _tmp238 * _tmp75;
  const Scalar _tmp241 = _tmp141 * _tmp239 + _tmp147 * _tmp240;
  const Scalar _tmp242 = _tmp168 * _tmp241 - _tmp169 * _tmp241 + _tmp240 * _tmp93;
  const Scalar _tmp243 = -_tmp38;
  const Scalar _tmp244 = -_tmp2;
  const Scalar _tmp245 = _tmp23 * (_tmp243 + _tmp40) + _tmp37 * (_tmp235 + _tmp56) +
                         _tmp46 * (_tmp0 + _tmp232 + _tmp244 + _tmp50);
  const Scalar _tmp246 = _tmp226 + _tmp244;
  const Scalar _tmp247 =
      -_tmp111 * (_tmp23 * (_tmp234 + _tmp246) + _tmp47) + _tmp116 * _tmp245 * _tmp48;
  const Scalar _tmp248 = _tmp123 * _tmp247 + _tmp126 * _tmp247 + _tmp129 * _tmp247 +
                         _tmp130 * _tmp247 + _tmp133 * _tmp247;
  const Scalar _tmp249 = _tmp245 * _tmp72;
  const Scalar _tmp250 = -_tmp143 * _tmp249 * _tmp49 + _tmp248 * _tmp75 + _tmp249 * _tmp74;
  const Scalar _tmp251 = -_tmp136 * _tmp249 * _tmp48 * _tmp59 + _tmp248 * _tmp78;
  const Scalar _tmp252 = _tmp141 * _tmp251 + _tmp147 * _tmp250;
  const Scalar _tmp253 = _tmp168 * _tmp252 - _tmp169 * _tmp252 + _tmp250 * _tmp93;
  const Scalar _tmp254 = _tmp0 + _tmp227;
  const Scalar _tmp255 = _tmp37 * (_tmp246 + _tmp254) + _tmp58;
  const Scalar _tmp256 =
      _tmp23 * (_tmp233 + _tmp254) + _tmp37 * (_tmp231 + _tmp53) + _tmp46 * (_tmp243 + _tmp39);
  const Scalar _tmp257 = _tmp113 * _tmp255 + _tmp114 * _tmp256;
  const Scalar _tmp258 = _tmp119 * _tmp257 * _tmp68 * _tmp73;
  const Scalar _tmp259 =
      (Scalar(3) / Scalar(4)) * _tmp121 * _tmp258 + (Scalar(9) / Scalar(4)) * _tmp124 * _tmp258 +
      (Scalar(7) / Scalar(4)) * _tmp127 * _tmp258 + (Scalar(5) / Scalar(4)) * _tmp131 * _tmp258 +
      (Scalar(1) / Scalar(4)) * _tmp258;
  const Scalar _tmp260 = (Scalar(1) / Scalar(2)) * _tmp257;
  const Scalar _tmp261 = _tmp109 * _tmp256 - _tmp138 * _tmp260 + _tmp259 * _tmp78;
  const Scalar _tmp262 = _tmp142 * _tmp255 - _tmp144 * _tmp260 + _tmp259 * _tmp75;
  const Scalar _tmp263 = _tmp141 * _tmp261 + _tmp147 * _tmp262;
  const Scalar _tmp264 = _tmp168 * _tmp263 - _tmp169 * _tmp263 + _tmp262 * _tmp93;
  const Scalar _tmp265 = -_tmp185 - _tmp187;
  const Scalar _tmp266 = _tmp117 * _tmp265 + _tmp192;
  const Scalar _tmp267 = _tmp123 * _tmp266 + _tmp126 * _tmp266 + _tmp129 * _tmp266 +
                         _tmp130 * _tmp266 + _tmp133 * _tmp266;
  const Scalar _tmp268 = -_tmp145 * _tmp265 - _tmp197 + _tmp267 * _tmp75;
  const Scalar _tmp269 = -_tmp139 * _tmp265 - _tmp190 + _tmp267 * _tmp78;
  const Scalar _tmp270 = _tmp141 * _tmp269 + _tmp147 * _tmp268;
  const Scalar _tmp271 = _tmp168 * _tmp270 - _tmp169 * _tmp270 + _tmp268 * _tmp93;
  const Scalar _tmp272 = -_tmp201 - _tmp202;
  const Scalar _tmp273 = (Scalar(1) / Scalar(2)) * _tmp272;
  const Scalar _tmp274 = _tmp117 * _tmp272 + _tmp204;
  const Scalar _tmp275 = _tmp120 * _tmp274;
  const Scalar _tmp276 = _tmp122 * _tmp275 + _tmp125 * _tmp275 + _tmp128 * _tmp275 +
                         _tmp130 * _tmp274 + _tmp132 * _tmp275;
  const Scalar _tmp277 = -_tmp138 * _tmp273 - _tmp208 + _tmp276 * _tmp78;
  const Scalar _tmp278 = -_tmp144 * _tmp273 - _tmp210 + _tmp276 * _tmp75;
  const Scalar _tmp279 = _tmp141 * _tmp277 + _tmp147 * _tmp278;
  const Scalar _tmp280 = _tmp168 * _tmp279 - _tmp169 * _tmp279 + _tmp278 * _tmp93;
  const Scalar _tmp281 = -_tmp216 - _tmp217;
  const Scalar _tmp282 = _tmp117 * _tmp281 + _tmp215;
  const Scalar _tmp283 = _tmp123 * _tmp282 + _tmp126 * _tmp282 + _tmp129 * _tmp282 +
                         _tmp130 * _tmp282 + _tmp133 * _tmp282;
  const Scalar _tmp284 = -_tmp139 * _tmp281 - _tmp214 + _tmp283 * _tmp78;
  const Scalar _tmp285 = -_tmp145 * _tmp281 - _tmp222 + _tmp283 * _tmp75;
  const Scalar _tmp286 = _tmp141 * _tmp284 + _tmp147 * _tmp285;
  const Scalar _tmp287 = _tmp168 * _tmp286 - _tmp169 * _tmp286 + _tmp285 * _tmp93;
  const Scalar _tmp288 = _tmp22 * _tmp54 + _tmp36 * _tmp52 + _tmp45 * _tmp57;
  const Scalar _tmp289 = _tmp22 * _tmp4 + _tmp28 * _tmp36 + _tmp41 * _tmp45;
  const Scalar _tmp290 = _tmp113 * _tmp289 + _tmp114 * _tmp288;
  const Scalar _tmp291 =
      -_tmp111 * (_tmp22 * _tmp64 + _tmp36 * _tmp66 + _tmp45 * _tmp63) + _tmp117 * _tmp290;
  const Scalar _tmp292 = _tmp123 * _tmp291 + _tmp126 * _tmp291 + _tmp129 * _tmp291 +
                         _tmp130 * _tmp291 + _tmp133 * _tmp291;
  const Scalar _tmp293 = _tmp142 * _tmp289 - _tmp145 * _tmp290 + _tmp292 * _tmp75;
  const Scalar _tmp294 = _tmp109 * _tmp288 - _tmp139 * _tmp290 + _tmp292 * _tmp78;
  const Scalar _tmp295 = _tmp141 * _tmp294 + _tmp147 * _tmp293;
  const Scalar _tmp296 = _tmp168 * _tmp295 - _tmp169 * _tmp295 + _tmp293 * _tmp93;
  const Scalar _tmp297 = (Scalar(1) / Scalar(2)) * _tmp79;
  const Scalar _tmp298 = _tmp152 * _tmp297;
  const Scalar _tmp299 = _tmp151 * _tmp297;
  const Scalar _tmp300 = _tmp140 * _tmp93 - _tmp148 * _tmp298 + _tmp148 * _tmp299;
  const Scalar _tmp301 = _tmp166 * _tmp93 - _tmp167 * _tmp298 + _tmp167 * _tmp299;
  const Scalar _tmp302 = _tmp180 * _tmp93 + _tmp181 * _tmp299 - _tmp182 * _tmp297;
  const Scalar _tmp303 = _tmp195 * _tmp93 - _tmp199 * _tmp298 + _tmp199 * _tmp299;
  const Scalar _tmp304 = _tmp209 * _tmp93 - _tmp212 * _tmp298 + _tmp212 * _tmp299;
  const Scalar _tmp305 = _tmp221 * _tmp93 - _tmp224 * _tmp298 + _tmp224 * _tmp299;
  const Scalar _tmp306 = _tmp239 * _tmp93 - _tmp241 * _tmp298 + _tmp241 * _tmp299;
  const Scalar _tmp307 = _tmp251 * _tmp93 - _tmp252 * _tmp298 + _tmp252 * _tmp299;
  const Scalar _tmp308 = _tmp261 * _tmp93 - _tmp263 * _tmp298 + _tmp263 * _tmp299;
  const Scalar _tmp309 = _tmp269 * _tmp93 - _tmp270 * _tmp298 + _tmp270 * _tmp299;
  const Scalar _tmp310 = _tmp277 * _tmp93 - _tmp279 * _tmp298 + _tmp279 * _tmp299;
  const Scalar _tmp311 = _tmp284 * _tmp93 - _tmp286 * _tmp298 + _tmp286 * _tmp299;
  const Scalar _tmp312 = _tmp294 * _tmp93 - _tmp295 * _tmp298 + _tmp295 * _tmp299;

  // Output terms (4)
  if (res != nullptr) {
    Eigen::Matrix<Scalar, 2, 1>& _res = (*res);

    _res(0, 0) = _tmp94;
    _res(1, 0) = _tmp95;
  }

  if (jacobian != nullptr) {
    Eigen::Matrix<Scalar, 2, 13>& _jacobian = (*jacobian);

    _jacobian(0, 0) = _tmp153;
    _jacobian(0, 1) = _tmp170;
    _jacobian(0, 2) = _tmp183;
    _jacobian(0, 3) = _tmp200;
    _jacobian(0, 4) = _tmp213;
    _jacobian(0, 5) = _tmp225;
    _jacobian(0, 6) = _tmp242;
    _jacobian(0, 7) = _tmp253;
    _jacobian(0, 8) = _tmp264;
    _jacobian(0, 9) = _tmp271;
    _jacobian(0, 10) = _tmp280;
    _jacobian(0, 11) = _tmp287;
    _jacobian(0, 12) = _tmp296;
    _jacobian(1, 0) = _tmp300;
    _jacobian(1, 1) = _tmp301;
    _jacobian(1, 2) = _tmp302;
    _jacobian(1, 3) = _tmp303;
    _jacobian(1, 4) = _tmp304;
    _jacobian(1, 5) = _tmp305;
    _jacobian(1, 6) = _tmp306;
    _jacobian(1, 7) = _tmp307;
    _jacobian(1, 8) = _tmp308;
    _jacobian(1, 9) = _tmp309;
    _jacobian(1, 10) = _tmp310;
    _jacobian(1, 11) = _tmp311;
    _jacobian(1, 12) = _tmp312;
  }

  if (hessian != nullptr) {
    Eigen::Matrix<Scalar, 13, 13>& _hessian = (*hessian);

    _hessian(0, 0) = std::pow(_tmp153, Scalar(2)) + std::pow(_tmp300, Scalar(2));
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
    _hessian(0, 12) = 0;
    _hessian(1, 0) = _tmp153 * _tmp170 + _tmp300 * _tmp301;
    _hessian(1, 1) = std::pow(_tmp170, Scalar(2)) + std::pow(_tmp301, Scalar(2));
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
    _hessian(1, 12) = 0;
    _hessian(2, 0) = _tmp153 * _tmp183 + _tmp300 * _tmp302;
    _hessian(2, 1) = _tmp170 * _tmp183 + _tmp301 * _tmp302;
    _hessian(2, 2) = std::pow(_tmp183, Scalar(2)) + std::pow(_tmp302, Scalar(2));
    _hessian(2, 3) = 0;
    _hessian(2, 4) = 0;
    _hessian(2, 5) = 0;
    _hessian(2, 6) = 0;
    _hessian(2, 7) = 0;
    _hessian(2, 8) = 0;
    _hessian(2, 9) = 0;
    _hessian(2, 10) = 0;
    _hessian(2, 11) = 0;
    _hessian(2, 12) = 0;
    _hessian(3, 0) = _tmp153 * _tmp200 + _tmp300 * _tmp303;
    _hessian(3, 1) = _tmp170 * _tmp200 + _tmp301 * _tmp303;
    _hessian(3, 2) = _tmp183 * _tmp200 + _tmp302 * _tmp303;
    _hessian(3, 3) = std::pow(_tmp200, Scalar(2)) + std::pow(_tmp303, Scalar(2));
    _hessian(3, 4) = 0;
    _hessian(3, 5) = 0;
    _hessian(3, 6) = 0;
    _hessian(3, 7) = 0;
    _hessian(3, 8) = 0;
    _hessian(3, 9) = 0;
    _hessian(3, 10) = 0;
    _hessian(3, 11) = 0;
    _hessian(3, 12) = 0;
    _hessian(4, 0) = _tmp153 * _tmp213 + _tmp300 * _tmp304;
    _hessian(4, 1) = _tmp170 * _tmp213 + _tmp301 * _tmp304;
    _hessian(4, 2) = _tmp183 * _tmp213 + _tmp302 * _tmp304;
    _hessian(4, 3) = _tmp200 * _tmp213 + _tmp303 * _tmp304;
    _hessian(4, 4) = std::pow(_tmp213, Scalar(2)) + std::pow(_tmp304, Scalar(2));
    _hessian(4, 5) = 0;
    _hessian(4, 6) = 0;
    _hessian(4, 7) = 0;
    _hessian(4, 8) = 0;
    _hessian(4, 9) = 0;
    _hessian(4, 10) = 0;
    _hessian(4, 11) = 0;
    _hessian(4, 12) = 0;
    _hessian(5, 0) = _tmp153 * _tmp225 + _tmp300 * _tmp305;
    _hessian(5, 1) = _tmp170 * _tmp225 + _tmp301 * _tmp305;
    _hessian(5, 2) = _tmp183 * _tmp225 + _tmp302 * _tmp305;
    _hessian(5, 3) = _tmp200 * _tmp225 + _tmp303 * _tmp305;
    _hessian(5, 4) = _tmp213 * _tmp225 + _tmp304 * _tmp305;
    _hessian(5, 5) = std::pow(_tmp225, Scalar(2)) + std::pow(_tmp305, Scalar(2));
    _hessian(5, 6) = 0;
    _hessian(5, 7) = 0;
    _hessian(5, 8) = 0;
    _hessian(5, 9) = 0;
    _hessian(5, 10) = 0;
    _hessian(5, 11) = 0;
    _hessian(5, 12) = 0;
    _hessian(6, 0) = _tmp153 * _tmp242 + _tmp300 * _tmp306;
    _hessian(6, 1) = _tmp170 * _tmp242 + _tmp301 * _tmp306;
    _hessian(6, 2) = _tmp183 * _tmp242 + _tmp302 * _tmp306;
    _hessian(6, 3) = _tmp200 * _tmp242 + _tmp303 * _tmp306;
    _hessian(6, 4) = _tmp213 * _tmp242 + _tmp304 * _tmp306;
    _hessian(6, 5) = _tmp225 * _tmp242 + _tmp305 * _tmp306;
    _hessian(6, 6) = std::pow(_tmp242, Scalar(2)) + std::pow(_tmp306, Scalar(2));
    _hessian(6, 7) = 0;
    _hessian(6, 8) = 0;
    _hessian(6, 9) = 0;
    _hessian(6, 10) = 0;
    _hessian(6, 11) = 0;
    _hessian(6, 12) = 0;
    _hessian(7, 0) = _tmp153 * _tmp253 + _tmp300 * _tmp307;
    _hessian(7, 1) = _tmp170 * _tmp253 + _tmp301 * _tmp307;
    _hessian(7, 2) = _tmp183 * _tmp253 + _tmp302 * _tmp307;
    _hessian(7, 3) = _tmp200 * _tmp253 + _tmp303 * _tmp307;
    _hessian(7, 4) = _tmp213 * _tmp253 + _tmp304 * _tmp307;
    _hessian(7, 5) = _tmp225 * _tmp253 + _tmp305 * _tmp307;
    _hessian(7, 6) = _tmp242 * _tmp253 + _tmp306 * _tmp307;
    _hessian(7, 7) = std::pow(_tmp253, Scalar(2)) + std::pow(_tmp307, Scalar(2));
    _hessian(7, 8) = 0;
    _hessian(7, 9) = 0;
    _hessian(7, 10) = 0;
    _hessian(7, 11) = 0;
    _hessian(7, 12) = 0;
    _hessian(8, 0) = _tmp153 * _tmp264 + _tmp300 * _tmp308;
    _hessian(8, 1) = _tmp170 * _tmp264 + _tmp301 * _tmp308;
    _hessian(8, 2) = _tmp183 * _tmp264 + _tmp302 * _tmp308;
    _hessian(8, 3) = _tmp200 * _tmp264 + _tmp303 * _tmp308;
    _hessian(8, 4) = _tmp213 * _tmp264 + _tmp304 * _tmp308;
    _hessian(8, 5) = _tmp225 * _tmp264 + _tmp305 * _tmp308;
    _hessian(8, 6) = _tmp242 * _tmp264 + _tmp306 * _tmp308;
    _hessian(8, 7) = _tmp253 * _tmp264 + _tmp307 * _tmp308;
    _hessian(8, 8) = std::pow(_tmp264, Scalar(2)) + std::pow(_tmp308, Scalar(2));
    _hessian(8, 9) = 0;
    _hessian(8, 10) = 0;
    _hessian(8, 11) = 0;
    _hessian(8, 12) = 0;
    _hessian(9, 0) = _tmp153 * _tmp271 + _tmp300 * _tmp309;
    _hessian(9, 1) = _tmp170 * _tmp271 + _tmp301 * _tmp309;
    _hessian(9, 2) = _tmp183 * _tmp271 + _tmp302 * _tmp309;
    _hessian(9, 3) = _tmp200 * _tmp271 + _tmp303 * _tmp309;
    _hessian(9, 4) = _tmp213 * _tmp271 + _tmp304 * _tmp309;
    _hessian(9, 5) = _tmp225 * _tmp271 + _tmp305 * _tmp309;
    _hessian(9, 6) = _tmp242 * _tmp271 + _tmp306 * _tmp309;
    _hessian(9, 7) = _tmp253 * _tmp271 + _tmp307 * _tmp309;
    _hessian(9, 8) = _tmp264 * _tmp271 + _tmp308 * _tmp309;
    _hessian(9, 9) = std::pow(_tmp271, Scalar(2)) + std::pow(_tmp309, Scalar(2));
    _hessian(9, 10) = 0;
    _hessian(9, 11) = 0;
    _hessian(9, 12) = 0;
    _hessian(10, 0) = _tmp153 * _tmp280 + _tmp300 * _tmp310;
    _hessian(10, 1) = _tmp170 * _tmp280 + _tmp301 * _tmp310;
    _hessian(10, 2) = _tmp183 * _tmp280 + _tmp302 * _tmp310;
    _hessian(10, 3) = _tmp200 * _tmp280 + _tmp303 * _tmp310;
    _hessian(10, 4) = _tmp213 * _tmp280 + _tmp304 * _tmp310;
    _hessian(10, 5) = _tmp225 * _tmp280 + _tmp305 * _tmp310;
    _hessian(10, 6) = _tmp242 * _tmp280 + _tmp306 * _tmp310;
    _hessian(10, 7) = _tmp253 * _tmp280 + _tmp307 * _tmp310;
    _hessian(10, 8) = _tmp264 * _tmp280 + _tmp308 * _tmp310;
    _hessian(10, 9) = _tmp271 * _tmp280 + _tmp309 * _tmp310;
    _hessian(10, 10) = std::pow(_tmp280, Scalar(2)) + std::pow(_tmp310, Scalar(2));
    _hessian(10, 11) = 0;
    _hessian(10, 12) = 0;
    _hessian(11, 0) = _tmp153 * _tmp287 + _tmp300 * _tmp311;
    _hessian(11, 1) = _tmp170 * _tmp287 + _tmp301 * _tmp311;
    _hessian(11, 2) = _tmp183 * _tmp287 + _tmp302 * _tmp311;
    _hessian(11, 3) = _tmp200 * _tmp287 + _tmp303 * _tmp311;
    _hessian(11, 4) = _tmp213 * _tmp287 + _tmp304 * _tmp311;
    _hessian(11, 5) = _tmp225 * _tmp287 + _tmp305 * _tmp311;
    _hessian(11, 6) = _tmp242 * _tmp287 + _tmp306 * _tmp311;
    _hessian(11, 7) = _tmp253 * _tmp287 + _tmp307 * _tmp311;
    _hessian(11, 8) = _tmp264 * _tmp287 + _tmp308 * _tmp311;
    _hessian(11, 9) = _tmp271 * _tmp287 + _tmp309 * _tmp311;
    _hessian(11, 10) = _tmp280 * _tmp287 + _tmp310 * _tmp311;
    _hessian(11, 11) = std::pow(_tmp287, Scalar(2)) + std::pow(_tmp311, Scalar(2));
    _hessian(11, 12) = 0;
    _hessian(12, 0) = _tmp153 * _tmp296 + _tmp300 * _tmp312;
    _hessian(12, 1) = _tmp170 * _tmp296 + _tmp301 * _tmp312;
    _hessian(12, 2) = _tmp183 * _tmp296 + _tmp302 * _tmp312;
    _hessian(12, 3) = _tmp200 * _tmp296 + _tmp303 * _tmp312;
    _hessian(12, 4) = _tmp213 * _tmp296 + _tmp304 * _tmp312;
    _hessian(12, 5) = _tmp225 * _tmp296 + _tmp305 * _tmp312;
    _hessian(12, 6) = _tmp242 * _tmp296 + _tmp306 * _tmp312;
    _hessian(12, 7) = _tmp253 * _tmp296 + _tmp307 * _tmp312;
    _hessian(12, 8) = _tmp264 * _tmp296 + _tmp308 * _tmp312;
    _hessian(12, 9) = _tmp271 * _tmp296 + _tmp309 * _tmp312;
    _hessian(12, 10) = _tmp280 * _tmp296 + _tmp310 * _tmp312;
    _hessian(12, 11) = _tmp287 * _tmp296 + _tmp311 * _tmp312;
    _hessian(12, 12) = std::pow(_tmp296, Scalar(2)) + std::pow(_tmp312, Scalar(2));
  }

  if (rhs != nullptr) {
    Eigen::Matrix<Scalar, 13, 1>& _rhs = (*rhs);

    _rhs(0, 0) = _tmp153 * _tmp94 + _tmp300 * _tmp95;
    _rhs(1, 0) = _tmp170 * _tmp94 + _tmp301 * _tmp95;
    _rhs(2, 0) = _tmp183 * _tmp94 + _tmp302 * _tmp95;
    _rhs(3, 0) = _tmp200 * _tmp94 + _tmp303 * _tmp95;
    _rhs(4, 0) = _tmp213 * _tmp94 + _tmp304 * _tmp95;
    _rhs(5, 0) = _tmp225 * _tmp94 + _tmp305 * _tmp95;
    _rhs(6, 0) = _tmp242 * _tmp94 + _tmp306 * _tmp95;
    _rhs(7, 0) = _tmp253 * _tmp94 + _tmp307 * _tmp95;
    _rhs(8, 0) = _tmp264 * _tmp94 + _tmp308 * _tmp95;
    _rhs(9, 0) = _tmp271 * _tmp94 + _tmp309 * _tmp95;
    _rhs(10, 0) = _tmp280 * _tmp94 + _tmp310 * _tmp95;
    _rhs(11, 0) = _tmp287 * _tmp94 + _tmp311 * _tmp95;
    _rhs(12, 0) = _tmp296 * _tmp94 + _tmp312 * _tmp95;
  }
}  // NOLINT(readability/fn_size)

// NOLINTNEXTLINE(readability/fn_size)
}  // namespace sym