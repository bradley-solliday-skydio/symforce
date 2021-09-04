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
 * Return the 2dof residual of reprojecting the landmark into the target camera and comparing
 * against the correspondence in the target camera.
 *
 * The landmark is specified as a pixel in the source camera and an inverse range; this means the
 * landmark is fixed in the source camera and always has residual 0 there (this 0 residual is not
 * returned, only the residual in the target camera is returned).
 *
 * The norm of the residual is whitened using the Barron noise model.  Whitening each component of
 * the reprojection error separately would result in rejecting individual components as outliers.
 * Instead, we minimize the whitened norm of the full reprojection error for each point.  See the
 * docstring for `NoiseModel.whiten_norm` for more information on this, and the docstring of
 * `BarronNoiseModel` for more information on the noise model.
 *
 * Args:
 *     source_pose: The pose of the source camera
 *     source_calibration_storage: The storage vector of the source (Linear) camera calibration
 *     target_pose: The pose of the target camera
 *     target_calibration_storage: The storage vector of the target (Linear) camera calibration
 *     source_inverse_range: The inverse range of the landmark in the source camera
 *     source_pixel: The location of the landmark in the source camera
 *     target_pixel: The location of the correspondence in the target camera
 *     weight: The weight of the factor
 *     gnc_mu: The mu convexity parameter for the Barron noise model
 *     gnc_scale: The scale parameter for the Barron noise model
 *     epsilon: Small positive value
 *
 * Outputs:
 *     res: 2dof residual of the reprojection
 *     jacobian: (2x13) jacobian of res wrt args source_pose (6), target_pose (6),
 *               source_inverse_range (1)
 *     hessian: (13x13) Gauss-Newton hessian for args source_pose (6), target_pose (6),
 *              source_inverse_range (1)
 *     rhs: (13x1) Gauss-Newton rhs for args source_pose (6), target_pose (6), source_inverse_range
 *          (1)
 */
template <typename Scalar>
void InverseRangeLandmarkReprojectionErrorFactor(
    const sym::Pose3<Scalar>& source_pose,
    const Eigen::Matrix<Scalar, 4, 1>& source_calibration_storage,
    const sym::Pose3<Scalar>& target_pose,
    const Eigen::Matrix<Scalar, 4, 1>& target_calibration_storage,
    const Scalar source_inverse_range, const Eigen::Matrix<Scalar, 2, 1>& source_pixel,
    const Eigen::Matrix<Scalar, 2, 1>& target_pixel, const Scalar weight, const Scalar gnc_mu,
    const Scalar gnc_scale, const Scalar epsilon, Eigen::Matrix<Scalar, 2, 1>* const res = nullptr,
    Eigen::Matrix<Scalar, 2, 13>* const jacobian = nullptr,
    Eigen::Matrix<Scalar, 13, 13>* const hessian = nullptr,
    Eigen::Matrix<Scalar, 13, 1>* const rhs = nullptr) {
  // Total ops: 1237

  // Input arrays
  const Eigen::Matrix<Scalar, 7, 1>& _source_pose = source_pose.Data();
  const Eigen::Matrix<Scalar, 7, 1>& _target_pose = target_pose.Data();

  // Intermediate terms (276)
  const Scalar _tmp0 = 2 * _target_pose[0] * _target_pose[2];
  const Scalar _tmp1 = 2 * _target_pose[1];
  const Scalar _tmp2 = _target_pose[3] * _tmp1;
  const Scalar _tmp3 = _tmp0 - _tmp2;
  const Scalar _tmp4 = _source_pose[6] - _target_pose[6];
  const Scalar _tmp5 = 2 * _target_pose[3];
  const Scalar _tmp6 = _target_pose[2] * _tmp5;
  const Scalar _tmp7 = _target_pose[0] * _tmp1;
  const Scalar _tmp8 = _tmp6 + _tmp7;
  const Scalar _tmp9 = _source_pose[5] - _target_pose[5];
  const Scalar _tmp10 = -2 * std::pow(_target_pose[2], Scalar(2));
  const Scalar _tmp11 = -2 * std::pow(_target_pose[1], Scalar(2));
  const Scalar _tmp12 = _tmp10 + _tmp11 + 1;
  const Scalar _tmp13 = _source_pose[4] - _target_pose[4];
  const Scalar _tmp14 = _tmp12 * _tmp13 + _tmp3 * _tmp4 + _tmp8 * _tmp9;
  const Scalar _tmp15 = 2 * _source_pose[1];
  const Scalar _tmp16 = _source_pose[3] * _tmp15;
  const Scalar _tmp17 = 2 * _source_pose[2];
  const Scalar _tmp18 = _source_pose[0] * _tmp17;
  const Scalar _tmp19 = -source_calibration_storage(3, 0) + source_pixel(1, 0);
  const Scalar _tmp20 = -source_calibration_storage(2, 0) + source_pixel(0, 0);
  const Scalar _tmp21 = std::pow(
      Scalar(std::pow(_tmp19, Scalar(2)) / std::pow(source_calibration_storage(1, 0), Scalar(2)) +
             std::pow(_tmp20, Scalar(2)) / std::pow(source_calibration_storage(0, 0), Scalar(2)) +
             1),
      Scalar(Scalar(-1) / Scalar(2)));
  const Scalar _tmp22 = -2 * std::pow(_source_pose[1], Scalar(2));
  const Scalar _tmp23 = 1 - 2 * std::pow(_source_pose[2], Scalar(2));
  const Scalar _tmp24 = _tmp20 / source_calibration_storage(0, 0);
  const Scalar _tmp25 = _tmp21 * _tmp24;
  const Scalar _tmp26 = _source_pose[0] * _tmp15;
  const Scalar _tmp27 = _source_pose[3] * _tmp17;
  const Scalar _tmp28 = _tmp19 / source_calibration_storage(1, 0);
  const Scalar _tmp29 = _tmp21 * _tmp28;
  const Scalar _tmp30 =
      _tmp21 * (_tmp16 + _tmp18) + _tmp25 * (_tmp22 + _tmp23) + _tmp29 * (_tmp26 - _tmp27);
  const Scalar _tmp31 = _source_pose[1] * _tmp17;
  const Scalar _tmp32 = 2 * _source_pose[0] * _source_pose[3];
  const Scalar _tmp33 = -2 * std::pow(_source_pose[0], Scalar(2));
  const Scalar _tmp34 =
      _tmp21 * (_tmp31 - _tmp32) + _tmp25 * (_tmp26 + _tmp27) + _tmp29 * (_tmp23 + _tmp33);
  const Scalar _tmp35 =
      _tmp21 * (_tmp22 + _tmp33 + 1) + _tmp25 * (-_tmp16 + _tmp18) + _tmp29 * (_tmp31 + _tmp32);
  const Scalar _tmp36 =
      _tmp12 * _tmp30 + _tmp14 * source_inverse_range + _tmp3 * _tmp35 + _tmp34 * _tmp8;
  const Scalar _tmp37 = _tmp0 + _tmp2;
  const Scalar _tmp38 = _target_pose[2] * _tmp1;
  const Scalar _tmp39 = _target_pose[0] * _tmp5;
  const Scalar _tmp40 = _tmp38 - _tmp39;
  const Scalar _tmp41 = 1 - 2 * std::pow(_target_pose[0], Scalar(2));
  const Scalar _tmp42 = _tmp11 + _tmp41;
  const Scalar _tmp43 = _tmp13 * _tmp37 + _tmp4 * _tmp42 + _tmp40 * _tmp9;
  const Scalar _tmp44 =
      _tmp30 * _tmp37 + _tmp34 * _tmp40 + _tmp35 * _tmp42 + _tmp43 * source_inverse_range;
  const Scalar _tmp45 = std::fabs(_tmp44);
  const Scalar _tmp46 = std::max<Scalar>(_tmp45, epsilon);
  const Scalar _tmp47 = Scalar(1.0) / (_tmp46);
  const Scalar _tmp48 = _tmp47 * target_calibration_storage(0, 0);
  const Scalar _tmp49 = _tmp36 * _tmp48 + target_calibration_storage(2, 0) - target_pixel(0, 0);
  const Scalar _tmp50 = _tmp38 + _tmp39;
  const Scalar _tmp51 = -_tmp6 + _tmp7;
  const Scalar _tmp52 = _tmp10 + _tmp41;
  const Scalar _tmp53 = _tmp13 * _tmp51 + _tmp4 * _tmp50 + _tmp52 * _tmp9;
  const Scalar _tmp54 =
      _tmp30 * _tmp51 + _tmp34 * _tmp52 + _tmp35 * _tmp50 + _tmp53 * source_inverse_range;
  const Scalar _tmp55 = _tmp47 * target_calibration_storage(1, 0);
  const Scalar _tmp56 = _tmp54 * _tmp55 + target_calibration_storage(3, 0) - target_pixel(1, 0);
  const Scalar _tmp57 = std::pow(_tmp49, Scalar(2)) + std::pow(_tmp56, Scalar(2)) + epsilon;
  const Scalar _tmp58 = std::pow(_tmp57, Scalar(Scalar(-1) / Scalar(2)));
  const Scalar _tmp59 = (((_tmp44) > 0) - ((_tmp44) < 0));
  const Scalar _tmp60 = std::sqrt(weight) * std::sqrt(std::max<Scalar>(0, _tmp59));
  const Scalar _tmp61 = Scalar(1.0) / (epsilon - gnc_mu + 1);
  const Scalar _tmp62 = epsilon + std::fabs(_tmp61);
  const Scalar _tmp63 = std::pow(gnc_scale, Scalar(-2));
  const Scalar _tmp64 = _tmp57 * _tmp63 / _tmp62 + 1;
  const Scalar _tmp65 = 2 - _tmp61;
  const Scalar _tmp66 =
      _tmp65 + epsilon * (2 * std::min<Scalar>(0, (((_tmp65) > 0) - ((_tmp65) < 0))) + 1);
  const Scalar _tmp67 = (Scalar(1) / Scalar(2)) * _tmp66;
  const Scalar _tmp68 = std::sqrt(2) * std::sqrt(_tmp62 * (std::pow(_tmp64, _tmp67) - 1) / _tmp66);
  const Scalar _tmp69 = _tmp60 * _tmp68;
  const Scalar _tmp70 = _tmp58 * _tmp69;
  const Scalar _tmp71 = _tmp49 * _tmp70;
  const Scalar _tmp72 = _tmp56 * _tmp70;
  const Scalar _tmp73 = _tmp15 * _tmp21;
  const Scalar _tmp74 = _tmp24 * _tmp73;
  const Scalar _tmp75 = 2 * _tmp21;
  const Scalar _tmp76 = _source_pose[0] * _tmp75;
  const Scalar _tmp77 = _tmp28 * _tmp76;
  const Scalar _tmp78 = -_tmp74 + _tmp77;
  const Scalar _tmp79 = _tmp17 * _tmp21;
  const Scalar _tmp80 = _tmp24 * _tmp79;
  const Scalar _tmp81 = -_tmp76 + _tmp80;
  const Scalar _tmp82 = _tmp28 * _tmp79;
  const Scalar _tmp83 = _tmp73 - _tmp82;
  const Scalar _tmp84 = _tmp54 * target_calibration_storage(1, 0);
  const Scalar _tmp85 = (Scalar(1) / Scalar(2)) * _tmp59 *
                        ((((_tmp45 - epsilon) > 0) - ((_tmp45 - epsilon) < 0)) + 1) /
                        std::pow(_tmp46, Scalar(2));
  const Scalar _tmp86 = _tmp85 * (_tmp37 * _tmp83 + _tmp40 * _tmp81 + _tmp42 * _tmp78);
  const Scalar _tmp87 =
      _tmp55 * (_tmp50 * _tmp78 + _tmp51 * _tmp83 + _tmp52 * _tmp81) - _tmp84 * _tmp86;
  const Scalar _tmp88 = 2 * _tmp56;
  const Scalar _tmp89 = _tmp36 * target_calibration_storage(0, 0);
  const Scalar _tmp90 =
      _tmp48 * (_tmp12 * _tmp83 + _tmp3 * _tmp78 + _tmp8 * _tmp81) - _tmp86 * _tmp89;
  const Scalar _tmp91 = 2 * _tmp49;
  const Scalar _tmp92 =
      (Scalar(1) / Scalar(2)) * _tmp87 * _tmp88 + (Scalar(1) / Scalar(2)) * _tmp90 * _tmp91;
  const Scalar _tmp93 = _tmp49 * _tmp92;
  const Scalar _tmp94 = _tmp58 * _tmp60 * _tmp63 * std::pow(_tmp64, Scalar(_tmp67 - 1)) / _tmp68;
  const Scalar _tmp95 = _tmp69 / (_tmp57 * std::sqrt(_tmp57));
  const Scalar _tmp96 = _tmp70 * _tmp90 + _tmp93 * _tmp94 - _tmp93 * _tmp95;
  const Scalar _tmp97 = (Scalar(1) / Scalar(2)) * _tmp96;
  const Scalar _tmp98 = _tmp28 * _tmp73;
  const Scalar _tmp99 = _tmp79 + _tmp98;
  const Scalar _tmp100 = _source_pose[3] * _tmp75;
  const Scalar _tmp101 = _tmp100 * _tmp28;
  const Scalar _tmp102 = 4 * _tmp21;
  const Scalar _tmp103 = _source_pose[0] * _tmp102;
  const Scalar _tmp104 = _tmp101 - _tmp103 + _tmp80;
  const Scalar _tmp105 = -_tmp100 - _tmp103 * _tmp28 + _tmp74;
  const Scalar _tmp106 = _tmp104 * _tmp42 + _tmp105 * _tmp40 + _tmp37 * _tmp99;
  const Scalar _tmp107 = _tmp85 * _tmp89;
  const Scalar _tmp108 =
      -_tmp106 * _tmp107 + _tmp48 * (_tmp104 * _tmp3 + _tmp105 * _tmp8 + _tmp12 * _tmp99);
  const Scalar _tmp109 = _tmp84 * _tmp85;
  const Scalar _tmp110 =
      -_tmp106 * _tmp109 + _tmp55 * (_tmp104 * _tmp50 + _tmp105 * _tmp52 + _tmp51 * _tmp99);
  const Scalar _tmp111 = _tmp108 * _tmp91 + _tmp110 * _tmp88;
  const Scalar _tmp112 = (Scalar(1) / Scalar(2)) * _tmp95;
  const Scalar _tmp113 = _tmp112 * _tmp49;
  const Scalar _tmp114 = (Scalar(1) / Scalar(2)) * _tmp94;
  const Scalar _tmp115 = _tmp114 * _tmp49;
  const Scalar _tmp116 = _tmp108 * _tmp70 - _tmp111 * _tmp113 + _tmp111 * _tmp115;
  const Scalar _tmp117 = (Scalar(1) / Scalar(2)) * _tmp116;
  const Scalar _tmp118 = _tmp24 * _tmp76;
  const Scalar _tmp119 = _tmp118 + _tmp79;
  const Scalar _tmp120 = _tmp100 * _tmp24;
  const Scalar _tmp121 = _source_pose[1] * _tmp102;
  const Scalar _tmp122 = -_tmp120 - _tmp121 + _tmp82;
  const Scalar _tmp123 = _tmp100 - _tmp121 * _tmp24 + _tmp77;
  const Scalar _tmp124 = _tmp119 * _tmp40 + _tmp122 * _tmp42 + _tmp123 * _tmp37;
  const Scalar _tmp125 =
      -_tmp107 * _tmp124 + _tmp48 * (_tmp119 * _tmp8 + _tmp12 * _tmp123 + _tmp122 * _tmp3);
  const Scalar _tmp126 =
      -_tmp109 * _tmp124 + _tmp55 * (_tmp119 * _tmp52 + _tmp122 * _tmp50 + _tmp123 * _tmp51);
  const Scalar _tmp127 = _tmp125 * _tmp91 + _tmp126 * _tmp88;
  const Scalar _tmp128 = -Scalar(1) / Scalar(2) * _tmp113 * _tmp127 +
                         (Scalar(1) / Scalar(2)) * _tmp115 * _tmp127 +
                         (Scalar(1) / Scalar(2)) * _tmp125 * _tmp70;
  const Scalar _tmp129 = _tmp118 + _tmp98;
  const Scalar _tmp130 = _source_pose[2] * _tmp102;
  const Scalar _tmp131 = -_tmp101 - _tmp130 * _tmp24 + _tmp76;
  const Scalar _tmp132 = _tmp120 - _tmp130 * _tmp28 + _tmp73;
  const Scalar _tmp133 = _tmp129 * _tmp42 + _tmp131 * _tmp37 + _tmp132 * _tmp40;
  const Scalar _tmp134 =
      -_tmp107 * _tmp133 + _tmp48 * (_tmp12 * _tmp131 + _tmp129 * _tmp3 + _tmp132 * _tmp8);
  const Scalar _tmp135 =
      -_tmp109 * _tmp133 + _tmp55 * (_tmp129 * _tmp50 + _tmp131 * _tmp51 + _tmp132 * _tmp52);
  const Scalar _tmp136 = _tmp134 * _tmp91 + _tmp135 * _tmp88;
  const Scalar _tmp137 = -_tmp113 * _tmp136 + _tmp115 * _tmp136 + _tmp134 * _tmp70;
  const Scalar _tmp138 = (Scalar(1) / Scalar(2)) * _source_pose[1];
  const Scalar _tmp139 = -_source_pose[0] * _tmp97 + _source_pose[2] * _tmp128 +
                         _source_pose[3] * _tmp117 - _tmp137 * _tmp138;
  const Scalar _tmp140 = (Scalar(1) / Scalar(2)) * _tmp137;
  const Scalar _tmp141 = _source_pose[0] * _tmp140 - _source_pose[2] * _tmp117 +
                         _source_pose[3] * _tmp128 - _tmp138 * _tmp96;
  const Scalar _tmp142 = -_source_pose[0] * _tmp128 - _source_pose[2] * _tmp97 +
                         _source_pose[3] * _tmp140 + _tmp116 * _tmp138;
  const Scalar _tmp143 = _tmp48 * source_inverse_range;
  const Scalar _tmp144 = _tmp12 * _tmp143;
  const Scalar _tmp145 = _tmp107 * source_inverse_range;
  const Scalar _tmp146 = _tmp145 * _tmp37;
  const Scalar _tmp147 = _tmp144 - _tmp146;
  const Scalar _tmp148 = _tmp55 * source_inverse_range;
  const Scalar _tmp149 = _tmp148 * _tmp51;
  const Scalar _tmp150 = _tmp109 * source_inverse_range;
  const Scalar _tmp151 = _tmp150 * _tmp37;
  const Scalar _tmp152 = _tmp149 - _tmp151;
  const Scalar _tmp153 = _tmp147 * _tmp91 + _tmp152 * _tmp88;
  const Scalar _tmp154 = -_tmp113 * _tmp153 + _tmp115 * _tmp153 + _tmp147 * _tmp70;
  const Scalar _tmp155 = _tmp145 * _tmp40;
  const Scalar _tmp156 = _tmp143 * _tmp8;
  const Scalar _tmp157 = -_tmp155 + _tmp156;
  const Scalar _tmp158 = _tmp148 * _tmp52;
  const Scalar _tmp159 = _tmp150 * _tmp40;
  const Scalar _tmp160 = _tmp158 - _tmp159;
  const Scalar _tmp161 = _tmp157 * _tmp91 + _tmp160 * _tmp88;
  const Scalar _tmp162 = _tmp112 * _tmp161;
  const Scalar _tmp163 = _tmp115 * _tmp161 + _tmp157 * _tmp70 - _tmp162 * _tmp49;
  const Scalar _tmp164 = _tmp143 * _tmp3;
  const Scalar _tmp165 = _tmp145 * _tmp42;
  const Scalar _tmp166 = _tmp164 - _tmp165;
  const Scalar _tmp167 = _tmp150 * _tmp42;
  const Scalar _tmp168 = _tmp148 * _tmp50;
  const Scalar _tmp169 = -_tmp167 + _tmp168;
  const Scalar _tmp170 = _tmp166 * _tmp91 + _tmp169 * _tmp88;
  const Scalar _tmp171 = -_tmp113 * _tmp170 + _tmp115 * _tmp170 + _tmp166 * _tmp70;
  const Scalar _tmp172 = 2 * _tmp13;
  const Scalar _tmp173 = _target_pose[0] * _tmp172;
  const Scalar _tmp174 = 2 * _tmp9;
  const Scalar _tmp175 = _target_pose[1] * _tmp174;
  const Scalar _tmp176 = 2 * _tmp30;
  const Scalar _tmp177 = _target_pose[0] * _tmp176;
  const Scalar _tmp178 = 2 * _tmp34;
  const Scalar _tmp179 = _target_pose[1] * _tmp178;
  const Scalar _tmp180 = _tmp177 + _tmp179 + source_inverse_range * (_tmp173 + _tmp175);
  const Scalar _tmp181 = _target_pose[3] * _tmp174;
  const Scalar _tmp182 = 4 * _target_pose[2];
  const Scalar _tmp183 = 2 * _tmp4;
  const Scalar _tmp184 = _target_pose[0] * _tmp183;
  const Scalar _tmp185 = _target_pose[3] * _tmp178;
  const Scalar _tmp186 = 2 * _tmp35;
  const Scalar _tmp187 = _target_pose[0] * _tmp186;
  const Scalar _tmp188 = -_tmp107 * _tmp180 +
                         _tmp48 * (-_tmp182 * _tmp30 + _tmp185 + _tmp187 +
                                   source_inverse_range * (-_tmp13 * _tmp182 + _tmp181 + _tmp184));
  const Scalar _tmp189 = _target_pose[3] * _tmp172;
  const Scalar _tmp190 = _target_pose[1] * _tmp183;
  const Scalar _tmp191 = _target_pose[3] * _tmp176;
  const Scalar _tmp192 = _tmp1 * _tmp35;
  const Scalar _tmp193 =
      -_tmp109 * _tmp180 + _tmp55 * (-_tmp182 * _tmp34 - _tmp191 + _tmp192 +
                                     source_inverse_range * (-_tmp182 * _tmp9 - _tmp189 + _tmp190));
  const Scalar _tmp194 = _tmp188 * _tmp91 + _tmp193 * _tmp88;
  const Scalar _tmp195 = -Scalar(1) / Scalar(2) * _tmp113 * _tmp194 +
                         (Scalar(1) / Scalar(2)) * _tmp115 * _tmp194 +
                         (Scalar(1) / Scalar(2)) * _tmp188 * _tmp70;
  const Scalar _tmp196 = _target_pose[2] * _tmp174;
  const Scalar _tmp197 = 4 * _target_pose[1];
  const Scalar _tmp198 = _target_pose[2] * _tmp178;
  const Scalar _tmp199 = _tmp85 * (_tmp191 - _tmp197 * _tmp35 + _tmp198 +
                                   source_inverse_range * (_tmp189 + _tmp196 - _tmp197 * _tmp4));
  const Scalar _tmp200 = _target_pose[3] * _tmp183;
  const Scalar _tmp201 = _target_pose[0] * _tmp174;
  const Scalar _tmp202 = _target_pose[0] * _tmp178;
  const Scalar _tmp203 = _target_pose[3] * _tmp186;
  const Scalar _tmp204 =
      -_tmp199 * _tmp89 + _tmp48 * (-_tmp197 * _tmp30 + _tmp202 - _tmp203 +
                                    source_inverse_range * (-_tmp13 * _tmp197 - _tmp200 + _tmp201));
  const Scalar _tmp205 = _target_pose[2] * _tmp183;
  const Scalar _tmp206 = _target_pose[2] * _tmp186;
  const Scalar _tmp207 =
      -_tmp199 * _tmp84 + _tmp55 * (_tmp177 + _tmp206 + source_inverse_range * (_tmp173 + _tmp205));
  const Scalar _tmp208 = _tmp204 * _tmp91 + _tmp207 * _tmp88;
  const Scalar _tmp209 = -_tmp113 * _tmp208 + _tmp115 * _tmp208 + _tmp204 * _tmp70;
  const Scalar _tmp210 = (Scalar(1) / Scalar(2)) * _target_pose[2];
  const Scalar _tmp211 = _target_pose[1] * _tmp172;
  const Scalar _tmp212 = _tmp1 * _tmp30;
  const Scalar _tmp213 = -_tmp202 + _tmp212 + source_inverse_range * (-_tmp201 + _tmp211);
  const Scalar _tmp214 =
      -_tmp107 * _tmp213 +
      _tmp48 * (-_tmp192 + _tmp198 + source_inverse_range * (-_tmp190 + _tmp196));
  const Scalar _tmp215 = _target_pose[2] * _tmp172;
  const Scalar _tmp216 = _target_pose[2] * _tmp176;
  const Scalar _tmp217 = -_tmp109 * _tmp213 +
                         _tmp55 * (_tmp187 - _tmp216 + source_inverse_range * (_tmp184 - _tmp215));
  const Scalar _tmp218 = _tmp214 * _tmp91 + _tmp217 * _tmp88;
  const Scalar _tmp219 = -_tmp113 * _tmp218 + _tmp115 * _tmp218 + _tmp214 * _tmp70;
  const Scalar _tmp220 = (Scalar(1) / Scalar(2)) * _tmp219;
  const Scalar _tmp221 = 4 * _target_pose[0];
  const Scalar _tmp222 = -_tmp185 + _tmp216 - _tmp221 * _tmp35 +
                         source_inverse_range * (-_tmp181 + _tmp215 - _tmp221 * _tmp4);
  const Scalar _tmp223 = -_tmp107 * _tmp222 +
                         _tmp48 * (_tmp179 + _tmp206 + source_inverse_range * (_tmp175 + _tmp205));
  const Scalar _tmp224 =
      -_tmp109 * _tmp222 + _tmp55 * (_tmp203 + _tmp212 - _tmp221 * _tmp34 +
                                     source_inverse_range * (_tmp200 + _tmp211 - _tmp221 * _tmp9));
  const Scalar _tmp225 = _tmp223 * _tmp91 + _tmp224 * _tmp88;
  const Scalar _tmp226 = -_tmp113 * _tmp225 + _tmp115 * _tmp225 + _tmp223 * _tmp70;
  const Scalar _tmp227 = (Scalar(1) / Scalar(2)) * _tmp226;
  const Scalar _tmp228 = -_target_pose[0] * _tmp220 - _target_pose[1] * _tmp195 +
                         _target_pose[3] * _tmp227 + _tmp209 * _tmp210;
  const Scalar _tmp229 = (Scalar(1) / Scalar(2)) * _tmp209;
  const Scalar _tmp230 = _target_pose[0] * _tmp195 - _target_pose[1] * _tmp220 +
                         _target_pose[3] * _tmp229 - _tmp210 * _tmp226;
  const Scalar _tmp231 = -_target_pose[0] * _tmp229 + _target_pose[1] * _tmp227 +
                         _target_pose[3] * _tmp195 - _tmp210 * _tmp219;
  const Scalar _tmp232 = -_tmp144 + _tmp146;
  const Scalar _tmp233 = -_tmp149 + _tmp151;
  const Scalar _tmp234 = _tmp232 * _tmp91 + _tmp233 * _tmp88;
  const Scalar _tmp235 = -_tmp113 * _tmp234 + _tmp115 * _tmp234 + _tmp232 * _tmp70;
  const Scalar _tmp236 = _tmp155 - _tmp156;
  const Scalar _tmp237 = -_tmp158 + _tmp159;
  const Scalar _tmp238 = _tmp236 * _tmp91 + _tmp237 * _tmp88;
  const Scalar _tmp239 = -_tmp113 * _tmp238 + _tmp115 * _tmp238 + _tmp236 * _tmp70;
  const Scalar _tmp240 = -_tmp164 + _tmp165;
  const Scalar _tmp241 = _tmp167 - _tmp168;
  const Scalar _tmp242 = _tmp240 * _tmp91 + _tmp241 * _tmp88;
  const Scalar _tmp243 = -_tmp113 * _tmp242 + _tmp115 * _tmp242 + _tmp240 * _tmp70;
  const Scalar _tmp244 = -_tmp109 * _tmp43 + _tmp53 * _tmp55;
  const Scalar _tmp245 = -_tmp107 * _tmp43 + _tmp14 * _tmp48;
  const Scalar _tmp246 = _tmp244 * _tmp88 + _tmp245 * _tmp91;
  const Scalar _tmp247 = -_tmp113 * _tmp246 + _tmp115 * _tmp246 + _tmp245 * _tmp70;
  const Scalar _tmp248 = _tmp56 * _tmp92;
  const Scalar _tmp249 = _tmp248 * _tmp94 - _tmp248 * _tmp95 + _tmp70 * _tmp87;
  const Scalar _tmp250 = (Scalar(1) / Scalar(2)) * _tmp249;
  const Scalar _tmp251 = _tmp114 * _tmp56;
  const Scalar _tmp252 = _tmp112 * _tmp56;
  const Scalar _tmp253 = (Scalar(1) / Scalar(2)) * _tmp126 * _tmp70 +
                         (Scalar(1) / Scalar(2)) * _tmp127 * _tmp251 -
                         Scalar(1) / Scalar(2) * _tmp127 * _tmp252;
  const Scalar _tmp254 = _tmp135 * _tmp70 + _tmp136 * _tmp251 - _tmp136 * _tmp252;
  const Scalar _tmp255 = _tmp110 * _tmp70 + _tmp111 * _tmp251 - _tmp111 * _tmp252;
  const Scalar _tmp256 = (Scalar(1) / Scalar(2)) * _tmp255;
  const Scalar _tmp257 = -_source_pose[0] * _tmp250 + _source_pose[2] * _tmp253 +
                         _source_pose[3] * _tmp256 - _tmp138 * _tmp254;
  const Scalar _tmp258 = (Scalar(1) / Scalar(2)) * _tmp254;
  const Scalar _tmp259 = _source_pose[0] * _tmp258 - _source_pose[2] * _tmp256 +
                         _source_pose[3] * _tmp253 - _tmp138 * _tmp249;
  const Scalar _tmp260 = -_source_pose[0] * _tmp253 - _source_pose[2] * _tmp250 +
                         _source_pose[3] * _tmp258 + _tmp138 * _tmp255;
  const Scalar _tmp261 = _tmp152 * _tmp70 + _tmp153 * _tmp251 - _tmp153 * _tmp252;
  const Scalar _tmp262 = _tmp160 * _tmp70 + _tmp161 * _tmp251 - _tmp162 * _tmp56;
  const Scalar _tmp263 = _tmp169 * _tmp70 + _tmp170 * _tmp251 - _tmp170 * _tmp252;
  const Scalar _tmp264 = (Scalar(1) / Scalar(2)) * _tmp217 * _tmp70 +
                         (Scalar(1) / Scalar(2)) * _tmp218 * _tmp251 -
                         Scalar(1) / Scalar(2) * _tmp218 * _tmp252;
  const Scalar _tmp265 = (Scalar(1) / Scalar(2)) * _tmp193 * _tmp70 +
                         (Scalar(1) / Scalar(2)) * _tmp194 * _tmp251 -
                         Scalar(1) / Scalar(2) * _tmp194 * _tmp252;
  const Scalar _tmp266 = _tmp224 * _tmp70 + _tmp225 * _tmp251 - _tmp225 * _tmp252;
  const Scalar _tmp267 = (Scalar(1) / Scalar(2)) * _tmp266;
  const Scalar _tmp268 = (Scalar(1) / Scalar(2)) * _tmp207 * _tmp70 +
                         (Scalar(1) / Scalar(2)) * _tmp208 * _tmp251 -
                         Scalar(1) / Scalar(2) * _tmp208 * _tmp252;
  const Scalar _tmp269 = -_target_pose[0] * _tmp264 - _target_pose[1] * _tmp265 +
                         _target_pose[2] * _tmp268 + _target_pose[3] * _tmp267;
  const Scalar _tmp270 = _target_pose[0] * _tmp265 - _target_pose[1] * _tmp264 +
                         _target_pose[3] * _tmp268 - _tmp210 * _tmp266;
  const Scalar _tmp271 = -_target_pose[0] * _tmp268 + _target_pose[1] * _tmp267 -
                         _target_pose[2] * _tmp264 + _target_pose[3] * _tmp265;
  const Scalar _tmp272 = _tmp233 * _tmp70 + _tmp234 * _tmp251 - _tmp234 * _tmp252;
  const Scalar _tmp273 = _tmp237 * _tmp70 + _tmp238 * _tmp251 - _tmp238 * _tmp252;
  const Scalar _tmp274 = _tmp241 * _tmp70 + _tmp242 * _tmp251 - _tmp242 * _tmp252;
  const Scalar _tmp275 = _tmp244 * _tmp70 + _tmp246 * _tmp251 - _tmp246 * _tmp252;

  // Output terms (4)
  if (res != nullptr) {
    Eigen::Matrix<Scalar, 2, 1>& _res = (*res);

    _res(0, 0) = _tmp71;
    _res(1, 0) = _tmp72;
  }

  if (jacobian != nullptr) {
    Eigen::Matrix<Scalar, 2, 13>& _jacobian = (*jacobian);

    _jacobian(0, 0) = _tmp139;
    _jacobian(0, 1) = _tmp141;
    _jacobian(0, 2) = _tmp142;
    _jacobian(0, 3) = _tmp154;
    _jacobian(0, 4) = _tmp163;
    _jacobian(0, 5) = _tmp171;
    _jacobian(0, 6) = _tmp228;
    _jacobian(0, 7) = _tmp230;
    _jacobian(0, 8) = _tmp231;
    _jacobian(0, 9) = _tmp235;
    _jacobian(0, 10) = _tmp239;
    _jacobian(0, 11) = _tmp243;
    _jacobian(0, 12) = _tmp247;
    _jacobian(1, 0) = _tmp257;
    _jacobian(1, 1) = _tmp259;
    _jacobian(1, 2) = _tmp260;
    _jacobian(1, 3) = _tmp261;
    _jacobian(1, 4) = _tmp262;
    _jacobian(1, 5) = _tmp263;
    _jacobian(1, 6) = _tmp269;
    _jacobian(1, 7) = _tmp270;
    _jacobian(1, 8) = _tmp271;
    _jacobian(1, 9) = _tmp272;
    _jacobian(1, 10) = _tmp273;
    _jacobian(1, 11) = _tmp274;
    _jacobian(1, 12) = _tmp275;
  }

  if (hessian != nullptr) {
    Eigen::Matrix<Scalar, 13, 13>& _hessian = (*hessian);

    _hessian(0, 0) = std::pow(_tmp139, Scalar(2)) + std::pow(_tmp257, Scalar(2));
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
    _hessian(1, 0) = _tmp139 * _tmp141 + _tmp257 * _tmp259;
    _hessian(1, 1) = std::pow(_tmp141, Scalar(2)) + std::pow(_tmp259, Scalar(2));
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
    _hessian(2, 0) = _tmp139 * _tmp142 + _tmp257 * _tmp260;
    _hessian(2, 1) = _tmp141 * _tmp142 + _tmp259 * _tmp260;
    _hessian(2, 2) = std::pow(_tmp142, Scalar(2)) + std::pow(_tmp260, Scalar(2));
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
    _hessian(3, 0) = _tmp139 * _tmp154 + _tmp257 * _tmp261;
    _hessian(3, 1) = _tmp141 * _tmp154 + _tmp259 * _tmp261;
    _hessian(3, 2) = _tmp142 * _tmp154 + _tmp260 * _tmp261;
    _hessian(3, 3) = std::pow(_tmp154, Scalar(2)) + std::pow(_tmp261, Scalar(2));
    _hessian(3, 4) = 0;
    _hessian(3, 5) = 0;
    _hessian(3, 6) = 0;
    _hessian(3, 7) = 0;
    _hessian(3, 8) = 0;
    _hessian(3, 9) = 0;
    _hessian(3, 10) = 0;
    _hessian(3, 11) = 0;
    _hessian(3, 12) = 0;
    _hessian(4, 0) = _tmp139 * _tmp163 + _tmp257 * _tmp262;
    _hessian(4, 1) = _tmp141 * _tmp163 + _tmp259 * _tmp262;
    _hessian(4, 2) = _tmp142 * _tmp163 + _tmp260 * _tmp262;
    _hessian(4, 3) = _tmp154 * _tmp163 + _tmp261 * _tmp262;
    _hessian(4, 4) = std::pow(_tmp163, Scalar(2)) + std::pow(_tmp262, Scalar(2));
    _hessian(4, 5) = 0;
    _hessian(4, 6) = 0;
    _hessian(4, 7) = 0;
    _hessian(4, 8) = 0;
    _hessian(4, 9) = 0;
    _hessian(4, 10) = 0;
    _hessian(4, 11) = 0;
    _hessian(4, 12) = 0;
    _hessian(5, 0) = _tmp139 * _tmp171 + _tmp257 * _tmp263;
    _hessian(5, 1) = _tmp141 * _tmp171 + _tmp259 * _tmp263;
    _hessian(5, 2) = _tmp142 * _tmp171 + _tmp260 * _tmp263;
    _hessian(5, 3) = _tmp154 * _tmp171 + _tmp261 * _tmp263;
    _hessian(5, 4) = _tmp163 * _tmp171 + _tmp262 * _tmp263;
    _hessian(5, 5) = std::pow(_tmp171, Scalar(2)) + std::pow(_tmp263, Scalar(2));
    _hessian(5, 6) = 0;
    _hessian(5, 7) = 0;
    _hessian(5, 8) = 0;
    _hessian(5, 9) = 0;
    _hessian(5, 10) = 0;
    _hessian(5, 11) = 0;
    _hessian(5, 12) = 0;
    _hessian(6, 0) = _tmp139 * _tmp228 + _tmp257 * _tmp269;
    _hessian(6, 1) = _tmp141 * _tmp228 + _tmp259 * _tmp269;
    _hessian(6, 2) = _tmp142 * _tmp228 + _tmp260 * _tmp269;
    _hessian(6, 3) = _tmp154 * _tmp228 + _tmp261 * _tmp269;
    _hessian(6, 4) = _tmp163 * _tmp228 + _tmp262 * _tmp269;
    _hessian(6, 5) = _tmp171 * _tmp228 + _tmp263 * _tmp269;
    _hessian(6, 6) = std::pow(_tmp228, Scalar(2)) + std::pow(_tmp269, Scalar(2));
    _hessian(6, 7) = 0;
    _hessian(6, 8) = 0;
    _hessian(6, 9) = 0;
    _hessian(6, 10) = 0;
    _hessian(6, 11) = 0;
    _hessian(6, 12) = 0;
    _hessian(7, 0) = _tmp139 * _tmp230 + _tmp257 * _tmp270;
    _hessian(7, 1) = _tmp141 * _tmp230 + _tmp259 * _tmp270;
    _hessian(7, 2) = _tmp142 * _tmp230 + _tmp260 * _tmp270;
    _hessian(7, 3) = _tmp154 * _tmp230 + _tmp261 * _tmp270;
    _hessian(7, 4) = _tmp163 * _tmp230 + _tmp262 * _tmp270;
    _hessian(7, 5) = _tmp171 * _tmp230 + _tmp263 * _tmp270;
    _hessian(7, 6) = _tmp228 * _tmp230 + _tmp269 * _tmp270;
    _hessian(7, 7) = std::pow(_tmp230, Scalar(2)) + std::pow(_tmp270, Scalar(2));
    _hessian(7, 8) = 0;
    _hessian(7, 9) = 0;
    _hessian(7, 10) = 0;
    _hessian(7, 11) = 0;
    _hessian(7, 12) = 0;
    _hessian(8, 0) = _tmp139 * _tmp231 + _tmp257 * _tmp271;
    _hessian(8, 1) = _tmp141 * _tmp231 + _tmp259 * _tmp271;
    _hessian(8, 2) = _tmp142 * _tmp231 + _tmp260 * _tmp271;
    _hessian(8, 3) = _tmp154 * _tmp231 + _tmp261 * _tmp271;
    _hessian(8, 4) = _tmp163 * _tmp231 + _tmp262 * _tmp271;
    _hessian(8, 5) = _tmp171 * _tmp231 + _tmp263 * _tmp271;
    _hessian(8, 6) = _tmp228 * _tmp231 + _tmp269 * _tmp271;
    _hessian(8, 7) = _tmp230 * _tmp231 + _tmp270 * _tmp271;
    _hessian(8, 8) = std::pow(_tmp231, Scalar(2)) + std::pow(_tmp271, Scalar(2));
    _hessian(8, 9) = 0;
    _hessian(8, 10) = 0;
    _hessian(8, 11) = 0;
    _hessian(8, 12) = 0;
    _hessian(9, 0) = _tmp139 * _tmp235 + _tmp257 * _tmp272;
    _hessian(9, 1) = _tmp141 * _tmp235 + _tmp259 * _tmp272;
    _hessian(9, 2) = _tmp142 * _tmp235 + _tmp260 * _tmp272;
    _hessian(9, 3) = _tmp154 * _tmp235 + _tmp261 * _tmp272;
    _hessian(9, 4) = _tmp163 * _tmp235 + _tmp262 * _tmp272;
    _hessian(9, 5) = _tmp171 * _tmp235 + _tmp263 * _tmp272;
    _hessian(9, 6) = _tmp228 * _tmp235 + _tmp269 * _tmp272;
    _hessian(9, 7) = _tmp230 * _tmp235 + _tmp270 * _tmp272;
    _hessian(9, 8) = _tmp231 * _tmp235 + _tmp271 * _tmp272;
    _hessian(9, 9) = std::pow(_tmp235, Scalar(2)) + std::pow(_tmp272, Scalar(2));
    _hessian(9, 10) = 0;
    _hessian(9, 11) = 0;
    _hessian(9, 12) = 0;
    _hessian(10, 0) = _tmp139 * _tmp239 + _tmp257 * _tmp273;
    _hessian(10, 1) = _tmp141 * _tmp239 + _tmp259 * _tmp273;
    _hessian(10, 2) = _tmp142 * _tmp239 + _tmp260 * _tmp273;
    _hessian(10, 3) = _tmp154 * _tmp239 + _tmp261 * _tmp273;
    _hessian(10, 4) = _tmp163 * _tmp239 + _tmp262 * _tmp273;
    _hessian(10, 5) = _tmp171 * _tmp239 + _tmp263 * _tmp273;
    _hessian(10, 6) = _tmp228 * _tmp239 + _tmp269 * _tmp273;
    _hessian(10, 7) = _tmp230 * _tmp239 + _tmp270 * _tmp273;
    _hessian(10, 8) = _tmp231 * _tmp239 + _tmp271 * _tmp273;
    _hessian(10, 9) = _tmp235 * _tmp239 + _tmp272 * _tmp273;
    _hessian(10, 10) = std::pow(_tmp239, Scalar(2)) + std::pow(_tmp273, Scalar(2));
    _hessian(10, 11) = 0;
    _hessian(10, 12) = 0;
    _hessian(11, 0) = _tmp139 * _tmp243 + _tmp257 * _tmp274;
    _hessian(11, 1) = _tmp141 * _tmp243 + _tmp259 * _tmp274;
    _hessian(11, 2) = _tmp142 * _tmp243 + _tmp260 * _tmp274;
    _hessian(11, 3) = _tmp154 * _tmp243 + _tmp261 * _tmp274;
    _hessian(11, 4) = _tmp163 * _tmp243 + _tmp262 * _tmp274;
    _hessian(11, 5) = _tmp171 * _tmp243 + _tmp263 * _tmp274;
    _hessian(11, 6) = _tmp228 * _tmp243 + _tmp269 * _tmp274;
    _hessian(11, 7) = _tmp230 * _tmp243 + _tmp270 * _tmp274;
    _hessian(11, 8) = _tmp231 * _tmp243 + _tmp271 * _tmp274;
    _hessian(11, 9) = _tmp235 * _tmp243 + _tmp272 * _tmp274;
    _hessian(11, 10) = _tmp239 * _tmp243 + _tmp273 * _tmp274;
    _hessian(11, 11) = std::pow(_tmp243, Scalar(2)) + std::pow(_tmp274, Scalar(2));
    _hessian(11, 12) = 0;
    _hessian(12, 0) = _tmp139 * _tmp247 + _tmp257 * _tmp275;
    _hessian(12, 1) = _tmp141 * _tmp247 + _tmp259 * _tmp275;
    _hessian(12, 2) = _tmp142 * _tmp247 + _tmp260 * _tmp275;
    _hessian(12, 3) = _tmp154 * _tmp247 + _tmp261 * _tmp275;
    _hessian(12, 4) = _tmp163 * _tmp247 + _tmp262 * _tmp275;
    _hessian(12, 5) = _tmp171 * _tmp247 + _tmp263 * _tmp275;
    _hessian(12, 6) = _tmp228 * _tmp247 + _tmp269 * _tmp275;
    _hessian(12, 7) = _tmp230 * _tmp247 + _tmp270 * _tmp275;
    _hessian(12, 8) = _tmp231 * _tmp247 + _tmp271 * _tmp275;
    _hessian(12, 9) = _tmp235 * _tmp247 + _tmp272 * _tmp275;
    _hessian(12, 10) = _tmp239 * _tmp247 + _tmp273 * _tmp275;
    _hessian(12, 11) = _tmp243 * _tmp247 + _tmp274 * _tmp275;
    _hessian(12, 12) = std::pow(_tmp247, Scalar(2)) + std::pow(_tmp275, Scalar(2));
  }

  if (rhs != nullptr) {
    Eigen::Matrix<Scalar, 13, 1>& _rhs = (*rhs);

    _rhs(0, 0) = _tmp139 * _tmp71 + _tmp257 * _tmp72;
    _rhs(1, 0) = _tmp141 * _tmp71 + _tmp259 * _tmp72;
    _rhs(2, 0) = _tmp142 * _tmp71 + _tmp260 * _tmp72;
    _rhs(3, 0) = _tmp154 * _tmp71 + _tmp261 * _tmp72;
    _rhs(4, 0) = _tmp163 * _tmp71 + _tmp262 * _tmp72;
    _rhs(5, 0) = _tmp171 * _tmp71 + _tmp263 * _tmp72;
    _rhs(6, 0) = _tmp228 * _tmp71 + _tmp269 * _tmp72;
    _rhs(7, 0) = _tmp230 * _tmp71 + _tmp270 * _tmp72;
    _rhs(8, 0) = _tmp231 * _tmp71 + _tmp271 * _tmp72;
    _rhs(9, 0) = _tmp235 * _tmp71 + _tmp272 * _tmp72;
    _rhs(10, 0) = _tmp239 * _tmp71 + _tmp273 * _tmp72;
    _rhs(11, 0) = _tmp243 * _tmp71 + _tmp274 * _tmp72;
    _rhs(12, 0) = _tmp247 * _tmp71 + _tmp275 * _tmp72;
  }
}  // NOLINT(readability/fn_size)

// NOLINTNEXTLINE(readability/fn_size)
}  // namespace sym
