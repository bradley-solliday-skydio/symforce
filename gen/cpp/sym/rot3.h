// -----------------------------------------------------------------------------
// This file was autogenerated by symforce from template:
//     cpp_templates/geo_package/CLASS.h.jinja
// Do NOT modify by hand.
// -----------------------------------------------------------------------------

#pragma once

#include <ostream>
#include <random>
#include <vector>

#include <Eigen/Dense>

#include "./ops/group_ops.h"
#include "./ops/lie_group_ops.h"
#include "./ops/storage_ops.h"

namespace sym {

/**
 * Autogenerated C++ implementation of <class 'symforce.geo.rot3.Rot3'>.
 *
 * Group of three-dimensional orthogonal matrices with determinant +1, representing
 * rotations in 3D space. Backed by a quaternion with (x, y, z, w) storage.
 */
template <typename ScalarType>
class Rot3 {
 public:
  // Typedefs
  using Scalar = ScalarType;
  using Self = Rot3<Scalar>;
  using DataVec = Eigen::Matrix<Scalar, 4, 1>;
  using TangentVec = Eigen::Matrix<Scalar, 3, 1>;
  using SelfJacobian = Eigen::Matrix<Scalar, 3, 3>;

  // Construct from data vec
  // For rotation types the storage is normalized on construction.
  // This ensures numerical stability as this constructor is called after each codegen operation.
  explicit Rot3(const DataVec& data) : data_(data.normalized()) {}

  // Default construct to identity
  Rot3() : Rot3(GroupOps<Self>::Identity()) {}

  // Access underlying storage as const
  inline const DataVec& Data() const {
    return data_;
  }

  // Matrix type aliases
  using Vector3 = Eigen::Matrix<Scalar, 3, 1>;

  // --------------------------------------------------------------------------
  // Handwritten methods included from "custom_methods/rot3.h.jinja"
  // --------------------------------------------------------------------------

  // Quaternion

  Eigen::Quaternion<Scalar> Quaternion() const {
    return Eigen::Quaternion<Scalar>(data_.data());
  }

  explicit Rot3(const Eigen::Quaternion<Scalar>& q) : Rot3(q.coeffs()) {}

  static Rot3 FromQuaternion(const Eigen::Quaternion<Scalar>& q) {
    return Rot3(q);
  }

  // Angle-axis

  Eigen::AngleAxis<Scalar> AngleAxis() const {
    return Eigen::AngleAxis<Scalar>(Quaternion());
  }

  explicit Rot3(const Eigen::AngleAxis<Scalar>& aa) : Rot3(Eigen::Quaternion<Scalar>(aa)) {}

  static Rot3 FromAngleAxis(const Eigen::AngleAxis<Scalar>& aa) {
    return Rot3(aa);
  }

  static Rot3 FromAngleAxis(const Scalar angle, const Vector3& axis) {
    return Rot3(Eigen::AngleAxis<Scalar>(angle, axis));
  }

  // Rotation matrix

  // Note: ToRotationMatrix is autogenerated below

  static Rot3 FromRotationMatrix(const Eigen::Matrix<Scalar, 3, 3>& mat) {
    return Rot3(Eigen::Quaternion<Scalar>(mat));
  }

  // Euler angles
  // TODO(hayk): Could codegen this.

  Vector3 YawPitchRoll() const {
    return ToRotationMatrix().eulerAngles(2, 1, 0);
  }

  static Rot3 FromYawPitchRoll(const Scalar yaw, const Scalar pitch, const Scalar roll) {
    return Rot3(Eigen::AngleAxis<Scalar>(yaw, Vector3::UnitZ()) *
                Eigen::AngleAxis<Scalar>(pitch, Vector3::UnitY()) *
                Eigen::AngleAxis<Scalar>(roll, Vector3::UnitX()));
  }

  static Rot3 FromYawPitchRoll(const Vector3& ypr) {
    return FromYawPitchRoll(ypr[0], ypr[1], ypr[2]);
  }

  // TODO(hayk): Could codegen this.
  Vector3 Compose(const Vector3& point) const {
    return Quaternion() * point;
  }

  // This function was autogenerated from the symbolic function:
  //    geo.Rot3.random_from_uniform_samples
  static Rot3 RandomFromUniformSamples(const Scalar u1, const Scalar u2, const Scalar u3) {
    // Output array
    Eigen::Matrix<Scalar, 4, 1> res;

    // Intermediate terms (7)
    const Scalar _tmp0 = 2 * M_PI;
    const Scalar _tmp1 = _tmp0 * u2;
    const Scalar _tmp2 = std::sqrt(u1);
    const Scalar _tmp3 = _tmp0 * u3;
    const Scalar _tmp4 = _tmp2 * std::cos(_tmp3);
    const Scalar _tmp5 = (((_tmp4) > 0) - ((_tmp4) < 0));
    const Scalar _tmp6 = _tmp5 * std::sqrt(1 - u1);

    // Output terms (4)
    res[0] = _tmp6 * std::sin(_tmp1);
    res[1] = _tmp6 * std::cos(_tmp1);
    res[2] = _tmp2 * _tmp5 * std::sin(_tmp3);
    res[3] = _tmp4 * _tmp5;

    return Rot3(res);
  }

  // Generate a random element in SO3
  template <typename Generator>
  static Rot3 Random(Generator& gen) {
    std::uniform_real_distribution<Scalar> dist(0.0, 1.0);
    // This cannot be combined into RandomFromUniformSamples(dist(gen), dist(gen), dist(gen)),
    // because the standard does not guarantee evaluation order of arguments,
    // meaning that we would get different results on different compilers.
    const auto dist0 = dist(gen);
    const auto dist1 = dist(gen);
    const auto dist2 = dist(gen);
    return RandomFromUniformSamples(dist0, dist1, dist2);
  }

  // Flip the quaternion if needed to give a positive real part (w).
  // This can be useful for comparing rotations where double cover is an issue.
  Rot3 ToPositiveReal() const {
    if (Data()[3] < 0) {
      return Rot3(-Data());
    } else {
      return Rot3(Data());
    }
  }

  // --------------------------------------------------------------------------
  // Custom generated methods
  // --------------------------------------------------------------------------

  Eigen::Matrix<Scalar, 3, 3> ToRotationMatrix() const;

  // --------------------------------------------------------------------------
  // StorageOps concept
  // --------------------------------------------------------------------------

  static constexpr int32_t StorageDim() {
    return StorageOps<Self>::StorageDim();
  }

  void ToStorage(Scalar* const vec) const {
    return StorageOps<Self>::ToStorage(*this, vec);
  }

  static Rot3 FromStorage(const Scalar* const vec) {
    return StorageOps<Self>::FromStorage(vec);
  }

  // --------------------------------------------------------------------------
  // GroupOps concept
  // --------------------------------------------------------------------------

  static Self Identity() {
    return GroupOps<Self>::Identity();
  }

  Self Inverse() const {
    return GroupOps<Self>::Inverse(*this);
  }

  Self Compose(const Self& b) const {
    return GroupOps<Self>::Compose(*this, b);
  }

  Self Between(const Self& b) const {
    return GroupOps<Self>::Between(*this, b);
  }

  Self InverseWithJacobian(SelfJacobian* const res_D_a = nullptr) const {
    return GroupOps<Self>::InverseWithJacobian(*this, res_D_a);
  }

  Self ComposeWithJacobians(const Self& b, SelfJacobian* const res_D_a = nullptr,
                            SelfJacobian* const res_D_b = nullptr) const {
    return GroupOps<Self>::ComposeWithJacobians(*this, b, res_D_a, res_D_b);
  }

  Self BetweenWithJacobians(const Self& b, SelfJacobian* const res_D_a = nullptr,
                            SelfJacobian* const res_D_b = nullptr) const {
    return GroupOps<Self>::BetweenWithJacobians(*this, b, res_D_a, res_D_b);
  }

  // Compose shorthand
  template <typename Other>
  auto operator*(const Other& b) const -> decltype(Compose(b)) {
    return Compose(b);
  }

  // --------------------------------------------------------------------------
  // LieGroupOps concept
  // --------------------------------------------------------------------------

  static constexpr int32_t TangentDim() {
    return LieGroupOps<Self>::TangentDim();
  }

  static Self FromTangent(const TangentVec& vec, const Scalar epsilon = 1e-8f) {
    return LieGroupOps<Self>::FromTangent(vec, epsilon);
  }

  TangentVec ToTangent(const Scalar epsilon = 1e-8f) const {
    return LieGroupOps<Self>::ToTangent(*this, epsilon);
  }

  Self Retract(const TangentVec& vec, const Scalar epsilon = 1e-8f) const {
    return LieGroupOps<Self>::Retract(*this, vec, epsilon);
  }

  TangentVec LocalCoordinates(const Self& b, const Scalar epsilon = 1e-8f) const {
    return LieGroupOps<Self>::LocalCoordinates(*this, b, epsilon);
  }

  // --------------------------------------------------------------------------
  // General Helpers
  // --------------------------------------------------------------------------

  bool IsApprox(const Self& b, const Scalar tol) const {
    // isApprox is multiplicative so we check the norm for the exact zero case
    // https://eigen.tuxfamily.org/dox/classEigen_1_1DenseBase.html#ae8443357b808cd393be1b51974213f9c
    if (b.Data() == DataVec::Zero()) {
      return Data().norm() < tol;
    }

    return Data().isApprox(b.Data(), tol);
  }

  template <typename ToScalar>
  Rot3<ToScalar> Cast() const {
    return Rot3<ToScalar>(Data().template cast<ToScalar>());
  }

  bool operator==(const Rot3& rhs) const {
    return data_ == rhs.Data();
  }

 protected:
  DataVec data_;
};

// Shorthand for scalar types
using Rot3d = Rot3<double>;
using Rot3f = Rot3<float>;

// Print definitions
std::ostream& operator<<(std::ostream& os, const Rot3<double>& a);
std::ostream& operator<<(std::ostream& os, const Rot3<float>& a);

}  // namespace sym

// Externs to reduce duplicate instantiation
extern template class sym::Rot3<double>;
extern template class sym::Rot3<float>;

// Concept implementations for this class (include order matters here)
// clang-format off
#include "./ops/rot3/storage_ops.h"
#include "./ops/rot3/lie_group_ops.h"
#include "./ops/rot3/group_ops.h"
// clang-format on
