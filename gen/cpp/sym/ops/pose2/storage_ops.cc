// -----------------------------------------------------------------------------
// This file was autogenerated by symforce from template:
//     cpp_templates/geo_package/ops/CLASS/storage_ops.cc.jinja
// Do NOT modify by hand.
// -----------------------------------------------------------------------------

#include "./storage_ops.h"

namespace sym {

template <typename ScalarType>
void StorageOps<Pose2<ScalarType>>::ToStorage(const Pose2<ScalarType>& a, ScalarType* out) {
  assert(out != nullptr);
  std::copy_n(a.Data().data(), a.StorageDim(), out);
}

template <typename ScalarType>
Pose2<ScalarType> StorageOps<Pose2<ScalarType>>::FromStorage(const ScalarType* data) {
  assert(data != nullptr);
  return Pose2<ScalarType>(Eigen::Map<const typename Pose2<ScalarType>::DataVec>(data));
}

}  // namespace sym

// Explicit instantiation
template struct sym::StorageOps<sym::Pose2<double>>;
template struct sym::StorageOps<sym::Pose2<float>>;
