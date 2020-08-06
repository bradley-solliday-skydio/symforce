//  -----------------------------------------------------------------------------
// This file was autogenerated by symforce. Do NOT modify by hand.
// -----------------------------------------------------------------------------
/**
C++ StorageOps implementation for types in "codegen_multi_function".
*/
#pragma once

#include <vector>

// Create default template traits that implement StorageOps. These will get specialized.
// TODO(hayk): Perhaps StorageOps should always live in the symforce namespace.
namespace codegen_multi_function {
namespace StorageOps {

template<typename T>
size_t StorageDim() {
    return T::StorageDim();
};

template<typename T>
void ToStorage(const T& value, std::vector<double>* vec) {
    value.ToStorage(vec);
}

template<typename Container, typename T>
void FromStorage(const Container& elements, T* out) {
    T::FromStorage(elements, out);
}

}  // namespace StorageOps
}  // namespace codegen_multi_function

// Include StorageOps specializations for each type
#include "./storage_ops/outputs_1_t.h"
#include "./storage_ops/inputs_t.h"
#include "./storage_ops/inputs_states_t.h"
#include "./storage_ops/inputs_constants_t.h"
#include "./storage_ops/outputs_2_t.h"