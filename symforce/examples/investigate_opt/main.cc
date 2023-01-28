#include <iostream>

#include <Eigen/Core>

#include <symforce/opt/factor.h>
#include <symforce/opt/values.h>
#include <symforce/opt/templates.h>
#include <symforce/opt/internal/factor_utils.h>

template <typename Scalar>
void ExampleFactor(double x, double y, double z, Eigen::Map<Eigen::Matrix<Scalar, 1, 1>>* const res = nullptr,
                   Eigen::Map<Eigen::Matrix<Scalar, 1, 3>>* const jacobian = nullptr,
                   Eigen::Map<Eigen::Matrix<Scalar, 3, 3>>* const hessian = nullptr,
                   Eigen::Map<Eigen::Matrix<Scalar, 3, 1>>* const rhs = nullptr) {
  if (res != nullptr) {
    (*res)(0, 0) = x;
  }
  if (jacobian != nullptr) {
    decltype(*jacobian)& jac = *jacobian;
    jac(0, 0) = x;
    jac(0, 1) = x + 1;
    jac(0, 2) = x + 2;
  }
  if (hessian != nullptr) {
    *hessian = x * Eigen::Matrix<Scalar, 3, 3>::Identity();
  }
  if (rhs != nullptr) {
    (*rhs)(0, 0) = x;
    (*rhs)(1, 0) = x - 1;
    (*rhs)(2, 0) = x - 2;
  }
}

template <typename T>
void PrintType() {
    std::cout << __PRETTY_FUNCTION__ << std::endl;
}

template <typename Functor>
void ViewFunctorInfo(Functor func) {
  using Traits = sym::function_traits<Functor>;

  using ResidualVec = typename sym::internal::HessianFuncTypeHelper<Functor>::ResidualVec;
  using JacobianMat = typename sym::internal::HessianFuncTypeHelper<Functor>::JacobianMat;
  using HessianMat = typename sym::internal::HessianFuncTypeHelper<Functor>::HessianMat;
  using RhsVec = typename sym::internal::HessianFuncTypeHelper<Functor>::RhsVec;
  std::cout << "Residual Vec: ";
  PrintType<ResidualVec>();
  std::cout << "JacobianMat: ";
  PrintType<JacobianMat>();
  std::cout << "HessianMat: ";
  PrintType<HessianMat>();
  std::cout << "RhsVec: ";
  PrintType<RhsVec>();

  constexpr int M = ResidualVec::RowsAtCompileTime;
  constexpr int N = RhsVec::RowsAtCompileTime;
  std::cout << "M: " << M << std::endl;
  std::cout << "N: " << N << std::endl;

  using JacobianMat2 = typename sym::internal::HessianFuncValuesExtractor<double, Functor>::JacobianMat;
  std::cout << "JacobianMat2: ";
  PrintType<JacobianMat2>();
}

int main() {
  double res_data;
  Eigen::Map<Eigen::Matrix<double, 1, 1>> res(&res_data);
  double jacobian_data[3];
  Eigen::Map<Eigen::Matrix<double, 1, 3>> jacobian(jacobian_data);
  double hessian_data[9];
  Eigen::Map<Eigen::Matrix3d> hessian(hessian_data);
  double rhs_data[3];
  Eigen::Map<Eigen::Vector3d> rhs(rhs_data);
  ExampleFactor(3.14, 0, 0, &res, &jacobian, &hessian, &rhs);
  std::cout << "res:\n" << res << "\n\n";
  std::cout << "jacobian:\n" << jacobian << "\n\n";
  std::cout << "hessian:\n" << hessian << "\n\n";
  std::cout << "rhs:\n" << rhs << '\n' << std::endl;

  //ViewFunctorInfo(ExampleFactor<double>);
  sym::Factord factor = sym::Factord::Hessian(ExampleFactor<double>, {'x', 'y', 'z'});

  sym::Valuesd values;
  values.Set<double>({'x'}, 3.14);
  values.Set<double>({'y'}, 0);
  values.Set<double>({'z'}, 0);
  const sym::Factord::LinearizedDenseFactor linearized_factor = factor.Linearize(values);
  std::cout << "res:\n" << linearized_factor.residual << "\n\n";
  std::cout << "jacobian:\n" << linearized_factor.jacobian << "\n\n";
  std::cout << "hessian:\n" << linearized_factor.hessian << "\n\n";
  std::cout << "rhs:\n" << linearized_factor.rhs << std::endl;
}
