/* ----------------------------------------------------------------------------
 * SymForce - Copyright 2022, Skydio, Inc.
 * This source code is under the Apache 2.0 license found in the LICENSE file.
 * ---------------------------------------------------------------------------- */

#include <algorithm>
#include <iostream>
#include <random>
#include <thread>

#include <Eigen/Dense>
#include <catch2/catch_test_macros.hpp>

#include <sym/factors/imu_manifold_preintegration_update.h>
#include <sym/rot3.h>
#include <sym/util/epsilon.h>
#include <symforce/slam/imu_preintegration/preintegrated_imu_measurements.h>

using Eigen::Ref;
using Eigen::Vector3d;
using sym::Rot3d;

void UpdateState(const Rot3d& DR, const Ref<const Vector3d>& Dv, const Ref<const Vector3d>& Dp,
                 const Ref<const Vector3d>& gyro, const Ref<const Vector3d>& accel,
                 const Ref<const Vector3d>& gyro_bias, const Ref<const Vector3d>& accel_bias,
                 const double dt, const double epsilon, Rot3d& new_DR, Ref<Vector3d> new_Dv,
                 Ref<Vector3d> new_Dp) {
  const auto corrected_accel = accel - accel_bias;
  new_DR = DR * Rot3d::FromTangent((gyro - gyro_bias) * dt, epsilon);
  new_Dv = Dv + DR * corrected_accel * dt;
  new_Dp = Dp + Dv * dt + DR * corrected_accel * (0.5 * dt * dt);
}

TEST_CASE("ImuManifoldPreintegrationUpdate basic tests", "[slam]") {
  std::mt19937 gen(1804);
  std::normal_distribution<double> dist{0.0, 10.0};
  const auto randomM3 = Eigen::Matrix3d::NullaryExpr([&]() { return dist(gen); });
  const auto randomV3 = Eigen::Vector3d::NullaryExpr([&]() { return dist(gen); });
  const auto randomM9 = Eigen::Matrix<double, 9, 9>::NullaryExpr([&]() { return dist(gen); });

  for (int i_ = 0; i_ < 10; i_++) {
    const sym::Rot3d DR = sym::Rot3d::Random(gen);
    const Eigen::Vector3d Dv = randomV3;
    const Eigen::Vector3d Dp = randomV3;
    const Eigen::Matrix<double, 9, 9> covariance = randomM9;
    const Eigen::Matrix3d DR_D_gyro_bias = randomM3;
    const Eigen::Matrix3d Dv_D_gyro_bias = randomM3;
    const Eigen::Matrix3d Dv_D_accel_bias = randomM3;
    const Eigen::Matrix3d Dp_D_gyro_bias = randomM3;
    const Eigen::Matrix3d Dp_D_accel_bias = randomM3;

    const Eigen::Vector3d gyro = randomV3;
    const Eigen::Vector3d accel = randomV3;

    const double dt = 1.24;

    const Eigen::Vector3d gyro_bias = randomV3;
    const Eigen::Vector3d accel_bias = randomV3;
    const Eigen::Matrix3d gyro_cov = randomM3;
    const Eigen::Matrix3d accel_cov = randomM3;

    sym::Rot3d new_DR;
    Eigen::Vector3d new_Dv;
    Eigen::Vector3d new_Dp;
    Eigen::Matrix<double, 9, 9> new_covarinace;
    Eigen::Matrix3d new_DR_D_gyro_bias;
    Eigen::Matrix3d new_Dp_D_gyro_bias;
    Eigen::Matrix3d new_Dp_D_accel_bias;
    Eigen::Matrix3d new_Dv_D_gyro_bias;
    Eigen::Matrix3d new_Dv_D_accel_bias;

    sym::ImuManifoldPreintegrationUpdate<double>(
        DR, Dv, Dp, covariance, DR_D_gyro_bias, Dv_D_gyro_bias, Dv_D_accel_bias, Dp_D_gyro_bias,
        Dp_D_accel_bias, gyro_bias, accel_bias, gyro_cov, accel_cov, gyro, accel, dt,
        sym::kDefaultEpsilond, &new_DR, &new_Dv, &new_Dp, nullptr, nullptr, nullptr, nullptr,
        nullptr, nullptr);

    sym::Rot3d expected_DR;
    Eigen::Vector3d expected_Dv;
    Eigen::Vector3d expected_Dp;

    UpdateState(DR, Dv, Dp, gyro, accel, gyro_bias, accel_bias, dt, sym::kDefaultEpsilond,
                expected_DR, expected_Dv, expected_Dp);

    CHECK(sym::IsClose(new_DR, expected_DR, 1e-14));
    CHECK(sym::IsClose(new_Dv, expected_Dv, 1e-14));
    CHECK(sym::IsClose(new_Dp, expected_Dp, 1e-14));
  }
}

/**
 * Helper class to generate samples from a multi-variate normal distribution with
 * the same covariance as that passed into the constructor.
 */
class MultiVarNormalDist {
 private:
  const Eigen::MatrixXd L;
  const long int vars;
  std::normal_distribution<double> dist;

 public:
  /**
   * covar is the covariance of the desired normal distribution.
   * Precondition: covar is a symmetric, positive definite matrix.
   */
  MultiVarNormalDist(const Eigen::MatrixXd covar)
      : L{covar.llt().matrixL()}, vars{covar.rows()}, dist{} {}

  /**
   * Returns a matrix whose (count) colums are samples from the distribution.
   */
  template <typename Generator>
  Eigen::MatrixXd Sample(Generator& gen, const int count) {
    Eigen::MatrixXd rand = Eigen::MatrixXd::NullaryExpr(vars, count, [&]() { return dist(gen); });
    return L * rand;
  }
};

/**
 * Calculates the covariance of the columns of samples.
 */
Eigen::MatrixXd Covariance(const Eigen::MatrixXd& samples) {
  const long int sample_size = samples.cols();
  const Eigen::VectorXd avg_col = samples.rowwise().sum() / sample_size;
  const Eigen::MatrixXd meanless_samples = samples.colwise() - avg_col;
  return (meanless_samples * meanless_samples.transpose()) / sample_size;
}

TEST_CASE("Test MultiVarNormalDist generates samples with correct covariance", "[slam]") {
  std::mt19937 gen(170);

  // Create covar, a symmetric positive definite matrix
  // NOTE(brad): This matrix is positive semi-definite because the diagonal entries all
  // have values of at least 2.0, and the max value of an off-diagonal entry is .2. Thus
  // within a row, the sum of the non-diagonal entries is capped by 0.2 * 8 = 1.6. Thus by the
  // Gershgorin circle theorem, any eigen value of covar is at least 0.4 (i.e., positive).
  std::uniform_real_distribution<double> uniform_dist(0.01, 0.1);
  Eigen::MatrixXd covar = Eigen::MatrixXd::NullaryExpr(9, 9, [&]() { return uniform_dist(gen); });
  covar += covar.transpose().eval();
  covar.diagonal().array() += 2.0;

  MultiVarNormalDist dist(covar);

  const Eigen::MatrixXd samples = dist.Sample(gen, 2 << 22);
  const Eigen::Matrix<double, 9, 9> calculated_covar = Covariance(samples);

  for (int col = 0; col < 9; col++) {
    for (int row = 0; row < 9; row++) {
      const double covar_rc = covar(row, col);
      const double difference = covar_rc - calculated_covar(row, col);
      // NOTE(brad): trade-off between sample count (time) and accuracy of calculated_covar
      CHECK(difference * difference < 9e-3 * covar_rc * covar_rc);
    }
  }
}

// TODO: finish tuning this
TEST_CASE("Test PreintegratedImuMeasurements.state_covariance", "[slam]") {
  // Parameters
  const double dt = 1e-3;

  const Eigen::Vector3d accel_bias = {3.4, 1.6, -5.9};
  const Eigen::Vector3d gyro_bias = {1.2, -2.4, 0.5};

  const Eigen::Matrix3d accel_cov =
      (Eigen::Matrix3d() << 7e-5, 1e-7, 1e-7, 1e-7, 7e-5, 1e-7, 1e-7, 1e-7, 7e-5).finished();
  const Eigen::Matrix3d gyro_cov =
      (Eigen::Matrix3d() << 1e-3, 1e-5, 1e-5, 1e-5, 1e-3, 1e-5, 1e-5, 1e-5, 1e-3).finished();

  const Eigen::Vector3d true_gyro = Eigen::Vector3d::Constant(1.2);
  const Eigen::Vector3d true_accel = Eigen::Vector3d::Constant(4.3);


  const int sample_count = 1 << 24;
  Eigen::MatrixXd samples(9, sample_count);
  const int measurements_per_sample = 5;

  // Multithreading
  const int thread_count = 16;
  const int steps_per_thread = sample_count / thread_count;
  // NOTE(brad): I assume I can equally distribute samples among the threads
  CHECK(sample_count % thread_count == 0);
  std::vector<std::thread> threads;
  std::array<Eigen::Matrix<double, 9, 9>, thread_count> covariance_sums;
  covariance_sums.fill(Eigen::Matrix<double, 9, 9>::Zero());

  for (int i = 0; i < thread_count; i++) {
    // Explicit about captures to make checking for thread safety easier
    threads.emplace_back([&accel_bias, &gyro_bias, &accel_cov, &gyro_cov, &true_accel, &true_gyro, &samples, &covariance_sums, dt, i]() {
      std::mt19937 gen(1813 + i);
      // Each thread needs its own distribution as they are not thread safe
      MultiVarNormalDist accel_dist(accel_cov / dt);
      MultiVarNormalDist gyro_dist(gyro_cov / dt);

      Eigen::MatrixXd accel_noise;
      Eigen::MatrixXd gyro_noise;

      for (int j = steps_per_thread * i; j < steps_per_thread * (i + 1); j++) {
        sym::PreintegratedImuMeasurements<double> pim(accel_bias, gyro_bias);

        gyro_noise = gyro_dist.Sample(gen, measurements_per_sample);
        accel_noise = accel_dist.Sample(gen, measurements_per_sample);

        for (int k = 0; k < measurements_per_sample; k++) {
          pim.IntegrateMeasurement(true_accel + accel_noise.col(k), true_gyro + gyro_noise.col(k), accel_cov, gyro_cov, dt, 0.0);
        }

        samples.col(j).segment(0, 3) = pim.DR.ToTangent(sym::kDefaultEpsilond);
        samples.col(j).segment(3, 3) = pim.Dv;
        samples.col(j).segment(6, 3) = pim.Dp;

        covariance_sums[i] += pim.state_covariance;
      }
    });
  }
  for (std::thread& t : threads) {
    t.join();
  }

  Eigen::Matrix<double, 9, 9> calculated_covariance = Eigen::Matrix<double, 9, 9>::Zero();
  for (Eigen::Matrix<double, 9, 9>& cov : covariance_sums) {
    calculated_covariance += cov;
  }
  calculated_covariance *= 1.0 / sample_count;

  const Eigen::MatrixXd true_covariance = Covariance(samples);

  // TODO change to be absolute value
  const double true_covariance_max = true_covariance.maxCoeff();

  auto relative_difference = [true_covariance_max](const double x, const double y) -> double {
    return std::abs((x - y) / x);
  };

  const Eigen::MatrixXd cov_relative_error =
      true_covariance.binaryExpr(calculated_covariance, relative_difference);

  for (int col = 0; col < 9; col++) {
    for (int row = 0; row < 9; row++) {
      const double abs_coef = std::abs(true_covariance(row, col));
      if (abs_coef > 2e-2 * true_covariance_max) {
        CHECK(cov_relative_error(row, col) < 0.01);
      } else if (abs_coef > 2e-3 * true_covariance_max) {
        CHECK(cov_relative_error(row, col) < 0.05);
      } else if (abs_coef > 3e-4 * true_covariance_max) {
        CHECK(cov_relative_error(row, col) < 0.15);
      } else {
        CHECK(cov_relative_error(row, col) < 2);
      }
    }
  }

  //const Eigen::MatrixXd avg_difference = true_covariance - calculated_covariance;

  //std::cout << "true_covariance:\n\n" << true_covariance << std::endl;
  //std::cout << "max of true covariance:\n" << true_covariance.maxCoeff() << std::endl;
  //std::cout << "average calculated_covariance:\n\n" << calculated_covariance << std::endl;
  //std::cout << "avg_relative_difference:\n\n" << cov_relative_error << std::endl;
}


TEST_CASE("New Test bias derivatives", "[slam]") {
  using M33 = Eigen::Matrix<double, 3, 3>;
  using M96 = Eigen::Matrix<double, 9, 6>;
  // Initialize parameters
  const Eigen::Vector3d gyro_bias = {1.2, -3.4, 2.2};
  const Eigen::Vector3d accel_bias = {5.2, -6.4, 5.2};

  const Eigen::Vector3d true_gyro = Eigen::Vector3d::Constant(1.2);
  const Eigen::Vector3d true_accel = Eigen::Vector3d::Constant(4.3);

  const M33 gyro_cov = (M33() << 1e-3, 1e-5, 1e-5, 1e-5, 1e-3, 1e-5, 1e-5, 1e-5, 1e-3).finished();
  const Eigen::Matrix3d accel_cov =
      (M33() << 7e-5, 1e-7, 1e-7, 1e-7, 7e-5, 1e-7, 1e-7, 1e-7, 7e-5).finished();

  const double dt = 0.001;

  const int iterations = 100;

  sym::PreintegratedImuMeasurements<double> pim(accel_bias, gyro_bias);

  for (int k_ = 0; k_ < iterations; k_++) {
    pim.IntegrateMeasurement(true_accel, true_gyro, accel_cov, gyro_cov, dt, sym::kDefaultEpsilond);
  }

  Eigen::Matrix<double, 9, 6> state_D_bias = (Eigen::Matrix<double, 9, 6>() <<
    pim.DR_D_gyro_bias, M33::Zero(),
    pim.Dv_D_gyro_bias, pim.Dv_D_accel_bias,
    pim.Dp_D_gyro_bias, pim.Dp_D_accel_bias
  ).finished();

  // Perturbed inputs and calculate derivatives numerically from them
  const double perturbation = 1e-5;
  const double inverse_perturbation = 1 / perturbation;
  M96 numerical_state_D_bias;
  for (int i = 0; i < 6; i++) {
    // Create biases and perturb one of their coefficients
    Eigen::Vector3d perturbed_gyro_bias = gyro_bias;
    Eigen::Vector3d perturbed_accel_bias = accel_bias;
    if (i < 3) {
      perturbed_gyro_bias[i] += perturbation;
    } else {
      perturbed_accel_bias[i - 3] += perturbation;
    }

    sym::PreintegratedImuMeasurements<double> perturbed_pim(perturbed_accel_bias, perturbed_gyro_bias);
    for (int k_ = 0; k_ < iterations; k_++) {
      perturbed_pim.IntegrateMeasurement(true_accel, true_gyro, accel_cov, gyro_cov, dt, sym::kDefaultEpsilond);
    }

    numerical_state_D_bias.col(i).segment(0, 3) =
        inverse_perturbation * pim.DR.LocalCoordinates(perturbed_pim.DR);
    numerical_state_D_bias.col(i).segment(3, 3) = inverse_perturbation * (perturbed_pim.Dv - pim.Dv);
    numerical_state_D_bias.col(i).segment(6, 3) = inverse_perturbation * (perturbed_pim.Dp - pim.Dp);
  }

  for (int col = 0; col < 6; col++) {
    for (int row = 0; row < 9; row++) {
      const double numerical_coef = numerical_state_D_bias(row, col);
      CHECK(std::abs(state_D_bias(row, col) - numerical_coef) < 1e-1 * std::abs(numerical_coef) + 1e-10);
    }
  }
  // Compare
  //std::cout << "Claimed state_D_bias\n\n" << state_D_bias << std::endl;
  //std::cout << "numerical_state_D_bias\n\n" << numerical_state_D_bias << std::endl;
  //std::cout << "difference\n\n" << (state_D_bias + numerical_state_D_bias) << std::endl;
}
