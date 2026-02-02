#include "vector.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <random>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

// 低于此阈值不启动 OpenMP
static constexpr size_t OMP_THRESHOLD = 4096;

namespace math {

// ------------------------------------------------------------------
// Constructors
// ------------------------------------------------------------------

Vector::Vector(size_t n) : data_(n, 0.0) {}

Vector::Vector(size_t n, double fill_value) : data_(n, fill_value) {}

Vector::Vector(std::initializer_list<double> init) : data_(std::move(init)) {}

// ------------------------------------------------------------------
// Binary arithmetic
// ------------------------------------------------------------------

Vector Vector::operator+(const Vector& rhs) const {
    if (size() != rhs.size()) {
        throw std::domain_error("Vector::operator+: size mismatch");
    }
    const size_t N = size();
    Vector result(N);
#ifdef _OPENMP
    if (N >= OMP_THRESHOLD) {
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < N; ++i) result[i] = data_[i] + rhs[i];
    } else
#endif
    {
        for (size_t i = 0; i < N; ++i) result[i] = data_[i] + rhs[i];
    }
    return result;
}

Vector Vector::operator-(const Vector& rhs) const {
    if (size() != rhs.size()) {
        throw std::domain_error("Vector::operator-: size mismatch");
    }
    const size_t N = size();
    Vector result(N);
#ifdef _OPENMP
    if (N >= OMP_THRESHOLD) {
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < N; ++i) result[i] = data_[i] - rhs[i];
    } else
#endif
    {
        for (size_t i = 0; i < N; ++i) result[i] = data_[i] - rhs[i];
    }
    return result;
}

Vector Vector::operator*(double scalar) const {
    const size_t N = size();
    Vector result(N);
#ifdef _OPENMP
    if (N >= OMP_THRESHOLD) {
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < N; ++i) result[i] = data_[i] * scalar;
    } else
#endif
    {
        for (size_t i = 0; i < N; ++i) result[i] = data_[i] * scalar;
    }
    return result;
}

// ------------------------------------------------------------------
// Compound assignment
// ------------------------------------------------------------------

Vector& Vector::operator+=(const Vector& rhs) {
    if (size() != rhs.size()) {
        throw std::domain_error("Vector::operator+=: size mismatch");
    }
    const size_t N = size();
#ifdef _OPENMP
    if (N >= OMP_THRESHOLD) {
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < N; ++i) data_[i] += rhs[i];
    } else
#endif
    {
        for (size_t i = 0; i < N; ++i) data_[i] += rhs[i];
    }
    return *this;
}

Vector& Vector::operator-=(const Vector& rhs) {
    if (size() != rhs.size()) {
        throw std::domain_error("Vector::operator-=: size mismatch");
    }
    const size_t N = size();
#ifdef _OPENMP
    if (N >= OMP_THRESHOLD) {
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < N; ++i) data_[i] -= rhs[i];
    } else
#endif
    {
        for (size_t i = 0; i < N; ++i) data_[i] -= rhs[i];
    }
    return *this;
}

Vector& Vector::operator*=(double scalar) {
    const size_t N = size();
#ifdef _OPENMP
    if (N >= OMP_THRESHOLD) {
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < N; ++i) data_[i] *= scalar;
    } else
#endif
    {
        for (size_t i = 0; i < N; ++i) data_[i] *= scalar;
    }
    return *this;
}

// ------------------------------------------------------------------
// Linear-algebra
// ------------------------------------------------------------------

double Vector::dot(const Vector& other) const {
    if (size() != other.size()) {
        throw std::domain_error("Vector::dot: size mismatch");
    }
    const size_t N = size();
    double sum = 0.0;
#ifdef _OPENMP
    if (N >= OMP_THRESHOLD) {
        #pragma omp parallel for schedule(static) reduction(+:sum)
        for (size_t i = 0; i < N; ++i) sum += data_[i] * other[i];
    } else
#endif
    {
        for (size_t i = 0; i < N; ++i) sum += data_[i] * other[i];
    }
    return sum;
}

double Vector::norm() const {
    return std::sqrt(dot(*this));
}

void Vector::axpy(double a, const Vector& x) {
    if (size() != x.size()) {
        throw std::domain_error("Vector::axpy: size mismatch");
    }
    const size_t N = size();
#ifdef _OPENMP
    if (N >= OMP_THRESHOLD) {
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < N; ++i) data_[i] += a * x[i];
    } else
#endif
    {
        for (size_t i = 0; i < N; ++i) data_[i] += a * x[i];
    }
}

// ------------------------------------------------------------------
// Mutation helpers (inline-eligible but kept here for clarity;
// the compiler will inline small loops at -O2+)
// ------------------------------------------------------------------

void Vector::zero() noexcept {
    std::fill(data_.begin(), data_.end(), 0.0);
}

void Vector::fill(double val) noexcept {
    std::fill(data_.begin(), data_.end(), val);
}

void Vector::swap(Vector& other) noexcept {
    data_.swap(other.data_);
}

// ------------------------------------------------------------------
// Comparison
// ------------------------------------------------------------------

bool Vector::operator==(const Vector& other) const {
    return approx_equal(other);
}

bool Vector::approx_equal(const Vector& other, double tolerance) const {
    if (size() != other.size()) return false;
    for (size_t i = 0; i < size(); ++i) {
        if (std::abs(data_[i] - other[i]) > tolerance) return false;
    }
    return true;
}

// ------------------------------------------------------------------
// Static factories
// ------------------------------------------------------------------

Vector Vector::zeros(size_t n) {
    return Vector(n, 0.0);
}

Vector Vector::ones(size_t n) {
    return Vector(n, 1.0);
}

Vector Vector::random(size_t n, unsigned seed) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    Vector v(n);
    for (size_t i = 0; i < n; ++i) {
        v[i] = dist(gen);
    }
    return v;
}

// ------------------------------------------------------------------
// Free functions
// ------------------------------------------------------------------

Vector operator*(double scalar, const Vector& vec) {
    return vec * scalar;
}

std::ostream& operator<<(std::ostream& os, const Vector& vec) {
    os << "[";
    for (size_t i = 0; i < vec.size(); ++i) {
        if (i != 0) os << ", ";
        os << std::setprecision(6) << vec[i];
    }
    os << "]";
    return os;
}

}  // namespace math
