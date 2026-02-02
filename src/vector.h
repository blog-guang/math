#pragma once

#include <cmath>
#include <initializer_list>
#include <iostream>
#include <vector>

namespace math {

class Vector {
public:
    // ------------------------------------------------------------------
    // Constructors
    // ------------------------------------------------------------------
    explicit Vector(size_t n);
    Vector(size_t n, double fill_value);
    Vector(std::initializer_list<double> init);

    // Copy / Move
    Vector(const Vector& other) = default;
    Vector(Vector&& other) noexcept = default;
    Vector& operator=(const Vector& other) = default;
    Vector& operator=(Vector&& other) noexcept = default;

    // ------------------------------------------------------------------
    // Element access (inline)
    // ------------------------------------------------------------------
    [[nodiscard]] double& operator[](size_t i) noexcept { return data_[i]; }
    [[nodiscard]] const double& operator[](size_t i) const noexcept { return data_[i]; }

    [[nodiscard]] size_t size() const noexcept { return data_.size(); }
    [[nodiscard]] bool empty() const noexcept { return data_.empty(); }

    [[nodiscard]] double* data() noexcept { return data_.data(); }
    [[nodiscard]] const double* data() const noexcept { return data_.data(); }

    // ------------------------------------------------------------------
    // Arithmetic – binary (return new vector)
    // ------------------------------------------------------------------
    [[nodiscard]] Vector operator+(const Vector& rhs) const;
    [[nodiscard]] Vector operator-(const Vector& rhs) const;

    // Scalar multiply: vec * scalar
    [[nodiscard]] Vector operator*(double scalar) const;

    // ------------------------------------------------------------------
    // Arithmetic – compound assignment
    // ------------------------------------------------------------------
    Vector& operator+=(const Vector& rhs);
    Vector& operator-=(const Vector& rhs);
    Vector& operator*=(double scalar);

    // ------------------------------------------------------------------
    // Linear-algebra operations
    // ------------------------------------------------------------------
    [[nodiscard]] double dot(const Vector& other) const;
    [[nodiscard]] double norm() const;

    /// BLAS-style axpy:  this = a * x + this
    void axpy(double a, const Vector& x);

    // ------------------------------------------------------------------
    // Mutation helpers
    // ------------------------------------------------------------------
    void zero() noexcept;
    void fill(double val) noexcept;
    void swap(Vector& other) noexcept;

    // ------------------------------------------------------------------
    // Comparison
    // ------------------------------------------------------------------
    [[nodiscard]] bool operator==(const Vector& other) const;
    bool approx_equal(const Vector& other, double tolerance = 1e-12) const;

    // ------------------------------------------------------------------
    // Static factories
    // ------------------------------------------------------------------
    [[nodiscard]] static Vector zeros(size_t n);
    [[nodiscard]] static Vector ones(size_t n);
    [[nodiscard]] static Vector random(size_t n, unsigned seed = 0);

private:
    std::vector<double> data_;
};

// ---------------------------------------------------------------------------
// Free functions
// ---------------------------------------------------------------------------

/// scalar * vec
[[nodiscard]] Vector operator*(double scalar, const Vector& vec);

/// Stream output
std::ostream& operator<<(std::ostream& os, const Vector& vec);

}  // namespace math
