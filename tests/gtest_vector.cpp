#include <gtest/gtest.h>
#include <sstream>
#include "vector.h"

using namespace math;

// ── Construction ────────────────────────────────────────────────

TEST(VectorTest, SizeConstructorZeroFills) {
    Vector v(5);
    EXPECT_EQ(v.size(), 5);
    for (size_t i = 0; i < v.size(); ++i)
        EXPECT_DOUBLE_EQ(v[i], 0.0);
}

TEST(VectorTest, FillConstructor) {
    Vector v(3, 7.0);
    EXPECT_EQ(v.size(), 3);
    for (size_t i = 0; i < v.size(); ++i)
        EXPECT_DOUBLE_EQ(v[i], 7.0);
}

TEST(VectorTest, InitializerList) {
    Vector v = {1.0, 2.0, 3.0};
    EXPECT_EQ(v.size(), 3);
    EXPECT_DOUBLE_EQ(v[0], 1.0);
    EXPECT_DOUBLE_EQ(v[1], 2.0);
    EXPECT_DOUBLE_EQ(v[2], 3.0);
}

TEST(VectorTest, CopyAndMove) {
    Vector orig = {1.0, 2.0, 3.0};

    Vector copy = orig;
    EXPECT_TRUE(copy == orig);

    Vector moved = std::move(copy);
    EXPECT_TRUE(moved == orig);

    Vector assigned(1);
    assigned = orig;
    EXPECT_TRUE(assigned == orig);

    Vector src = {10.0, 20.0};
    Vector dst(1);
    dst = std::move(src);
    EXPECT_EQ(dst.size(), 2);
    EXPECT_DOUBLE_EQ(dst[0], 10.0);
    EXPECT_DOUBLE_EQ(dst[1], 20.0);
}

TEST(VectorTest, EmptyAndDataPtr) {
    Vector empty(0);
    EXPECT_TRUE(empty.empty());
    EXPECT_EQ(empty.size(), 0);

    Vector v = {1.0, 2.0};
    EXPECT_FALSE(v.empty());
    EXPECT_NE(v.data(), nullptr);
    EXPECT_DOUBLE_EQ(v.data()[0], 1.0);
}

// ── Arithmetic ──────────────────────────────────────────────────

TEST(VectorTest, Addition) {
    Vector a = {1.0, 2.0, 3.0};
    Vector b = {4.0, 5.0, 6.0};
    Vector c = a + b;
    EXPECT_NEAR(c[0], 5.0, 1e-15);
    EXPECT_NEAR(c[1], 7.0, 1e-15);
    EXPECT_NEAR(c[2], 9.0, 1e-15);
}

TEST(VectorTest, Subtraction) {
    Vector a = {1.0, 2.0, 3.0};
    Vector b = {4.0, 5.0, 6.0};
    Vector d = b - a;
    EXPECT_NEAR(d[0], 3.0, 1e-15);
    EXPECT_NEAR(d[1], 3.0, 1e-15);
    EXPECT_NEAR(d[2], 3.0, 1e-15);
}

TEST(VectorTest, ScalarMultiply) {
    Vector a = {1.0, 2.0, 3.0};
    Vector e = a * 3.0;
    EXPECT_NEAR(e[0], 3.0, 1e-15);
    EXPECT_NEAR(e[1], 6.0, 1e-15);
    EXPECT_NEAR(e[2], 9.0, 1e-15);

    Vector f = 2.0 * Vector{4.0, 5.0, 6.0};
    EXPECT_NEAR(f[0], 8.0, 1e-15);
    EXPECT_NEAR(f[1], 10.0, 1e-15);
    EXPECT_NEAR(f[2], 12.0, 1e-15);
}

TEST(VectorTest, CompoundAssignment) {
    Vector a = {1.0, 2.0, 3.0};
    Vector b = {4.0, 5.0, 6.0};

    Vector g = a; g += b;
    EXPECT_TRUE(g == (a + b));

    Vector h = b; h -= a;
    EXPECT_TRUE(h == (b - a));

    Vector i = a; i *= 3.0;
    EXPECT_TRUE(i == (a * 3.0));
}

// ── Dot & Norm ──────────────────────────────────────────────────

TEST(VectorTest, DotProduct) {
    Vector a = {1.0, 2.0, 3.0};
    Vector b = {4.0, 5.0, 6.0};
    // 1*4 + 2*5 + 3*6 = 32
    EXPECT_NEAR(a.dot(b), 32.0, 1e-12);
}

TEST(VectorTest, Norm) {
    Vector a = {1.0, 2.0, 3.0};
    EXPECT_NEAR(a.norm(), std::sqrt(14.0), 1e-12);

    Vector z(3);
    EXPECT_NEAR(z.norm(), 0.0, 1e-15);
}

TEST(VectorTest, DotSelfEqualsNormSquared) {
    Vector a = {1.0, 2.0, 3.0};
    EXPECT_NEAR(a.dot(a), a.norm() * a.norm(), 1e-12);
}

// ── Axpy ────────────────────────────────────────────────────────

TEST(VectorTest, Axpy) {
    Vector x = {1.0, 2.0, 3.0};
    Vector y = {10.0, 20.0, 30.0};
    y.axpy(2.0, x);  // y = 2*x + y
    EXPECT_NEAR(y[0], 12.0, 1e-15);
    EXPECT_NEAR(y[1], 24.0, 1e-15);
    EXPECT_NEAR(y[2], 36.0, 1e-15);
}

TEST(VectorTest, AxpyZeroAlpha) {
    Vector y = {5.0, 6.0, 7.0};
    Vector x = {100.0, 200.0, 300.0};
    y.axpy(0.0, x);
    EXPECT_NEAR(y[0], 5.0, 1e-15);
    EXPECT_NEAR(y[1], 6.0, 1e-15);
    EXPECT_NEAR(y[2], 7.0, 1e-15);
}

TEST(VectorTest, AxpyNegativeAlpha) {
    Vector y = {10.0, 10.0, 10.0};
    Vector x = {3.0, 5.0, 7.0};
    y.axpy(-1.0, x);  // y = -x + y
    EXPECT_NEAR(y[0], 7.0, 1e-15);
    EXPECT_NEAR(y[1], 5.0, 1e-15);
    EXPECT_NEAR(y[2], 3.0, 1e-15);
}

// ── Factories ───────────────────────────────────────────────────

TEST(VectorTest, Zeros) {
    Vector z = Vector::zeros(5);
    EXPECT_EQ(z.size(), 5);
    for (size_t i = 0; i < z.size(); ++i)
        EXPECT_DOUBLE_EQ(z[i], 0.0);
}

TEST(VectorTest, Ones) {
    Vector o = Vector::ones(4);
    EXPECT_EQ(o.size(), 4);
    for (size_t i = 0; i < o.size(); ++i)
        EXPECT_DOUBLE_EQ(o[i], 1.0);
}

TEST(VectorTest, RandomDeterministic) {
    Vector r1 = Vector::random(10, 12345);
    Vector r2 = Vector::random(10, 12345);
    EXPECT_TRUE(r1 == r2);  // 同 seed → 同序列

    for (size_t i = 0; i < r1.size(); ++i) {
        EXPECT_GE(r1[i], 0.0);
        EXPECT_LT(r1[i], 1.0);
    }

    Vector r3 = Vector::random(10, 99999);
    EXPECT_FALSE(r1 == r3);  // 不同 seed → 不同向量
}

// ── Edge Cases ──────────────────────────────────────────────────

TEST(VectorTest, EmptyVectorOps) {
    Vector empty(0);
    EXPECT_TRUE(empty.empty());
    EXPECT_NEAR(empty.norm(), 0.0, 1e-15);

    Vector e2 = empty + empty;
    EXPECT_TRUE(e2.empty());

    Vector e3 = empty * 5.0;
    EXPECT_TRUE(e3.empty());

    EXPECT_TRUE(empty == Vector(0));
}

TEST(VectorTest, SingleElement) {
    Vector v = {42.0};
    EXPECT_EQ(v.size(), 1);
    EXPECT_NEAR(v.norm(), 42.0, 1e-12);
    EXPECT_NEAR(v.dot(v), 42.0 * 42.0, 1e-10);

    Vector sum = v + Vector{3.0};
    EXPECT_NEAR(sum[0], 45.0, 1e-15);
}

TEST(VectorTest, ZeroAndFill) {
    Vector v = {1.0, 2.0, 3.0};
    v.zero();
    for (size_t i = 0; i < v.size(); ++i)
        EXPECT_DOUBLE_EQ(v[i], 0.0);

    v.fill(9.5);
    for (size_t i = 0; i < v.size(); ++i)
        EXPECT_DOUBLE_EQ(v[i], 9.5);
}

TEST(VectorTest, Swap) {
    Vector s1 = {1.0, 2.0};
    Vector s2 = {3.0, 4.0, 5.0};
    s1.swap(s2);
    EXPECT_EQ(s1.size(), 3);
    EXPECT_EQ(s2.size(), 2);
    EXPECT_NEAR(s1[0], 3.0, 1e-15);
    EXPECT_NEAR(s2[0], 1.0, 1e-15);
}

TEST(VectorTest, ApproxEqualTolerance) {
    Vector t1 = {1.0, 2.0};
    Vector t2 = {1.0 + 1e-8, 2.0 - 1e-8};
    EXPECT_FALSE(t1.approx_equal(t2, 1e-12));  // 严格：不等
    EXPECT_TRUE(t1.approx_equal(t2, 1e-7));    // 宽松：相等
}

TEST(VectorTest, ApproxEqualDifferentSize) {
    Vector a = {1.0, 2.0};
    Vector b = {1.0, 2.0, 3.0};
    EXPECT_FALSE(a.approx_equal(b));
    EXPECT_FALSE(a == b);
}

TEST(VectorTest, StreamOutput) {
    Vector v = {1.0, 2.0, 3.0};
    std::ostringstream oss;
    oss << v;
    std::string s = oss.str();
    // 输出应包含三个数值
    EXPECT_NE(s.find("1"), std::string::npos);
    EXPECT_NE(s.find("2"), std::string::npos);
    EXPECT_NE(s.find("3"), std::string::npos);
}

TEST(VectorTest, StreamOutputEmpty) {
    Vector v(0);
    std::ostringstream oss;
    oss << v;
    // 空向量不应崩溃，输出为合法字符串
    EXPECT_TRUE(oss.good());
}

TEST(VectorTest, LargeScalarMultiply) {
    size_t N = 10000;
    Vector v(N, 1.0);
    Vector scaled = v * 1e6;
    for (size_t i = 0; i < N; ++i)
        EXPECT_DOUBLE_EQ(scaled[i], 1e6);

    Vector scaled2 = 1e-6 * v;
    for (size_t i = 0; i < N; ++i)
        EXPECT_NEAR(scaled2[i], 1e-6, 1e-20);
}

TEST(VectorTest, NegativeValues) {
    Vector v = {-3.0, -2.0, -1.0};
    EXPECT_NEAR(v.norm(), std::sqrt(14.0), 1e-12);
    EXPECT_NEAR(v.dot(v), 14.0, 1e-12);

    Vector neg = v * (-1.0);
    EXPECT_NEAR(neg[0], 3.0, 1e-15);
    EXPECT_NEAR(neg[1], 2.0, 1e-15);
    EXPECT_NEAR(neg[2], 1.0, 1e-15);
}
