#include <gtest/gtest.h>
#include "preconditioners/ilu0.h"
#include "solvers/cg.h"
#include "solvers/bicgstab.h"
#include "sparse_matrix.h"
#include "vector.h"

using namespace math;

static SparseMatrix makeSymmetric(size_t N) {
    std::vector<size_t> rows, cols;
    std::vector<double> vals;
    rows.reserve(3*N); cols.reserve(3*N); vals.reserve(3*N);
    for (size_t i = 0; i < N; ++i) {
        rows.push_back(i); cols.push_back(i); vals.push_back(4.0);
        if (i > 0)   { rows.push_back(i); cols.push_back(i-1); vals.push_back(-1.0); }
        if (i+1 < N) { rows.push_back(i); cols.push_back(i+1); vals.push_back(-1.0); }
    }
    return SparseMatrix::fromCOO(N, N, rows, cols, vals);
}

static SparseMatrix makeNonSymmetric(size_t N) {
    std::vector<size_t> rows, cols;
    std::vector<double> vals;
    rows.reserve(3*N); cols.reserve(3*N); vals.reserve(3*N);
    for (size_t i = 0; i < N; ++i) {
        rows.push_back(i); cols.push_back(i); vals.push_back(4.0);
        if (i > 0)   { rows.push_back(i); cols.push_back(i-1); vals.push_back(-1.0); }
        if (i+1 < N) { rows.push_back(i); cols.push_back(i+1); vals.push_back(-2.0); }
    }
    return SparseMatrix::fromCOO(N, N, rows, cols, vals);
}

// ── 大规模向量运算正确性 ──────────────────────────────

TEST(OpenMPTest, VectorOpsLarge) {
    size_t N = 100000;
    Vector a = Vector::random(N, 42);
    Vector b = Vector::random(N, 99);

    Vector c = a + b;
    EXPECT_NEAR(c[0], a[0] + b[0], 1e-15);
    EXPECT_NEAR(c[N-1], a[N-1] + b[N-1], 1e-15);

    Vector d = a - b;
    EXPECT_NEAR(d[0], a[0] - b[0], 1e-15);

    Vector e = a * 3.14;
    EXPECT_NEAR(e[0], a[0] * 3.14, 1e-14);

    double dot_val = a.dot(b);
    EXPECT_NE(dot_val, 0.0);

    Vector f = a;
    f.axpy(2.0, b);
    EXPECT_NEAR(f[0], a[0] + 2.0 * b[0], 1e-14);
}

// ── 大规模 SpMV 正确性 ──────────────────────────────────

TEST(OpenMPTest, SpMVLarge) {
    size_t N = 100000;
    auto A = makeSymmetric(N);
    A.toCSR();
    Vector x = Vector::random(N, 7);
    Vector y = A.multiply(x);

    // 验证首行: 4*x[0] - x[1]
    EXPECT_NEAR(y[0], 4.0*x[0] - x[1], 1e-12);

    // 中间行: -x[i-1] + 4*x[i] - x[i+1]
    size_t mid = N / 2;
    EXPECT_NEAR(y[mid], -x[mid-1] + 4.0*x[mid] - x[mid+1], 1e-12);

    // 末行: -x[N-2] + 4*x[N-1]
    EXPECT_NEAR(y[N-1], -x[N-2] + 4.0*x[N-1], 1e-12);
}

// ── 大规模 CG 正确性 ──────────────────────────────────

TEST(OpenMPTest, CGLargeCorrectness) {
    size_t N = 50000;
    auto A = makeSymmetric(N);
    Vector b(N, 1.0);
    Vector x(N);

    SolverConfig cfg;
    cfg.tol = 1e-10;
    cfg.max_iter = 200;

    CGSolver cg;
    auto result = cg.solve(A, b, x, cfg);

    EXPECT_TRUE(result.converged);
    EXPECT_LT(result.relative_residual, 1e-10);

    Vector r = b - A.multiply(x);
    EXPECT_LT(r.norm() / b.norm(), 1e-10);
}

// ── 大规模 BiCGSTAB 正确性（非对称） ────────────────

TEST(OpenMPTest, BiCGSTABLargeCorrectness) {
    size_t N = 50000;
    auto A = makeNonSymmetric(N);
    Vector b(N, 1.0);
    Vector x(N);

    SolverConfig cfg;
    cfg.tol = 1e-10;
    cfg.max_iter = 500;

    BiCGSTABSolver solver;
    auto result = solver.solve(A, b, x, cfg);

    EXPECT_TRUE(result.converged);
    EXPECT_LT(result.relative_residual, 1e-10);
}

// ── 大规模 ILU(0)+CG ──────────────────────────────────

TEST(OpenMPTest, ILU0CGLarge) {
    size_t N = 50000;
    auto A = makeSymmetric(N);
    A.toCSR();

    ILU0Preconditioner ilu;
    ilu.build(A);

    Vector b(N, 1.0);
    Vector x(N);

    SolverConfig cfg;
    cfg.tol = 1e-10;
    cfg.max_iter = 200;
    cfg.precond = &ilu;

    CGSolver cg;
    auto result = cg.solve(A, b, x, cfg);

    EXPECT_TRUE(result.converged);
    EXPECT_LT(result.relative_residual, 1e-10);
}
