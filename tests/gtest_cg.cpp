#include <gtest/gtest.h>
#include "solvers/cg.h"
#include "sparse_matrix.h"
#include "vector.h"

using namespace math;

static SparseMatrix makeTridiagonal(size_t N) {
    std::vector<size_t> rows, cols;
    std::vector<double> vals;
    for (size_t i = 0; i < N; ++i) {
        rows.push_back(i); cols.push_back(i); vals.push_back(4.0);
        if (i > 0)     { rows.push_back(i); cols.push_back(i-1); vals.push_back(-1.0); }
        if (i+1 < N)   { rows.push_back(i); cols.push_back(i+1); vals.push_back(-1.0); }
    }
    return SparseMatrix::fromCOO(N, N, rows, cols, vals);
}

// ── 小系统精确收敛 ──────────────────────────────────────

TEST(CGTest, SmallSystem) {
    auto A = makeTridiagonal(3);
    Vector b = {1.0, 2.0, 3.0};
    Vector x(3);

    SolverConfig cfg;
    cfg.tol = 1e-12;

    CGSolver cg;
    auto result = cg.solve(A, b, x, cfg);

    EXPECT_TRUE(result.converged);
    EXPECT_LT(result.relative_residual, 1e-12);
    EXPECT_LE(result.iterations, 3);  // CG 理论上 ≤ N 步

    Vector r = b - A.multiply(x);
    EXPECT_LT(r.norm(), 1e-10);
}

// ── 对角矩阵 1 步精确 ──────────────────────────────────

TEST(CGTest, DiagonalOneStep) {
    Vector d = {2.0, 5.0, 3.0, 7.0};
    auto D = SparseMatrix::diagonal(d);
    Vector b = {4.0, 15.0, 9.0, 28.0};
    Vector x(4);

    SolverConfig cfg;
    cfg.tol = 1e-12;

    CGSolver cg;
    auto result = cg.solve(D, b, x, cfg);

    EXPECT_TRUE(result.converged);
    EXPECT_LE(result.iterations, 4);  // 4 个不同特征值 → ≤ 4 步
    EXPECT_TRUE(x.approx_equal(Vector{2.0, 3.0, 3.0, 4.0}));
}

// ── 单位矩阵 ────────────────────────────────────────────

TEST(CGTest, Identity) {
    auto I = SparseMatrix::identity(5);
    Vector b = {1.0, -2.0, 3.0, -4.0, 5.0};
    Vector x(5);

    SolverConfig cfg;
    cfg.tol = 1e-12;

    CGSolver cg;
    auto result = cg.solve(I, b, x, cfg);

    EXPECT_TRUE(result.converged);
    EXPECT_EQ(result.iterations, 1);
    EXPECT_TRUE(x.approx_equal(b));
}

// ── N=100 收敛 ──────────────────────────────────────────

TEST(CGTest, N100) {
    size_t N = 100;
    auto A = makeTridiagonal(N);
    Vector b(N, 1.0);
    Vector x(N);

    SolverConfig cfg;
    cfg.tol = 1e-10;
    cfg.max_iter = 200;

    CGSolver cg;
    auto result = cg.solve(A, b, x, cfg);

    EXPECT_TRUE(result.converged);
    EXPECT_LT(result.relative_residual, 1e-10);
    EXPECT_LE(result.iterations, (int)N);

    Vector r = b - A.multiply(x);
    EXPECT_LT(r.norm(), 1e-8);
}

// ── N=1000 收敛 ─────────────────────────────────────────

TEST(CGTest, N1000) {
    size_t N = 1000;
    auto A = makeTridiagonal(N);
    Vector b(N, 1.0);
    Vector x(N);

    SolverConfig cfg;
    cfg.tol = 1e-10;
    cfg.max_iter = 2000;

    CGSolver cg;
    auto result = cg.solve(A, b, x, cfg);

    EXPECT_TRUE(result.converged);
    EXPECT_LT(result.relative_residual, 1e-10);
}

// ── warm start 更快收敛 ─────────────────────────────────

TEST(CGTest, WarmStartFaster) {
    size_t N = 20;
    auto A1 = makeTridiagonal(N);
    auto A2 = makeTridiagonal(N);
    Vector b(N, 1.0);

    SolverConfig cfg;
    cfg.tol = 1e-10;

    CGSolver cg;

    // cold start
    Vector x_cold(N);
    auto r1 = cg.solve(A1, b, x_cold, cfg);

    // warm start（接近解 + 少量噪声）
    Vector x_warm = x_cold;
    for (size_t i = 0; i < N; ++i) x_warm[i] += 0.01;
    auto r2 = cg.solve(A2, b, x_warm, cfg);

    EXPECT_TRUE(r1.converged);
    EXPECT_TRUE(r2.converged);
    EXPECT_LE(r2.iterations, r1.iterations);
    EXPECT_TRUE(x_cold.approx_equal(x_warm, 1e-8));
}

// ── b=0 ─────────────────────────────────────────────────

TEST(CGTest, ZeroRHS) {
    auto A = makeTridiagonal(5);
    Vector b(5, 0.0);
    Vector x = {1.0, 2.0, 3.0, 4.0, 5.0};

    SolverConfig cfg;
    CGSolver cg;
    auto result = cg.solve(A, b, x, cfg);

    EXPECT_TRUE(result.converged);
    EXPECT_EQ(result.iterations, 0);
    EXPECT_TRUE(x.approx_equal(Vector::zeros(5)));
}

// ── name() ──────────────────────────────────────────────

TEST(CGTest, Name) {
    CGSolver cg;
    EXPECT_EQ(cg.name(), "CG");
}

// ── max_iter 未收敛 ────────────────────────────────────

TEST(CGTest, MaxIterNotConverged) {
    size_t N = 100;
    auto A = makeTridiagonal(N);
    Vector b(N, 1.0);
    Vector x(N);

    SolverConfig cfg;
    cfg.tol = 1e-15;  // 极严格容差
    cfg.max_iter = 2;  // 极少迭代

    CGSolver cg;
    auto result = cg.solve(A, b, x, cfg);

    EXPECT_FALSE(result.converged);
    EXPECT_EQ(result.iterations, 2);
}

// ── 2×2 已知解精确验证 ─────────────────────────────────

TEST(CGTest, TwoByTwo) {
    // A = [[2, 1], [1, 3]], b = [4, 7]
    // 解: x = [1, 2]
    auto A = SparseMatrix::fromCOO(
        2, 2,
        {0, 0, 1, 1},
        {0, 1, 0, 1},
        {2.0, 1.0, 1.0, 3.0});
    Vector b = {4.0, 7.0};
    Vector x(2);

    SolverConfig cfg;
    cfg.tol = 1e-12;

    CGSolver cg;
    auto result = cg.solve(A, b, x, cfg);

    EXPECT_TRUE(result.converged);
    EXPECT_TRUE(x.approx_equal(Vector{1.0, 2.0}, 1e-10));
}
