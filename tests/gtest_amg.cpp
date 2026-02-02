#include <gtest/gtest.h>
#include "preconditioners/amg.h"
#include "solvers/cg.h"
#include "sparse_matrix.h"
#include "vector.h"

using namespace math;

static SparseMatrix poisson1D(size_t N) {
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

static SparseMatrix poisson2D(size_t n) {
    size_t N = n * n;
    std::vector<size_t> rows, cols;
    std::vector<double> vals;
    rows.reserve(5*N); cols.reserve(5*N); vals.reserve(5*N);
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            size_t k = i * n + j;
            rows.push_back(k); cols.push_back(k); vals.push_back(4.0);
            if (j > 0)   { rows.push_back(k); cols.push_back(k-1); vals.push_back(-1.0); }
            if (j+1 < n) { rows.push_back(k); cols.push_back(k+1); vals.push_back(-1.0); }
            if (i > 0)   { rows.push_back(k); cols.push_back(k-n); vals.push_back(-1.0); }
            if (i+1 < n) { rows.push_back(k); cols.push_back(k+n); vals.push_back(-1.0); }
        }
    }
    return SparseMatrix::fromCOO(N, N, rows, cols, vals);
}

// ── 多层级结构 ──────────────────────────────────────────

TEST(AMGTest, BuildPoisson1D) {
    auto A = poisson1D(256);
    A.toCSR();
    AMGPreconditioner amg;
    amg.build(A);

    EXPECT_GT(amg.numLevels(), size_t(1));
}

// ── apply() 输出合理 ───────────────────────────────────

TEST(AMGTest, ApplyNonZeroFinite) {
    auto A = poisson1D(128);
    A.toCSR();
    AMGPreconditioner amg;
    amg.build(A);

    Vector r(128, 1.0);
    Vector z = amg.apply(r);

    EXPECT_GT(z.norm(), 1e-10);
    EXPECT_TRUE(std::isfinite(z.norm()));
}

// ── CG+AMG 迭代次数少于 CG（1D） ────────────────────

TEST(AMGTest, CG_vs_CGAMG_Poisson1D) {
    size_t N = 1000;
    Vector b(N, 1.0);

    int iters_cg;
    {
        auto A = poisson1D(N);
        CGSolver cg;
        SolverConfig cfg;
        cfg.tol = 1e-10;
        cfg.max_iter = 5000;
        cfg.precond = nullptr;
        Vector x(N);
        auto res = cg.solve(A, b, x, cfg);
        iters_cg = res.iterations;
    }

    int iters_amg;
    {
        auto A = poisson1D(N);
        A.toCSR();
        AMGPreconditioner amg;
        amg.build(A);

        CGSolver cg;
        SolverConfig cfg;
        cfg.tol = 1e-10;
        cfg.max_iter = 5000;
        cfg.precond = &amg;
        Vector x(N);
        auto res = cg.solve(A, b, x, cfg);
        iters_amg = res.iterations;
        EXPECT_TRUE(res.converged);
    }

    EXPECT_LT(iters_amg, iters_cg);
}

// ── CG+AMG 迭代次数少于 CG（2D） ────────────────────

TEST(AMGTest, CG_vs_CGAMG_Poisson2D) {
    size_t n = 32, N = n * n;
    Vector b(N, 1.0);

    int iters_cg;
    {
        auto A = poisson2D(n);
        CGSolver cg;
        SolverConfig cfg;
        cfg.tol = 1e-10;
        cfg.max_iter = 10000;
        cfg.precond = nullptr;
        Vector x(N);
        auto res = cg.solve(A, b, x, cfg);
        iters_cg = res.iterations;
    }

    int iters_amg;
    {
        auto A = poisson2D(n);
        A.toCSR();
        AMGPreconditioner amg;
        amg.build(A);

        CGSolver cg;
        SolverConfig cfg;
        cfg.tol = 1e-10;
        cfg.max_iter = 10000;
        cfg.precond = &amg;
        Vector x(N);
        auto res = cg.solve(A, b, x, cfg);
        iters_amg = res.iterations;
        EXPECT_TRUE(res.converged);
    }

    EXPECT_LT(iters_amg, iters_cg);
}

// ── 小矩阵直接求解路径 ──────────────────────────────────

TEST(AMGTest, SmallMatrixDirectSolve) {
    auto A = poisson1D(4);
    A.toCSR();
    AMGPreconditioner amg;
    amg.build(A);

    // N=4 <= coarse_threshold=64 → 仅 1 层
    EXPECT_EQ(amg.numLevels(), 1);

    Vector b(4, 1.0);
    Vector z = amg.apply(b);

    // apply 等价于直接求解，验证 ||Az - b|| < 1e-10
    Vector Az = A.multiply(z);
    EXPECT_LT((Az - b).norm(), 1e-10);
}

// ── name() ──────────────────────────────────────────────

TEST(AMGTest, Name) {
    AMGPreconditioner amg;
    EXPECT_EQ(amg.name(), "AMG");
}

// ── 自定义配置 ──────────────────────────────────────────

TEST(AMGTest, CustomConfig) {
    AMGConfig cfg;
    cfg.max_levels = 3;
    cfg.coarse_threshold = 32;
    cfg.nu_pre = 1;
    cfg.nu_post = 1;

    auto A = poisson1D(256);
    A.toCSR();

    AMGPreconditioner amg(cfg);
    amg.build(A);

    // 最多3层
    EXPECT_LE(amg.numLevels(), size_t(3));

    Vector b(256, 1.0);
    Vector z = amg.apply(b);
    EXPECT_TRUE(std::isfinite(z.norm()));
    EXPECT_GT(z.norm(), 1e-10);
}

// ── 零向量 apply ──────────────────────────────────────

TEST(AMGTest, ApplyZeroVector) {
    auto A = poisson1D(64);
    A.toCSR();
    AMGPreconditioner amg;
    amg.build(A);

    Vector r(64, 0.0);
    Vector z = amg.apply(r);
    EXPECT_LT(z.norm(), 1e-12);  // M⁻¹ * 0 = 0
}
