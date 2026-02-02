#include <gtest/gtest.h>
#include "preconditioners/amgcl_wrapper.h"
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

// ── AMGCL 构建成功 ─────────────────────────────────────

TEST(AMGCLTest, BuildPoisson1D) {
    auto A = poisson1D(128);
    A.toCSR();
    AMGCLPreconditioner amgcl;
    
    EXPECT_NO_THROW({
        amgcl.build(A);
    });
    
    EXPECT_EQ(amgcl.name(), "AMGCL");
}

// ── AMGCL apply() 输出合理 ────────────────────────────

TEST(AMGCLTest, ApplyNonZeroFinite) {
    auto A = poisson1D(64);
    A.toCSR();
    AMGCLPreconditioner amgcl;
    amgcl.build(A);

    Vector r(64, 1.0);
    Vector z = amgcl.apply(r);

    EXPECT_GT(z.norm(), 1e-10);
    EXPECT_TRUE(std::isfinite(z.norm()));
}

// ── CG+AMGCL 迭代次数少于 CG（1D） ──────────────────

TEST(AMGCLTest, CG_vs_CGAMGCL_Poisson1D) {
    size_t N = 500;
    Vector b(N, 1.0);

    int iters_cg;
    {
        auto A = poisson1D(N);
        CGSolver cg;
        SolverConfig cfg;
        cfg.tol = 1e-8;
        cfg.max_iter = 5000;
        cfg.precond = nullptr;
        Vector x(N);
        auto res = cg.solve(A, b, x, cfg);
        iters_cg = res.iterations;
        EXPECT_TRUE(res.converged);
    }

    int iters_amgcl;
    {
        auto A = poisson1D(N);
        A.toCSR();
        AMGCLPreconditioner amgcl;
        amgcl.build(A);

        CGSolver cg;
        SolverConfig cfg;
        cfg.tol = 1e-8;
        cfg.max_iter = 5000;
        cfg.precond = &amgcl;
        Vector x(N);
        auto res = cg.solve(A, b, x, cfg);
        iters_amgcl = res.iterations;
        EXPECT_TRUE(res.converged);
    }

    EXPECT_LT(iters_amgcl, iters_cg);
}

// ── CG+AMGCL 迭代次数少于 CG（2D） ──────────────────

TEST(AMGCLTest, CG_vs_CGAMGCL_Poisson2D) {
    size_t n = 16, N = n * n;
    Vector b(N, 1.0);

    int iters_cg;
    {
        auto A = poisson2D(n);
        CGSolver cg;
        SolverConfig cfg;
        cfg.tol = 1e-8;
        cfg.max_iter = 10000;
        cfg.precond = nullptr;
        Vector x(N);
        auto res = cg.solve(A, b, x, cfg);
        iters_cg = res.iterations;
        EXPECT_TRUE(res.converged);
    }

    int iters_amgcl;
    {
        auto A = poisson2D(n);
        A.toCSR();
        AMGCLPreconditioner amgcl;
        amgcl.build(A);

        CGSolver cg;
        SolverConfig cfg;
        cfg.tol = 1e-8;
        cfg.max_iter = 10000;
        cfg.precond = &amgcl;
        Vector x(N);
        auto res = cg.solve(A, b, x, cfg);
        iters_amgcl = res.iterations;
        EXPECT_TRUE(res.converged);
    }

    EXPECT_LT(iters_amgcl, iters_cg);
}

// ── 小矩阵构建和应用 ──────────────────────────────────

TEST(AMGCLTest, SmallMatrix) {
    auto A = SparseMatrix::fromCOO(
        3, 3,
        {0,0,1,1,1,2,2},
        {0,1,0,1,2,1,2},
        {4,-1,-1,4,-1,-1,4});
    A.toCSR();

    AMGCLPreconditioner amgcl;
    EXPECT_NO_THROW(amgcl.build(A));

    Vector b = {1.0, 2.0, 3.0};
    Vector z = amgcl.apply(b);

    EXPECT_EQ(z.size(), 3);
    EXPECT_TRUE(std::isfinite(z.norm()));
}

// ── name() ──────────────────────────────────────────────

TEST(AMGCLTest, Name) {
    AMGCLPreconditioner amgcl;
    EXPECT_EQ(amgcl.name(), "AMGCL");
}

// ── 零向量 apply ──────────────────────────────────────

TEST(AMGCLTest, ApplyZeroVector) {
    auto A = poisson1D(64);
    A.toCSR();
    AMGCLPreconditioner amgcl;
    amgcl.build(A);

    Vector r(64, 0.0);
    Vector z = amgcl.apply(r);
    EXPECT_LT(z.norm(), 1e-12);  // M⁻¹ * 0 = 0
}