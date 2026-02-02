#include <gtest/gtest.h>
#include "preconditioners/jacobi_precond.h"
#include "solvers/cg.h"
#include "sparse_matrix.h"
#include "vector.h"

using namespace math;

static SparseMatrix makeTridiagonal(size_t N) {
    std::vector<size_t> rows, cols;
    std::vector<double> vals;
    for (size_t i = 0; i < N; ++i) {
        rows.push_back(i); cols.push_back(i); vals.push_back(4.0);
        if (i > 0)   { rows.push_back(i); cols.push_back(i-1); vals.push_back(-1.0); }
        if (i+1 < N) { rows.push_back(i); cols.push_back(i+1); vals.push_back(-1.0); }
    }
    return SparseMatrix::fromCOO(N, N, rows, cols, vals);
}

static SparseMatrix makeIllConditioned(size_t N) {
    std::vector<size_t> rows, cols;
    std::vector<double> vals;
    for (size_t i = 0; i < N; ++i) {
        rows.push_back(i); cols.push_back(i); vals.push_back(2.0 * (i + 1));
        if (i > 0)   { rows.push_back(i); cols.push_back(i-1); vals.push_back(-1.0); }
        if (i+1 < N) { rows.push_back(i); cols.push_back(i+1); vals.push_back(-1.0); }
    }
    return SparseMatrix::fromCOO(N, N, rows, cols, vals);
}

// ── Jacobi 预条件基本正确性 ──────────────────────────

TEST(JacobiPrecondTest, DiagonalExact) {
    // M⁻¹ * [2, 8, 18] = [1, 2, 3] when diag = [2, 4, 6]
    Vector d = {2.0, 4.0, 6.0};
    auto D = SparseMatrix::diagonal(d);
    D.toCSR();

    JacobiPreconditioner precond;
    precond.build(D);

    Vector r = {2.0, 8.0, 18.0};
    Vector z = precond.apply(r);
    EXPECT_TRUE(z.approx_equal(Vector{1.0, 2.0, 3.0}));
}

// ── 三对角矩阵上 ──────────────────────────────────────

TEST(JacobiPrecondTest, Tridiagonal) {
    // 对角=4 → M⁻¹ = diag(1/4)
    size_t N = 3;
    auto A = makeTridiagonal(N);
    A.toCSR();

    JacobiPreconditioner precond;
    precond.build(A);

    Vector r = {4.0, 8.0, 12.0};
    Vector z = precond.apply(r);
    EXPECT_TRUE(z.approx_equal(Vector{1.0, 2.0, 3.0}));
}

// ── PCG 与 CG 解一致 ────────────────────────────────────

TEST(JacobiPrecondTest, PCGSameSolutionAsCG) {
    size_t N = 50;
    auto A_cg  = makeTridiagonal(N);
    auto A_pcg = makeTridiagonal(N);
    Vector b(N, 1.0);

    SolverConfig cfg;
    cfg.tol = 1e-10;
    cfg.max_iter = 500;

    CGSolver cg;

    Vector x_cg(N);
    cfg.precond = nullptr;
    auto r_cg = cg.solve(A_cg, b, x_cg, cfg);

    A_pcg.toCSR();
    JacobiPreconditioner precond;
    precond.build(A_pcg);
    cfg.precond = &precond;

    Vector x_pcg(N);
    auto r_pcg = cg.solve(A_pcg, b, x_pcg, cfg);

    EXPECT_TRUE(r_cg.converged);
    EXPECT_TRUE(r_pcg.converged);
    EXPECT_TRUE(x_cg.approx_equal(x_pcg, 1e-8));
}

// ── 条件数大矩阵上 PCG 显著加速 ──────────────────────

TEST(JacobiPrecondTest, AccelerationIllConditioned) {
    size_t N = 200;
    auto A_cg  = makeIllConditioned(N);
    auto A_pcg = makeIllConditioned(N);
    Vector b(N, 1.0);

    SolverConfig cfg;
    cfg.tol = 1e-10;
    cfg.max_iter = 5000;

    CGSolver cg;

    Vector x_cg(N);
    cfg.precond = nullptr;
    auto r_cg = cg.solve(A_cg, b, x_cg, cfg);

    A_pcg.toCSR();
    JacobiPreconditioner precond;
    precond.build(A_pcg);
    cfg.precond = &precond;

    Vector x_pcg(N);
    auto r_pcg = cg.solve(A_pcg, b, x_pcg, cfg);

    EXPECT_TRUE(r_cg.converged);
    EXPECT_TRUE(r_pcg.converged);
    EXPECT_LT(r_pcg.iterations, r_cg.iterations);
    EXPECT_TRUE(x_cg.approx_equal(x_pcg, 1e-7));
}

// ── 大规模 PCG ──────────────────────────────────────────

TEST(JacobiPrecondTest, LargeSystem) {
    size_t N = 1000;
    auto A = makeIllConditioned(N);
    A.toCSR();

    JacobiPreconditioner precond;
    precond.build(A);

    Vector b(N, 1.0);
    Vector x(N);

    SolverConfig cfg;
    cfg.tol = 1e-10;
    cfg.max_iter = 5000;
    cfg.precond = &precond;

    CGSolver cg;
    auto result = cg.solve(A, b, x, cfg);

    EXPECT_TRUE(result.converged);
    EXPECT_LT(result.relative_residual, 1e-10);

    Vector r = b - A.multiply(x);
    EXPECT_LT(r.norm(), 1e-8);
}

// ── 对角矩阵 + Jacobi = M⁻¹A = I → 1 步 ───────────

// ── name() ──────────────────────────────────────────────

TEST(JacobiPrecondTest, Name) {
    JacobiPreconditioner p;
    EXPECT_EQ(p.name(), "Jacobi");
}

// ── 零向量 apply ──────────────────────────────────────

TEST(JacobiPrecondTest, ApplyZeroVector) {
    Vector d = {2.0, 4.0, 6.0};
    auto D = SparseMatrix::diagonal(d);
    D.toCSR();

    JacobiPreconditioner precond;
    precond.build(D);

    Vector r = {0.0, 0.0, 0.0};
    Vector z = precond.apply(r);
    EXPECT_TRUE(z.approx_equal(Vector::zeros(3)));
}

TEST(JacobiPrecondTest, DiagonalPCGOneStep) {
    Vector d = {3.0, 7.0, 2.0, 5.0};
    auto D = SparseMatrix::diagonal(d);
    D.toCSR();

    JacobiPreconditioner precond;
    precond.build(D);

    Vector b = {9.0, 21.0, 6.0, 25.0};
    Vector x(4);

    SolverConfig cfg;
    cfg.tol = 1e-12;
    cfg.precond = &precond;

    CGSolver cg;
    auto result = cg.solve(D, b, x, cfg);

    EXPECT_TRUE(result.converged);
    EXPECT_EQ(result.iterations, 1);
    EXPECT_TRUE(x.approx_equal(Vector{3.0, 3.0, 3.0, 5.0}));
}
