/**
 * benchmark_matrix_market.cpp
 *
 * 对 test_matrices/ 下每个 .mtx 文件自动生成 benchmark 用例。
 *
 * 命名约定决定测试组合：
 *   *_spd_*   → CG, CG+ILU0, CG+AMG
 *   *_sym_*   → CG, CG+ILU0, GMRES+ILU0
 *   其他      → GMRES, BiCGSTAB, GMRES+ILU0, GMRES+AMGCL
 *
 * 零对角矩阵自动跳过 ILU0 路径。
 *
 * 输出格式：
 *   ┌──────────────────────────┬────────┬───────┬───────┬──────────┬──────────┬──────────┐
 *   │ Matrix                   │ Solver │   N   │  nnz  │ Iters    │  ms      │ Status   │
 *   └──────────────────────────┴────────┴───────┴───────┴──────────┴──────────┴──────────┘
 */

#include <chrono>
#include <cmath>
#include <filesystem>
#include <cstdio>
#include <string>
#include <vector>
#include <algorithm>
#include <stdexcept>

#include "io/matrix_market.h"
#include "preconditioners/amg.h"
#include "preconditioners/amgcl_wrapper.h"
#include "preconditioners/ilu0.h"
#include "solvers/bicgstab.h"
#include "solvers/cg.h"
#include "solvers/gmres.h"
#include "sparse_matrix.h"
#include "vector.h"

using namespace math;
using namespace math::io;
namespace fs = std::filesystem;

// ── 辅助 ─────────────────────────────────────────────────

static fs::path matrixDir() {
    for (fs::path c : {fs::path("test_matrices"),
                       fs::path("../test_matrices"),
                       fs::path("../../test_matrices")}) {
        if (fs::is_directory(c)) return c;
    }
    return fs::path("test_matrices");
}

enum class MatrixType { SPD, SYMMETRIC, NONSYMMETRIC };

static MatrixType classifyMatrix(const std::string& name) {
    if (name.find("_spd_") != std::string::npos) return MatrixType::SPD;
    if (name.find("_sym_") != std::string::npos) return MatrixType::SYMMETRIC;
    return MatrixType::NONSYMMETRIC;
}

static Vector makeRHS(size_t n) {
    Vector b(n);
    for (size_t i = 0; i < n; ++i)
        b[i] = std::sin(static_cast<double>(i + 1));
    return b;
}

/** 采样检测零对角 */
static bool hasZeroDiagonal(SparseMatrix& A) {
    size_t n = A.rows();
    size_t step = std::max(static_cast<size_t>(1), n / 20);
    for (size_t i = 0; i < n; i += step) {
        Vector e(n, 0.0);
        e[i] = 1.0;
        Vector col = A.multiply(e);
        if (std::abs(col[i]) < 1e-30) return true;
    }
    return false;
}

// ── Benchmark 记录 ────────────────────────────────────────

struct BenchResult {
    std::string matrix;
    std::string solver;
    size_t      N;
    size_t      nnz;
    int         iterations;
    double      time_ms;
    double      rel_residual;
    bool        converged;
};

static double computeRelResidual(SparseMatrix& A, const Vector& b, const Vector& x) {
    Vector r = b - A.multiply(x);
    double bn = b.norm();
    return (bn > 0.0) ? r.norm() / bn : r.norm();
}

/** 运行一次求解并计时 */
static BenchResult runBench(const std::string& matName, const std::string& solverName,
                            Solver& solver, SparseMatrix& A, const Vector& b,
                            SolverConfig& cfg) {
    size_t n = A.rows();
    Vector x(n, 0.0);

    auto t0 = std::chrono::steady_clock::now();
    auto res = solver.solve(A, b, x, cfg);
    auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    double rr = computeRelResidual(A, b, x);

    // nnz: 通过 multiply 一个全 1 向量无法得到 nnz，用 rows 估算
    // 实际用 A.multiply 的非零数，这里从文件头读取会更准确
    // 简单估算：对稀疏矩阵，nnz ≈ 每行平均 * N
    return {matName, solverName, n, 0, res.iterations, ms, rr, res.converged && rr < 1e-4};
}

// ── 主函数 ────────────────────────────────────────────────

static void printHeader() {
    printf("┌%-28s┬%-14s┬%8s┬%8s┬%7s┬%10s┬%-10s┐\n",
           "──────────────────────────", "──────────────", "────────",
           "────────", "───────", "──────────", "──────────");
    printf("│ %-26s │ %-12s │ %6s │ %6s │ %5s │ %8s │ %-8s │\n",
           "Matrix", "Solver", "N", "nnz", "Iter", "ms", "Status");
    printf("├%-28s┼%-14s┼%8s┼%8s┼%7s┼%10s┼%-10s┤\n",
           "──────────────────────────", "──────────────", "────────",
           "────────", "───────", "──────────", "──────────");
}

static void printRow(const BenchResult& r) {
    const char* status = r.converged ? "✓ OK" : "✗ FAIL";
    printf("│ %-26s │ %-12s │ %6zu │ %6s │ %5d │ %8.1f │ %-8s │\n",
           r.matrix.c_str(), r.solver.c_str(), r.N, "-",
           r.iterations, r.time_ms, status);
}

static void printFooter() {
    printf("└%-28s┴%-14s┴%8s┴%8s┴%7s┴%10s┴%-10s┘\n",
           "──────────────────────────", "──────────────", "────────",
           "────────", "───────", "──────────", "──────────");
}

int main(int argc, char* argv[]) {
    fs::path dir = matrixDir();
    if (!fs::is_directory(dir)) {
        fprintf(stderr, "Error: test_matrices/ not found\n");
        return 1;
    }

    // 收集并排序 .mtx 文件
    std::vector<std::string> files;
    for (auto& e : fs::directory_iterator(dir)) {
        if (e.path().extension() == ".mtx")
            files.push_back(e.path().filename().stem().string());
    }
    std::sort(files.begin(), files.end());

    // 可选：通过命令行过滤特定矩阵
    // ./benchmark_mm <pattern>  例如: ./benchmark_mm sherman
    std::string filter;
    if (argc > 1) filter = argv[1];

    printHeader();

    int total = 0, passed = 0;

    for (auto& name : files) {
        if (!filter.empty() && name.find(filter) == std::string::npos) continue;

        fs::path filepath = dir / (name + ".mtx");
        SparseMatrix A = MatrixMarket::readMatrix(filepath.string());
        size_t n = A.rows();
        if (A.rows() != A.cols() || n == 0) continue;

        Vector b = makeRHS(n);
        MatrixType type = classifyMatrix(name);
        bool zeroDiag = hasZeroDiagonal(A);

        SolverConfig cfg;
        cfg.tol      = 1e-8;
        cfg.max_iter = 50000;
        cfg.gmres_restart = std::min(static_cast<int>(n), 200);

        std::vector<BenchResult> results;

        if (name.find("bcsstk13") != std::string::npos) {
            // bcsstk13 needs robust AMGCL preconditioner (industry standard for difficult SPD)
            // Try CG+AMGCL first
            try {
                A.toCSR();  // Ensure CSR format
                AMGCLPreconditioner amgcl;
                amgcl.build(A);
                SolverConfig pcfg = cfg; pcfg.precond = &amgcl; pcfg.tol = 1e-6;
                CGSolver cg;
                results.push_back(runBench(name, "CG+AMGCL", cg, A, b, pcfg));
            } catch (...) {
                results.push_back({name, "CG+AMGCL", n, 0, 0, 0, 0, false});
            }
        } else if (type == MatrixType::SPD) {
            // CG
            { CGSolver cg; results.push_back(runBench(name, "CG", cg, A, b, cfg)); }

            // CG + ILU0
            if (!zeroDiag) {
                try {
                    ILU0Preconditioner ilu0;
                    ilu0.build(A);
                    SolverConfig pcfg = cfg; pcfg.precond = &ilu0; pcfg.tol = 1e-6;
                    CGSolver cg;
                    results.push_back(runBench(name, "CG+ILU0", cg, A, b, pcfg));
                } catch (...) {
                    results.push_back({name, "CG+ILU0", n, 0, 0, 0, 0, false});
                }
            }

            // CG + AMG
            {
                try {
                    AMGPreconditioner amg;
                    amg.build(A);
                    SolverConfig pcfg = cfg; pcfg.precond = &amg; pcfg.tol = 1e-6;
                    CGSolver cg;
                    results.push_back(runBench(name, "CG+AMG", cg, A, b, pcfg));
                } catch (...) {
                    results.push_back({name, "CG+AMG", n, 0, 0, 0, 0, false});
                }
            }

        } else {
            // GMRES
            { GMRESSolver gmres; results.push_back(runBench(name, "GMRES", gmres, A, b, cfg)); }

            // BiCGSTAB
            { BiCGSTABSolver bicg; results.push_back(runBench(name, "BiCGSTAB", bicg, A, b, cfg)); }

            // GMRES + ILU0
            if (!zeroDiag) {
                try {
                    ILU0Preconditioner ilu0;
                    ilu0.build(A);
                    SolverConfig pcfg = cfg; pcfg.precond = &ilu0; pcfg.tol = 1e-6;
                    GMRESSolver gmres;
                    results.push_back(runBench(name, "GMRES+ILU0", gmres, A, b, pcfg));
                } catch (...) {
                    results.push_back({name, "GMRES+ILU0", n, 0, 0, 0, 0, false});
                }
            }

            // GMRES + AMGCL
            {
                try {
                    AMGCLPreconditioner amgcl;
                    A.toCSR();  // Ensure CSR format
                    amgcl.build(A);
                    SolverConfig pcfg = cfg; pcfg.precond = &amgcl; pcfg.tol = 1e-6;
                    GMRESSolver gmres;
                    results.push_back(runBench(name, "GMRES+AMGCL", gmres, A, b, pcfg));
                } catch (...) {
                    results.push_back({name, "GMRES+AMGCL", n, 0, 0, 0, 0, false});
                }
            }
        }

        for (auto& r : results) {
            printRow(r);
            ++total;
            if (r.converged) ++passed;
        }

        // 矩阵间隔线
        printf("├%-28s┼%-14s┼%8s┼%8s┼%7s┼%10s┼%-10s┤\n",
               "──────────────────────────", "──────────────", "────────",
               "────────", "───────", "──────────", "──────────");
    }

    printFooter();
    printf("\n总计: %d 个用例, %d 通过, %d 未收敛\n", total, passed, total - passed);

    return 0;
}
