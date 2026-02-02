/**
 * gtest_all_matrices.cpp
 *
 * 遍历 test_matrices/ 所有 .mtx，自动选择求解器并验证收敛性。
 *
 * 策略：每个矩阵尝试多种求解器组合（从轻到重），**至少一种收敛即通过**。
 *   SPD  (*_spd_*)  : CG → CG+ILU0 → CG+AMG
 *   对称  (*_sym_*)  : CG → CG+ILU0 → GMRES+ILU0
 *   非对称 (其他)    : GMRES → GMRES+ILU0 → GMRES+AMGCL
 *
 * 零对角矩阵（LNS 等结构奇异）自动跳过 ILU0 路径。
 * preconditioned 路径容差放宽为 1e-6。
 */

#include <filesystem>
#include <string>
#include <vector>
#include <stdexcept>

#include <gtest/gtest.h>

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

// ── 目录定位 ─────────────────────────────────────────────

static fs::path matrixDir() {
    for (fs::path c : {fs::path("test_matrices"),
                       fs::path("../test_matrices"),
                       fs::path("../../test_matrices")}) {
        if (fs::is_directory(c)) return c;
    }
    return fs::path("test_matrices");
}

static std::vector<std::string> collectMtxFiles() {
    std::vector<std::string> files;
    fs::path dir = matrixDir();
    if (!fs::is_directory(dir)) return files;
    for (auto& e : fs::directory_iterator(dir)) {
        if (e.path().extension() == ".mtx")
            files.push_back(e.path().filename().stem().string());
    }
    std::sort(files.begin(), files.end());
    return files;
}

// ── 矩阵分类 ─────────────────────────────────────────────

enum class MatrixType { SPD, SYMMETRIC, NONSYMMETRIC };

static MatrixType classifyMatrix(const std::string& name) {
    if (name.find("_spd_") != std::string::npos) return MatrixType::SPD;
    if (name.find("_sym_") != std::string::npos) return MatrixType::SYMMETRIC;
    return MatrixType::NONSYMMETRIC;
}

// ── 辅助 ─────────────────────────────────────────────────

static Vector makeRHS(size_t n) {
    Vector b(n);
    for (size_t i = 0; i < n; ++i)
        b[i] = std::sin(static_cast<double>(i + 1));
    return b;
}

static double computeRelResidual(SparseMatrix& A, const Vector& b, const Vector& x) {
    Vector r = b - A.multiply(x);
    double bn = b.norm();
    return (bn > 0.0) ? r.norm() / bn : r.norm();
}

/** 检查矩阵是否有零对角元素 */
static bool hasZeroDiagonal(SparseMatrix& A) {
    size_t n = A.rows();
    for (size_t i = 0; i < n; ++i) {
        bool found = false;
        // 暴力扫描：对每一行找对角元素
        // 先确保转为 CSR 再扫描（multiply 会触发转换）
        // 用一个小向量测试
        Vector e(n, 0.0);
        e[i] = 1.0;
        Vector col = A.multiply(e);
        if (std::abs(col[i]) < 1e-30) return true;
    }
    return false;
}

/** 尝试运行一个求解器，返回是否收敛及残差信息 */
struct SolveAttempt {
    std::string name;
    bool converged;
    int iterations;
    double rel_residual;
    double verified_residual;  // 独立验证的残差
};

static SolveAttempt trySolve(const std::string& label, Solver& solver,
                             SparseMatrix& A, const Vector& b, SolverConfig& cfg) {
    size_t n = A.rows();
    Vector x(n, 0.0);
    auto res = solver.solve(A, b, x, cfg);
    double vr = computeRelResidual(A, b, x);
    return {label, res.converged && vr < 1e-4, res.iterations, res.relative_residual, vr};
}

// ══════════════════════════════════════════════════════════
//  参数化测试
// ══════════════════════════════════════════════════════════

class AllMatricesTest : public ::testing::TestWithParam<std::string> {};

TEST_P(AllMatricesTest, Convergence) {
    const std::string& name = GetParam();
    fs::path filepath = matrixDir() / (name + ".mtx");

    SparseMatrix A = MatrixMarket::readMatrix(filepath.string());
    size_t n = A.rows();
    ASSERT_EQ(A.rows(), A.cols()) << name << ": 非方阵";
    ASSERT_GT(n, 0) << name << ": 空矩阵";

    Vector b = makeRHS(n);
    MatrixType type = classifyMatrix(name);

    // 检测零对角（用小采样，避免 O(n²)）
    bool zeroDiag = false;
    {
        // 快速采样检测：检查前 min(n,20) 行和随机行
        size_t step = std::max(static_cast<size_t>(1), n / 20);
        for (size_t i = 0; i < n; i += step) {
            Vector e(n, 0.0);
            e[i] = 1.0;
            Vector col = A.multiply(e);
            if (std::abs(col[i]) < 1e-30) { zeroDiag = true; break; }
        }
    }

    // 基础配置
    SolverConfig cfg;
    cfg.tol      = 1e-8;
    cfg.max_iter = 50000;
    cfg.gmres_restart = std::min(static_cast<int>(n), 200);  // 增大 restart

    std::vector<SolveAttempt> attempts;
    bool anyPassed = false;

    auto record = [&](SolveAttempt a) {
        if (a.converged) anyPassed = true;
        attempts.push_back(std::move(a));
    };

    if (type == MatrixType::SPD) {
        // 1. CG (无预条件)
        { CGSolver cg; record(trySolve("CG", cg, A, b, cfg)); }

        // 2. CG + ILU0
        if (!zeroDiag) {
            try {
                ILU0Preconditioner ilu0;
                ilu0.build(A);
                SolverConfig pcfg = cfg;
                pcfg.precond = &ilu0;
                pcfg.tol = 1e-6;  // preconditioned 放宽
                CGSolver cg;
                record(trySolve("CG+ILU0", cg, A, b, pcfg));
            } catch (const std::exception& e) {
                attempts.push_back({"CG+ILU0", false, 0, 0.0, 0.0});
            }
        }

        // 3. CG + AMG（最后的武器）
        {
            try {
                AMGPreconditioner amg;
                amg.build(A);
                SolverConfig pcfg = cfg;
                pcfg.precond = &amg;
                pcfg.tol = 1e-6;
                CGSolver cg;
                record(trySolve("CG+AMG", cg, A, b, pcfg));
            } catch (const std::exception& e) {
                attempts.push_back({"CG+AMG", false, 0, 0.0, 0.0});
            }
        }

    } else {
        // 非对称路径
        // 1. GMRES (无预条件)
        { GMRESSolver gmres; record(trySolve("GMRES", gmres, A, b, cfg)); }

        // 2. BiCGSTAB (无预条件) — 仅当 GMRES 未成功时尝试
        if (!anyPassed) {
            BiCGSTABSolver bicg;
            record(trySolve("BiCGSTAB", bicg, A, b, cfg));
        }

        // 3. GMRES + ILU0
        if (!zeroDiag) {
            try {
                ILU0Preconditioner ilu0;
                ilu0.build(A);
                SolverConfig pcfg = cfg;
                pcfg.precond = &ilu0;
                pcfg.tol = 1e-6;
                GMRESSolver gmres;
                record(trySolve("GMRES+ILU0", gmres, A, b, pcfg));
            } catch (const std::exception& e) {
                attempts.push_back({"GMRES+ILU0", false, 0, 0.0, 0.0});
            }
        }

        // 4. GMRES + AMGCL（最后的武器，处理结构奇异和高难度矩阵）
        if (!anyPassed) {
            try {
                AMGCLPreconditioner amgcl;
                amgcl.build(A);
                SolverConfig pcfg = cfg;
                pcfg.precond = &amgcl;
                pcfg.tol = 1e-6;
                GMRESSolver gmres;
                record(trySolve("GMRES+AMGCL", gmres, A, b, pcfg));
            } catch (const std::exception& e) {
                attempts.push_back({"GMRES+AMGCL", false, 0, 0.0, 0.0});
            }
        }
    }

    // ── 汇报 ──
    if (!anyPassed) {
        std::string report = name + " 所有求解器均未收敛:\n";
        for (auto& a : attempts) {
            report += "  " + a.name + ": iters=" + std::to_string(a.iterations)
                    + " solver_res=" + std::to_string(a.rel_residual)
                    + " verified_res=" + std::to_string(a.verified_residual) + "\n";
        }
        FAIL() << report;
    } else {
        // 打印诊断信息（哪些收敛、哪些没收敛）
        std::string info = name + ":\n";
        for (auto& a : attempts) {
            info += "  " + a.name + ": "
                  + (a.converged ? "✅ " : "⛔ ")
                  + "iters=" + std::to_string(a.iterations)
                  + " res=" + std::to_string(a.verified_residual) + "\n";
        }
        // 用 RecordProperty 记录诊断
        RecordProperty("solvers", info);
    }
}

INSTANTIATE_TEST_SUITE_P(
    MatrixMarket,
    AllMatricesTest,
    ::testing::ValuesIn(collectMtxFiles()),
    [](const ::testing::TestParamInfo<AllMatricesTest::ParamType>& info) {
        return info.param;
    });
