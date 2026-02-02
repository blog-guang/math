#pragma once
#include "solver.h"

namespace math {

/**
 * GMRES (Generalized Minimal Residual) 求解器。
 * 带重启策略 GMRES(m)。
 *
 * 适用于：一般非对称矩阵。
 * 原理：在 Krylov 子空间中找使残差 ||b-Ax|| 最小的解。
 * 使用 Arnoldi 过程构建正交基，Givens 旋转做最小二乘求解。
 *
 * 参数：
 *   config.gmres_restart = m  重启长度，默认 50。
 *   m 越大收敛越好但内存越多（存 m+1 个向量）。
 *   典型取值：20 ~ 100。
 *
 * 支持预条件（左预条件）：
 *   若 config.precond != nullptr，求解 M⁻¹Ax = M⁻¹b。
 */
class GMRESSolver : public Solver {
  public:
    SolveResult solve(SparseMatrix& A,
                      const Vector& b,
                      Vector& x,
                      const SolverConfig& config) override;

    [[nodiscard]] std::string name() const override { return "GMRES"; }
};

}  // namespace math
