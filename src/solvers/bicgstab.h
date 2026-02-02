#pragma once
#include "solver.h"

namespace math {

/**
 * BiCGSTAB (Biconjugate Gradient Stabilized)。
 *
 * 适用于：一般非对称矩阵。
 * 每步：2 次 SpMV（比 GMRES(m) 每步仅 1 次多，但无需存储 Krylov 基）。
 * 内存：O(N)（固定，不随迭代增长）。
 * 实践中收敛比 GMRES 更稳定，适合大规模稀疏系统。
 *
 * 支持预条件（左预条件）：config.precond != nullptr。
 *
 * 算法（标准版）：
 *   r = b - A*x,  r̂ = r,  ρ₀ = 1, α = 1, ω = 1
 *   v = 0, p = 0
 *   loop:
 *     ρ = r · r̂
 *     β = (ρ/ρ₀) * (α/ω)
 *     p = r + β*(p - ω*v)
 *     v = A*p  (or M⁻¹*A*p)
 *     α = ρ / (r̂ · v)
 *     s = r - α*v
 *     if ||s||/||b|| < tol → x += α*p; done
 *     t = A*s  (or M⁻¹*A*s)
 *     ω = (t·s) / (t·t)
 *     x += α*p + ω*s
 *     r = s - ω*t
 *     if ||r||/||b|| < tol → done
 *     ρ₀ = ρ
 */
class BiCGSTABSolver : public Solver {
  public:
    SolveResult solve(SparseMatrix& A,
                      const Vector& b,
                      Vector& x,
                      const SolverConfig& config) override;

    [[nodiscard]] std::string name() const override { return "BiCGSTAB"; }
};

}  // namespace math
