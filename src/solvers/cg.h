#pragma once
#include "solver.h"

namespace math {

/**
 * CG (Conjugate Gradient) 共轭梯度法。
 *
 * 适用于：对称正定 (SPD) 矩阵。
 * 理论收敛：至多 N 步精确收敛（有限精度实际更少）。
 * 每步核心运算：1 次 SpMV + O(N) 向量运算。
 *
 * 算法：
 *   r = b - A*x,  p = r,  rsold = r·r
 *   loop:
 *     Ap = A*p
 *     α  = rsold / (p·Ap)
 *     x  = x + α*p
 *     r  = r - α*Ap
 *     rsnew = r·r
 *     if √rsnew / ||b|| < tol → 收敛
 *     β  = rsnew / rsold
 *     p  = r + β*p
 *     rsold = rsnew
 */
class CGSolver : public Solver {
  public:
    SolveResult solve(SparseMatrix& A,
                      const Vector& b,
                      Vector& x,
                      const SolverConfig& config) override;

    [[nodiscard]] std::string name() const override { return "CG"; }
};

}  // namespace math
