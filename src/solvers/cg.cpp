#include "cg.h"

#include <chrono>
#include <iostream>
#include <stdexcept>

namespace math {

/**
 * CG / PCG 统一实现。
 *
 * 无预条件（config.precond == nullptr）：标准 CG
 *   内积用 r·r，搜索方向 p 从 r 开始。
 *
 * 有预条件（config.precond != nullptr）：PCG
 *   每步额外做 z = M⁻¹*r，内积换为 r·z，搜索方向从 z 开始。
 *   PCG 算法：
 *     r = b - A*x
 *     z = M⁻¹*r,  p = z,  rz = r·z
 *     loop:
 *       Ap = A*p
 *       α  = rz / (p·Ap)
 *       x += α*p
 *       r -= α*Ap
 *       if ||r||/||b|| < tol → done
 *       z  = M⁻¹*r
 *       rz_new = r·z
 *       β  = rz_new / rz
 *       p  = z + β*p
 *       rz = rz_new
 */
SolveResult CGSolver::solve(SparseMatrix& A,
                            const Vector& b,
                            Vector& x,
                            const SolverConfig& config)
{
    auto t_start = std::chrono::steady_clock::now();

    if (A.rows() != A.cols()) {
        throw std::invalid_argument("CG: matrix must be square");
    }
    if (b.size() != A.rows() || x.size() != A.rows()) {
        throw std::invalid_argument("CG: vector size mismatch");
    }

    A.toCSR();

    const double b_norm = b.norm();
    if (b_norm == 0.0) {
        x.zero();
        return packResult(true, 0, 0.0, 0.0, t_start);
    }

    const bool use_precond = (config.precond != nullptr);

    // r = b - A*x
    Vector r = b - A.multiply(x);

    // z = M⁻¹*r （无预条件时 z = r）
    Vector z = use_precond ? config.precond->apply(r) : r;

    // p = z
    Vector p = z;

    // rz = r·z （无预条件时即 r·r）
    double rz = r.dot(z);

    if (config.verbose) logHeader(use_precond ? "PCG" : "CG");

    for (int iter = 0; iter < config.max_iter; ++iter) {
        Vector Ap = A.multiply(p);

        // α = rz / (p·Ap)
        double pAp = p.dot(Ap);
        if (pAp == 0.0) break;   // 精确解或奇异

        double alpha = rz / pAp;

        x.axpy(alpha, p);       // x += α*p
        r.axpy(-alpha, Ap);     // r -= α*Ap

        // 收敛检查（用 ||r|| 而非 ||r·z||，保证残差语义清晰）
        double res     = r.norm();
        double rel_res = res / b_norm;

        if (config.verbose) logIter(iter, res, rel_res);

        if (rel_res < config.tol) {
            return packResult(true, iter + 1, res, b_norm, t_start);
        }

        // z = M⁻¹*r
        z = use_precond ? config.precond->apply(r) : r;

        double rz_new = r.dot(z);
        double beta   = rz_new / rz;

        // p = z + β*p
        p *= beta;
        p += z;

        rz = rz_new;
    }

    return packResult(false, config.max_iter, r.norm(), b_norm, t_start);
}

}  // namespace math
