#include "bicgstab.h"

#include <chrono>
#include <cmath>
#include <iostream>
#include <stdexcept>

namespace math {

SolveResult BiCGSTABSolver::solve(SparseMatrix& A,
                                  const Vector& b,
                                  Vector& x,
                                  const SolverConfig& config)
{
    auto t_start = std::chrono::steady_clock::now();

    if (A.rows() != A.cols()) {
        throw std::invalid_argument("BiCGSTAB: matrix must be square");
    }
    if (b.size() != A.rows() || x.size() != A.rows()) {
        throw std::invalid_argument("BiCGSTAB: vector size mismatch");
    }

    A.toCSR();
    const size_t N        = A.rows();
    const bool use_precond = (config.precond != nullptr);

    const double b_norm = b.norm();
    if (b_norm == 0.0) {
        x.zero();
        return packResult(true, 0, 0.0, 0.0, t_start);
    }

    // r = b - A*x
    Vector r = b - A.multiply(x);

    // r̂ = r（初始阴影残差，取 r 本身最优实践）
    Vector r_hat = r;

    double rho_old = 1.0;
    double alpha   = 1.0;
    double omega   = 1.0;

    // v = 0, p = 0
    Vector v(N, 0.0);
    Vector p(N, 0.0);

    if (config.verbose) logHeader("BiCGSTAB");

    for (int iter = 0; iter < config.max_iter; ++iter) {

        // ρ = r · r̂
        double rho = r.dot(r_hat);
        if (std::abs(rho) < detail::BREAKDOWN_TOL) {
            break;   // 方法失败：r 与 r̂ 正交
        }

        // β = (ρ / ρ_old) * (α / ω)
        double beta = (rho / rho_old) * (alpha / omega);

        // p = r + β * (p - ω * v)
        p *= beta;
        p.axpy(-beta * omega, v);   // p -= β*ω*v
        p += r;                      // p += r

        // v = A * p（或 M⁻¹*A*p）
        v = A.multiply(p);
        if (use_precond) v = config.precond->apply(v);

        // α = ρ / (r̂ · v)
        double r_hat_v = r_hat.dot(v);
        if (std::abs(r_hat_v) < detail::BREAKDOWN_TOL) break;   // 退化
        alpha = rho / r_hat_v;

        // s = r - α * v
        Vector s = r;
        s.axpy(-alpha, v);

        // 中间收敛检查（避免不必要的第二次 SpMV）
        double s_norm = s.norm();
        double s_rel  = s_norm / b_norm;

        if (config.verbose) logIter(iter + 1, s_norm, s_rel);

        if (s_rel < config.tol) {
            x.axpy(alpha, p);   // x += α * p
            return packResult(true, iter + 1, s_norm, b_norm, t_start);
        }

        // t = A * s（或 M⁻¹*A*s）
        Vector t = A.multiply(s);
        if (use_precond) t = config.precond->apply(t);

        // ω = (t · s) / (t · t)
        double t_dot_t = t.dot(t);
        if (std::abs(t_dot_t) < detail::BREAKDOWN_TOL) break;   // 退化
        omega = t.dot(s) / t_dot_t;

        // x += α*p + ω*s
        x.axpy(alpha, p);
        x.axpy(omega, s);

        // r = s - ω * t
        r = s;
        r.axpy(-omega, t);

        // 收敛检查
        double r_norm = r.norm();
        double r_rel  = r_norm / b_norm;

        if (r_rel < config.tol) {
            return packResult(true, iter + 1, r_norm, b_norm, t_start);
        }

        // ω 接近 0 → 方法失败
        if (std::abs(omega) < detail::BREAKDOWN_TOL) break;

        rho_old = rho;
    }

    // 未收敛
    double final_res = (b - A.multiply(x)).norm();
    return packResult(false, config.max_iter, final_res, b_norm, t_start);
}

}  // namespace math
