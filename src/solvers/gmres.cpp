#include "gmres.h"

#include <chrono>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace math {

/**
 * GMRES(m) with Givens rotations + optional left preconditioning.
 *
 * Outer loop: restart
 *   r = b - A*x (or M⁻¹*(b - A*x) if preconditioned)
 *   β = ||r||, V[0] = r/β
 *
 *   Inner loop (Arnoldi, j = 0..m-1):
 *     w = A*V[j]  (或 M⁻¹*A*V[j])
 *     Modified Gram-Schmidt 正交化
 *     H[j+1][j] = ||w||, V[j+1] = w / H[j+1][j]
 *     Apply previous Givens rotations to H column
 *     Compute & apply new Givens rotation
 *     Check convergence via |g[j+1]|/||b||
 *
 *   Solve upper triangular R*y = g (back substitution)
 *   x = x + V * y
 */
SolveResult GMRESSolver::solve(SparseMatrix& A,
                               const Vector& b,
                               Vector& x,
                               const SolverConfig& config)
{
    auto t_start = std::chrono::steady_clock::now();

    if (A.rows() != A.cols()) {
        throw std::invalid_argument("GMRES: matrix must be square");
    }
    if (b.size() != A.rows() || x.size() != A.rows()) {
        throw std::invalid_argument("GMRES: vector size mismatch");
    }

    A.toCSR();
    const size_t N = A.rows();

    int m = config.gmres_restart;
    if (m <= 0) m = 50;   // 默认重启长度
    if (m > static_cast<int>(N)) m = static_cast<int>(N);

    const bool use_precond = (config.precond != nullptr);

    const double b_norm = b.norm();
    if (b_norm == 0.0) {
        x.zero();
        return packResult(true, 0, 0.0, 0.0, t_start);
    }

    int total_iters = 0;

    if (config.verbose) {
        std::cout << "GMRES(m=" << m << "):\n";
        logHeader();
    }

    // ── 外层：重启循环 ──────────────────────────────────
    for (;;) {

        // r = b - A*x (左预条件时叠加 M⁻¹)
        Vector r = b - A.multiply(x);
        if (use_precond) r = config.precond->apply(r);

        double beta    = r.norm();
        double rel_res = beta / b_norm;

        if (config.verbose) logIter(total_iters, beta, rel_res);

        if (rel_res < config.tol) {
            return packResult(true, total_iters, beta, b_norm, t_start);
        }

        // Krylov 基向量 V[0..m]
        std::vector<Vector> V;
        V.reserve(m + 1);
        V.emplace_back(r * (1.0 / beta));

        // Hessenberg 矩阵 H: (m+1) × m
        std::vector<std::vector<double>> H(m + 1, std::vector<double>(m, 0.0));

        // 最小二乘右端 g = β*e1
        std::vector<double> g(m + 1, 0.0);
        g[0] = beta;

        // Givens 旋转系数
        std::vector<double> cs(m, 0.0), sn(m, 0.0);

        int  j                = 0;    // 内层迭代数
        bool inner_converged  = false;

        // ── 内层：Arnoldi 迭代 ──────────────────────────
        for (j = 0; j < m; ++j) {

            Vector w = A.multiply(V[j]);
            if (use_precond) w = config.precond->apply(w);

            // Modified Gram-Schmidt 正交化
            for (int i = 0; i <= j; ++i) {
                H[i][j] = w.dot(V[i]);
                w.axpy(-H[i][j], V[i]);   // w -= H[i][j] * V[i]
            }
            H[j + 1][j] = w.norm();

            // Arnoldi 断裂：精确解在当前 Krylov 子空间中
            if (H[j + 1][j] < detail::ARNOLDI_TOL) {
                j++;   // j 现在等于使用的列数
                break;
            }

            V.emplace_back(w * (1.0 / H[j + 1][j]));

            // Apply 之前的 Givens 旋转到 H 的第 j 列
            for (int i = 0; i < j; ++i) {
                double h_ij  = H[i][j];
                double h_i1j = H[i + 1][j];
                H[i][j]     =  cs[i] * h_ij + sn[i] * h_i1j;
                H[i + 1][j] = -sn[i] * h_ij + cs[i] * h_i1j;
            }

            // 计算新的 Givens 旋转消去 H[j+1][j]
            double denom = std::sqrt(H[j][j] * H[j][j] + H[j + 1][j] * H[j + 1][j]);
            cs[j] = H[j][j]     / denom;
            sn[j] = H[j + 1][j] / denom;

            // Apply 新旋转到 H 和 g
            H[j][j]     =  cs[j] * H[j][j] + sn[j] * H[j + 1][j];
            H[j + 1][j] = 0.0;

            double g_j  = g[j];
            double g_j1 = g[j + 1];
            g[j]     =  cs[j] * g_j  + sn[j] * g_j1;
            g[j + 1] = -sn[j] * g_j  + cs[j] * g_j1;

            total_iters++;

            // 收敛检查：|g[j+1]| 即当前残差估计
            double cur_res = std::abs(g[j + 1]);
            double cur_rel = cur_res / b_norm;

            if (config.verbose) logIter(total_iters, cur_res, cur_rel);

            if (cur_rel < config.tol) {
                inner_converged = true;
                j++;   // j 现在等于使用的列数
                break;
            }

            if (total_iters >= config.max_iter) break;
        }

        // ── 求解上三角系统 R*y = g ──────────────────────
        const int k = j;   // 实际使用的列数
        std::vector<double> y(k, 0.0);

        for (int i = k - 1; i >= 0; --i) {
            y[i] = g[i];
            for (int l = i + 1; l < k; ++l) {
                y[i] -= H[i][l] * y[l];
            }
            if (std::abs(H[i][i]) < detail::SINGULAR_TOL) {
                y[i] = 0.0;   // 退化情况
            } else {
                y[i] /= H[i][i];
            }
        }

        // ── 更新解：x = x + V * y ──────────────────────
        for (int i = 0; i < k; ++i) {
            x.axpy(y[i], V[i]);
        }

        // ── 判断是否全局收敛 ──────────────────────────
        if (inner_converged) {
            // 用精确残差验证（Givens 残差估计可能因预条件偏离）
            Vector r_final  = b - A.multiply(x);
            double final_res = r_final.norm();
            return packResult(final_res / b_norm < config.tol,
                              total_iters, final_res, b_norm, t_start);
        }

        // 超过 max_iter → 退出重启循环
        if (total_iters >= config.max_iter) break;

        // 否则继续重启
    }

    // 未收敛：计算精确残差
    double final_res = (b - A.multiply(x)).norm();
    return packResult(false, total_iters, final_res, b_norm, t_start);
}

}  // namespace math
