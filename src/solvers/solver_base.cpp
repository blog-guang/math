#include "solver.h"

#include <iomanip>
#include <iostream>
#include <stdexcept>

namespace math {

// ── 残差与对角线 ─────────────────────────────────────────

double Solver::computeResidual(SparseMatrix& A, const Vector& b, const Vector& x) {
    Vector r = b - A.multiply(x);
    return r.norm();
}

Vector Solver::extractDiagonal(const SparseMatrix& A) {
    Vector diag(A.rows());
    const auto& row_ptr = A.csr_row_ptr();
    const auto& col_idx = A.csr_col_idx();
    const auto& val     = A.csr_val();

    for (size_t i = 0; i < A.rows(); ++i) {
        diag[i] = 0.0;
        for (size_t k = row_ptr[i]; k < row_ptr[i + 1]; ++k) {
            if (col_idx[k] == i) {
                diag[i] = val[k];
                break;
            }
        }
    }
    return diag;
}

// ── 驻点迭代公共辅助 ────────────────────────────────────

std::optional<Solver::StationarySetup>
Solver::prepareStationary(SparseMatrix& A, const Vector& b, Vector& x,
                          const std::string& solver_name)
{
    if (A.rows() != A.cols()) {
        throw std::invalid_argument(solver_name + ": matrix must be square");
    }
    if (b.size() != A.rows() || x.size() != A.rows()) {
        throw std::invalid_argument(solver_name + ": vector size mismatch");
    }

    A.toCSR();

    StationarySetup setup{extractDiagonal(A), b.norm()};

    // 零对角检查
    for (size_t i = 0; i < A.rows(); ++i) {
        if (setup.diag[i] == 0.0) {
            throw std::runtime_error(
                solver_name + ": zero diagonal at row " + std::to_string(i));
        }
    }

    // b = 0 → 解为零向量，返回 nullopt 通知调用者
    if (setup.b_norm == 0.0) {
        x.zero();
        return std::nullopt;
    }

    return setup;
}

SolveResult Solver::packResult(bool converged, int iterations,
                               double residual, double b_norm,
                               std::chrono::steady_clock::time_point t_start)
{
    auto t_end = std::chrono::steady_clock::now();
    double rel_res = (b_norm > 0.0) ? residual / b_norm : 0.0;
    return {converged, iterations, residual, rel_res,
            std::chrono::duration<double, std::milli>(t_end - t_start).count()};
}

// ── Verbose 日志 ────────────────────────────────────────

void Solver::logHeader(const std::string& prefix) {
    if (!prefix.empty()) std::cout << prefix << ":\n";
    std::cout << std::setw(6)  << "Iter"
              << std::setw(14) << "Residual"
              << std::setw(14) << "Rel.Residual" << "\n";
}

void Solver::logIter(int iter, double residual, double rel_residual) {
    std::cout << std::setw(6)  << iter
              << std::setw(14) << std::scientific << residual
              << std::setw(14) << rel_residual << "\n";
}

}  // namespace math
