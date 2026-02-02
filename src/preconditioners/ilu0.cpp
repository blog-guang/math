#include "ilu0.h"

#include <stdexcept>
#include <algorithm>

namespace math {

/**
 * ILU(0) 分解核心：
 *
 * 在 A 的非零结构上原地做 LU 分解。
 * 对每一行 i：
 *   对 j < i 且 a[i][j] != 0（即 L 的非零位置）：
 *     L[i][j] = A[i][j] / U[j][j]      (乘子)
 *     对 k > j 且 a[i][k] != 0 且 a[j][k] != 0：
 *       A[i][k] -= L[i][j] * U[j][k]   (更新上三角)
 *
 * 关键：ILU(0) 只更新 A 中**原本就有**的非零位置，忽略 fill-in。
 * 这也是它叫"不完全"的原因——分解不精确，但结构不变，代价低。
 */
void ILU0Preconditioner::build(const SparseMatrix& A) {
    if (A.format() != StorageFormat::CSR) {
        throw std::runtime_error("ILU0::build: matrix must be in CSR format");
    }
    if (A.rows() != A.cols()) {
        throw std::runtime_error("ILU0::build: matrix must be square");
    }

    N_ = A.rows();
    row_ptr_ = A.csr_row_ptr();
    col_idx_ = A.csr_col_idx();
    lu_val_  = A.csr_val();         // 复制值数组，后续原地修改

    size_t nnz = lu_val_.size();

    // ── 预计算对角线位置 ──────────────────────────────
    diag_pos_.resize(N_);
    for (size_t i = 0; i < N_; ++i) {
        diag_pos_[i] = nnz;   // 哨兵：未找到
        for (size_t k = row_ptr_[i]; k < row_ptr_[i + 1]; ++k) {
            if (col_idx_[k] == i) {
                diag_pos_[i] = k;
                break;
            }
        }
        if (diag_pos_[i] == nnz) {
            throw std::runtime_error(
                "ILU0::build: zero/missing diagonal at row " + std::to_string(i));
        }
    }

    // ── ILU(0) 分解 ───────────────────────────────────
    for (size_t i = 0; i < N_; ++i) {
        // 遍历 row i 中 col < i 的元素（L 的非零位置）
        for (size_t ki = row_ptr_[i]; ki < row_ptr_[i + 1]; ++ki) {
            size_t j = col_idx_[ki];
            if (j >= i) continue;   // 只处理严格下三角

            // L[i][j] = A[i][j] / U[j][j]
            lu_val_[ki] /= lu_val_[diag_pos_[j]];

            // 更新 row i 中 col > j 的位置（与 row j 的上三角交集）
            // 用双指针遍历 row i 和 row j 的交集（col > j）
            size_t ptr_i = ki + 1;                    // row i 中 col > j 的起点
            size_t ptr_j = diag_pos_[j] + 1;          // row j 中 col > j 的起点（从对角线之后）

            while (ptr_i < row_ptr_[i + 1] &&
                   ptr_j < row_ptr_[j + 1]) {
                size_t ci = col_idx_[ptr_i];
                size_t cj = col_idx_[ptr_j];
                if (ci == cj) {
                    // 交集位置：更新
                    lu_val_[ptr_i] -= lu_val_[ki] * lu_val_[ptr_j];
                    ++ptr_i;
                    ++ptr_j;
                } else if (ci < cj) {
                    ++ptr_i;   // row i 有但 row j 没有 → fill-in，ILU(0) 跳过
                } else {
                    ++ptr_j;   // row j 有但 row i 没有 → 无交集
                }
            }
        }
    }
}

/**
 * apply(r) → z = (LU)⁻¹ * r
 *
 * 前代：L * y = r
 *   y[i] = r[i] - Σ L[i][j]*y[j]   (j < i, L[i][j] 已存为乘子，对角线=1)
 *
 * 回代：U * z = y
 *   z[i] = (y[i] - Σ U[i][j]*z[j]) / U[i][i]   (j > i)
 */
Vector ILU0Preconditioner::apply(const Vector& r) const {
    if (r.size() != N_) {
        throw std::invalid_argument("ILU0::apply: size mismatch");
    }

    // ── 前代求解 L*y = r ──────────────────────────────
    Vector y = r;
    for (size_t i = 0; i < N_; ++i) {
        // 遍历 row i 中 col < i（L 的严格下三角，已存乘子）
        for (size_t k = row_ptr_[i]; k < row_ptr_[i + 1]; ++k) {
            size_t j = col_idx_[k];
            if (j >= i) break;   // 假设列索引有序，到达对角线可停止
            y[i] -= lu_val_[k] * y[j];
        }
        // L 的对角线 = 1，无需除法
    }

    // ── 回代求解 U*z = y ──────────────────────────────
    Vector z = y;
    for (size_t i = N_; i > 0; ) {
        --i;
        // 遍历 row i 中 col > i（U 的严格上三角）
        for (size_t k = diag_pos_[i] + 1; k < row_ptr_[i + 1]; ++k) {
            size_t j = col_idx_[k];
            z[i] -= lu_val_[k] * z[j];
        }
        // 除以 U[i][i]
        z[i] /= lu_val_[diag_pos_[i]];
    }

    return z;
}

}  // namespace math
