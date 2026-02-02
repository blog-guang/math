#pragma once
#include "preconditioner.h"

namespace math {

/**
 * ILU(0) — 不完全 LU 分解预条件器。
 *
 * 在矩阵 A 的**原有非零结构**上做 LU 分解（不产生 fill-in）。
 * 预条件效果远优于 Jacobi，构建代价 O(nnz)。
 *
 * 存储：
 *   L 的严格下三角 + U 的上三角（含对角线）共用 A 的 CSR 结构。
 *   实际只存一个值数组 lu_val[nnz]，按位置原地修改。
 *
 * apply(r) → z = (LU)⁻¹ * r：
 *   1. 前代求解 L*y = r   (forward substitution)
 *   2. 回代求解 U*z = y   (back substitution)
 *
 * 使用前必须调用 build()。
 */
class ILU0Preconditioner : public Preconditioner {
  public:
    void build(const SparseMatrix& A) override;
    [[nodiscard]] Vector apply(const Vector& r) const override;
    [[nodiscard]] std::string name() const override { return "ILU0"; }

  private:
    size_t N_{0};

    // 共用 A 的 CSR 结构，值数组原地存 L 和 U
    std::vector<size_t>  row_ptr_;   // 行指针
    std::vector<size_t>  col_idx_;   // 列索引
    std::vector<double>  lu_val_;    // ILU(0) 分解后的值

    // 对角线元素位置缓存（加速前代/回代中查找 L[i][i] 和 U[i][i]）
    std::vector<size_t>  diag_pos_;  // diag_pos_[i] = row 中对角元素的位置
};

}  // namespace math
