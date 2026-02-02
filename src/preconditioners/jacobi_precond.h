#pragma once
#include "preconditioner.h"

namespace math {

/**
 * Jacobi（对角）预条件器。
 *
 * M = diag(A)
 * M⁻¹ * v：逐元素除以对角线值
 *
 * 构建代价：O(nnz)（一次遍历提取对角线）
 * 应用代价：O(N)（逐元素除法）
 * 适合作为轻量基础预条件器。
 */
class JacobiPreconditioner : public Preconditioner {
  public:
    void build(const SparseMatrix& A) override;
    [[nodiscard]] Vector apply(const Vector& r) const override;
    [[nodiscard]] std::string name() const override { return "Jacobi"; }

  private:
    Vector inv_diag_{0};   // 存储 1/A[i][i]，避免迭代时反复做除法
};

}  // namespace math
