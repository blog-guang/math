#pragma once

#include <string>

#include "sparse_matrix.h"
#include "vector.h"

namespace math {

/**
 * 预条件器基类。
 *
 * 预条件器将 Ax=b 转化为 M⁻¹Ax = M⁻¹b，降低条件数加速收敛。
 * 核心接口：apply(r) → z = M⁻¹ * r
 *
 * 使用流程：
 *   1. build(A)   — 从矩阵构建预条件器（一次性）
 *   2. apply(r)   — 迭代过程中反复调用
 */
class Preconditioner {
  public:
    virtual ~Preconditioner() = default;

    /**
     * 构建预条件器。
     * @param A 系数矩阵（需已转为 CSR）
     */
    virtual void build(const SparseMatrix& A) = 0;

    /**
     * 应用预条件：z = M⁻¹ * r
     * @param r 输入向量
     * @return  预条件后的向量 z
     */
    [[nodiscard]] virtual Vector apply(const Vector& r) const = 0;

    [[nodiscard]] virtual std::string name() const = 0;
};

}  // namespace math
