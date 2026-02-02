#pragma once
#include "preconditioner.h"
#include "sparse_matrix.h"
#include "vector.h"
#include <memory>

namespace math {

/**
 * AMGCL Wrapper - 使用 AMGCL 库的代数多重网格预条件器。
 * 提供与现有 AMGPreconditioner 相同的接口，但使用 AMGCL 库实现。
 */
class AMGCLPreconditioner : public Preconditioner {
public:
    AMGCLPreconditioner();
    virtual ~AMGCLPreconditioner();

    /**
     * 构建预条件器（setup 阶段）
     * @param A 输入矩阵（必须是 CSR 格式）
     */
    void build(const SparseMatrix& A) override;

    /**
     * 应用预条件器（solve 阶段）：求解 Mz = r
     * @param r 输入向量
     * @return z = M^{-1} * r
     */
    [[nodiscard]] Vector apply(const Vector& r) const override;

    /**
     * 返回预条件器名称
     */
    [[nodiscard]] std::string name() const override { return "AMGCL"; }

private:
    struct Impl;  // PIMPL 模式避免暴露 AMGCL 细节
    std::unique_ptr<Impl> pimpl_;
};

} // namespace math