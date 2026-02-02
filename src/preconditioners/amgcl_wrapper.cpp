#include "amgcl_wrapper.h"
#include <amgcl/amg.hpp>
#include <amgcl/backend/builtin.hpp>
#include <amgcl/coarsening/smoothed_aggregation.hpp>
#include <amgcl/relaxation/spai0.hpp>
#include <amgcl/adapter/crs_tuple.hpp>
#include <stdexcept>

namespace math {

struct AMGCLPreconditioner::Impl {
    // Define the AMG type as a preconditioner only (not a full solver)
    using Backend = amgcl::backend::builtin<double>;
    using Precond_Type = amgcl::amg<
        Backend,
        amgcl::coarsening::smoothed_aggregation,
        amgcl::relaxation::spai0
    >;

    std::unique_ptr<Precond_Type> amg;

    // Store as vectors (needed for AMGCL interface)
    std::vector<ptrdiff_t> row_ptr;
    std::vector<ptrdiff_t> col_idx;
    std::vector<double> values;
    ptrdiff_t n_rows;
};

AMGCLPreconditioner::AMGCLPreconditioner() : pimpl_(std::make_unique<Impl>()) {}

AMGCLPreconditioner::~AMGCLPreconditioner() = default;

void AMGCLPreconditioner::build(const SparseMatrix& A) {
    if (A.format() != StorageFormat::CSR) {
        throw std::invalid_argument("AMGCLPreconditioner: Matrix must be in CSR format");
    }

    // 获取 CSR 数据
    const auto& rp = A.csr_row_ptr();
    const auto& ci = A.csr_col_idx();
    const auto& a = A.csr_val();

    if (rp.empty() || ci.empty() || a.empty() || rp.size() != A.rows() + 1) {
        throw std::invalid_argument("AMGCLPreconditioner: Invalid CSR format");
    }

    // Copy to vectors for AMGCL (which expects containers with iterators)
    pimpl_->row_ptr.assign(rp.begin(), rp.end());
    pimpl_->col_idx.assign(ci.begin(), ci.end());
    pimpl_->values.assign(a.begin(), a.end());
    pimpl_->n_rows = static_cast<ptrdiff_t>(A.rows());

    try {
        // 创建 AMGCL 预条件器 - use make_tuple with references
        auto matrix_tuple = std::make_tuple(
            pimpl_->n_rows,
            std::cref(pimpl_->row_ptr),
            std::cref(pimpl_->col_idx),
            std::cref(pimpl_->values)
        );
        pimpl_->amg = std::make_unique<Impl::Precond_Type>(matrix_tuple);
    } catch (const std::exception& e) {
        pimpl_->amg.reset();
        throw std::runtime_error(std::string("AMGCLPreconditioner build failed: ") + e.what());
    }
}

Vector AMGCLPreconditioner::apply(const Vector& r) const {
    if (!pimpl_->amg) {
        throw std::runtime_error("AMGCLPreconditioner: Not built yet");
    }

    const size_t n = r.size();

    // 从 Vector 的原始数据直接构建 backend 输入（避免逐元素拷贝）
    std::vector<double> rhs(r.data(), r.data() + n);
    std::vector<double> x(n, 0.0);

    try {
        using Backend = amgcl::backend::builtin<double>;
        auto rhs_backend = Backend::copy_vector(rhs, typename Backend::params());
        auto x_backend   = Backend::copy_vector(x,   typename Backend::params());

        // 应用预条件器：求解 M*z = r，得 z = M⁻¹*r
        pimpl_->amg->apply(*rhs_backend, *x_backend);

        // 直接从 backend 写入结果 Vector（跳过中间 std::vector）
        Vector result(n);
        const auto& x_ref = *x_backend;
        for (size_t i = 0; i < n; ++i) {
            result[i] = x_ref[i];
        }
        return result;
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("AMGCLPreconditioner apply failed: ") + e.what());
    }
}

} // namespace math