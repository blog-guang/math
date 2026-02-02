#include "jacobi_precond.h"

#include <stdexcept>

namespace math {

void JacobiPreconditioner::build(const SparseMatrix& A) {
    if (A.format() != StorageFormat::CSR) {
        throw std::runtime_error("JacobiPreconditioner::build: matrix must be in CSR format");
    }

    size_t N = A.rows();
    inv_diag_ = Vector(N);

    const auto& row_ptr = A.csr_row_ptr();
    const auto& col_idx = A.csr_col_idx();
    const auto& val     = A.csr_val();

    for (size_t i = 0; i < N; ++i) {
        double diag = 0.0;
        for (size_t k = row_ptr[i]; k < row_ptr[i + 1]; ++k) {
            if (col_idx[k] == i) {
                diag = val[k];
                break;
            }
        }
        if (diag == 0.0) {
            throw std::runtime_error(
                "JacobiPreconditioner: zero diagonal at row " + std::to_string(i));
        }
        inv_diag_[i] = 1.0 / diag;
    }
}

Vector JacobiPreconditioner::apply(const Vector& r) const {
    if (r.size() != inv_diag_.size()) {
        throw std::invalid_argument("JacobiPreconditioner::apply: size mismatch");
    }

    Vector z(r.size());
    for (size_t i = 0; i < r.size(); ++i) {
        z[i] = inv_diag_[i] * r[i];
    }
    return z;
}

}  // namespace math
