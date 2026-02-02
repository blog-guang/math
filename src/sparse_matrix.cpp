#include "sparse_matrix.h"

#include <algorithm>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

// 低于此阈值的向量不启动 OpenMP（线程启动开销 > 计算收益）
static constexpr size_t OMP_THRESHOLD = 4096;

namespace math {

// ── nnz ─────────────────────────────────────────────────

size_t SparseMatrix::nnz() const noexcept {
    switch (format_) {
        case StorageFormat::COO: return coo_val_.size();
        case StorageFormat::CSR: return csr_val_.size();
        case StorageFormat::CSC: return csc_val_.size();
    }
    return 0;
}

// ── 静态工厂 ────────────────────────────────────────────

SparseMatrix SparseMatrix::fromCOO(
    size_t rows, size_t cols,
    const std::vector<size_t>& row,
    const std::vector<size_t>& col,
    const std::vector<double>& val)
{
    if (row.size() != col.size() || row.size() != val.size()) {
        throw std::invalid_argument("fromCOO: row/col/val size mismatch");
    }
    for (size_t i = 0; i < row.size(); ++i) {
        if (row[i] >= rows || col[i] >= cols) {
            throw std::out_of_range("fromCOO: index out of range");
        }
    }

    SparseMatrix m;
    m.rows_ = rows;
    m.cols_ = cols;
    m.format_ = StorageFormat::COO;
    m.coo_row_ = row;
    m.coo_col_ = col;
    m.coo_val_ = val;
    return m;
}

SparseMatrix SparseMatrix::fromCSR(
    size_t rows, size_t cols,
    const std::vector<size_t>& row_ptr,
    const std::vector<size_t>& col_idx,
    const std::vector<double>& val)
{
    if (row_ptr.size() != rows + 1) {
        throw std::invalid_argument("fromCSR: row_ptr size must be rows+1");
    }
    if (col_idx.size() != val.size()) {
        throw std::invalid_argument("fromCSR: col_idx/val size mismatch");
    }

    SparseMatrix m;
    m.rows_ = rows;
    m.cols_ = cols;
    m.format_ = StorageFormat::CSR;
    m.csr_row_ptr_ = row_ptr;
    m.csr_col_idx_ = col_idx;
    m.csr_val_ = val;
    return m;
}

SparseMatrix SparseMatrix::identity(size_t n) {
    std::vector<size_t> row(n), col(n);
    std::vector<double> val(n, 1.0);
    for (size_t i = 0; i < n; ++i) {
        row[i] = i;
        col[i] = i;
    }
    return fromCOO(n, n, row, col, val);
}

SparseMatrix SparseMatrix::diagonal(const Vector& diag) {
    size_t n = diag.size();
    std::vector<size_t> row(n), col(n);
    std::vector<double> val(n);
    for (size_t i = 0; i < n; ++i) {
        row[i] = i;
        col[i] = i;
        val[i] = diag[i];
    }
    return fromCOO(n, n, row, col, val);
}

// ── 元素访问 ────────────────────────────────────────────

double SparseMatrix::get(size_t i, size_t j) const {
    if (i >= rows_ || j >= cols_) {
        throw std::out_of_range("get: index out of range");
    }

    switch (format_) {
        case StorageFormat::COO:
            for (size_t k = 0; k < coo_val_.size(); ++k) {
                if (coo_row_[k] == i && coo_col_[k] == j) {
                    return coo_val_[k];
                }
            }
            return 0.0;

        case StorageFormat::CSR:
            for (size_t k = csr_row_ptr_[i]; k < csr_row_ptr_[i + 1]; ++k) {
                if (csr_col_idx_[k] == j) {
                    return csr_val_[k];
                }
            }
            return 0.0;

        case StorageFormat::CSC:
            for (size_t k = csc_col_ptr_[j]; k < csc_col_ptr_[j + 1]; ++k) {
                if (csc_row_idx_[k] == i) {
                    return csc_val_[k];
                }
            }
            return 0.0;
    }
    return 0.0;
}

// ── 格式转换 ────────────────────────────────────────────

void SparseMatrix::toCSR() {
    if (format_ == StorageFormat::CSR) return;
    convertCOOtoCSR();
    format_ = StorageFormat::CSR;
}

void SparseMatrix::toCOO() {
    if (format_ == StorageFormat::COO) return;
    convertCSRtoCOO();
    format_ = StorageFormat::COO;
}

void SparseMatrix::convertCOOtoCSR() {
    size_t nnz = coo_val_.size();

    // 初始化 row_ptr 为零
    csr_row_ptr_.assign(rows_ + 1, 0);

    // 第一趟：统计每行非零元素个数
    for (size_t k = 0; k < nnz; ++k) {
        csr_row_ptr_[coo_row_[k] + 1]++;
    }

    // 前缀求和 → row_ptr[i] = 第 i 行起始位置
    for (size_t i = 1; i <= rows_; ++i) {
        csr_row_ptr_[i] += csr_row_ptr_[i - 1];
    }

    // 第二趟：填充 col_idx 和 val
    csr_col_idx_.resize(nnz);
    csr_val_.resize(nnz);
    std::vector<size_t> cur_pos(csr_row_ptr_.begin(), csr_row_ptr_.end());

    for (size_t k = 0; k < nnz; ++k) {
        size_t r = coo_row_[k];
        size_t pos = cur_pos[r]++;
        csr_col_idx_[pos] = coo_col_[k];
        csr_val_[pos] = coo_val_[k];
    }

    // 每行内按列索引排序（CSR 规范要求）
    for (size_t i = 0; i < rows_; ++i) {
        size_t start = csr_row_ptr_[i];
        size_t end   = csr_row_ptr_[i + 1];
        if (end - start <= 1) continue;

        std::vector<std::pair<size_t, double>> tmp(end - start);
        for (size_t k = start; k < end; ++k) {
            tmp[k - start] = {csr_col_idx_[k], csr_val_[k]};
        }
        std::sort(tmp.begin(), tmp.end());
        for (size_t k = 0; k < tmp.size(); ++k) {
            csr_col_idx_[start + k] = tmp[k].first;
            csr_val_[start + k]     = tmp[k].second;
        }
    }
}

void SparseMatrix::convertCSRtoCOO() {
    size_t nnz = csr_val_.size();
    coo_row_.resize(nnz);
    coo_col_.resize(nnz);
    coo_val_.resize(nnz);

    size_t k = 0;
    for (size_t i = 0; i < rows_; ++i) {
        for (size_t j = csr_row_ptr_[i]; j < csr_row_ptr_[i + 1]; ++j) {
            coo_row_[k] = i;
            coo_col_[k] = csr_col_idx_[j];
            coo_val_[k] = csr_val_[j];
            ++k;
        }
    }
}

// ── SpMV ────────────────────────────────────────────────

Vector SparseMatrix::multiply(const Vector& x) {
    if (x.size() != cols_) {
        throw std::invalid_argument(
            "multiply: vector size " + std::to_string(x.size()) +
            " != matrix cols " + std::to_string(cols_));
    }

    // 确保 CSR 格式
    if (format_ != StorageFormat::CSR) {
        toCSR();
    }

    Vector y(rows_);

    const size_t N      = rows_;
    const double* val   = csr_val_.data();
    const size_t* col   = csr_col_idx_.data();
    const size_t* rptr  = csr_row_ptr_.data();
    const double* xp    = x.data();

#ifdef _OPENMP
    if (N >= OMP_THRESHOLD) {
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < N; ++i) {
            double sum = 0.0;
            for (size_t j = rptr[i]; j < rptr[i + 1]; ++j) {
                sum += val[j] * xp[col[j]];
            }
            y[i] = sum;
        }
    } else
#endif
    {
        for (size_t i = 0; i < N; ++i) {
            double sum = 0.0;
            for (size_t j = rptr[i]; j < rptr[i + 1]; ++j) {
                sum += val[j] * xp[col[j]];
            }
            y[i] = sum;
        }
    }

    return y;
}

// ── CSC 转换 ────────────────────────────────────────────

void SparseMatrix::toCSC() {
    if (format_ == StorageFormat::CSC) return;
    // 确保先有 CSR
    if (format_ != StorageFormat::CSR) toCSR();
    convertCSRtoCSC();
    format_ = StorageFormat::CSC;
}

/**
 * CSR → CSC 转换。
 *
 * 本质上是将 CSR 的 (row, col, val) 按列重新组织。
 * 算法与 COO→CSR 对称：
 *   1. 统计每列非零数 → col_ptr（前缀和）
 *   2. 遍历 CSR 各行，将每个元素写入对应列的位置
 *   3. 每列内按行索引排序
 */
void SparseMatrix::convertCSRtoCSC() {
    size_t nnz = csr_val_.size();

    csc_col_ptr_.assign(cols_ + 1, 0);

    // 第一趟：统计每列非零元素个数
    for (size_t k = 0; k < nnz; ++k) {
        csc_col_ptr_[csr_col_idx_[k] + 1]++;
    }

    // 前缀求和
    for (size_t j = 1; j <= cols_; ++j) {
        csc_col_ptr_[j] += csc_col_ptr_[j - 1];
    }

    // 第二趟：填充 row_idx 和 val
    csc_row_idx_.resize(nnz);
    csc_val_.resize(nnz);
    std::vector<size_t> cur_pos(csc_col_ptr_.begin(), csc_col_ptr_.end());

    for (size_t i = 0; i < rows_; ++i) {
        for (size_t k = csr_row_ptr_[i]; k < csr_row_ptr_[i + 1]; ++k) {
            size_t j   = csr_col_idx_[k];
            size_t pos = cur_pos[j]++;
            csc_row_idx_[pos] = i;
            csc_val_[pos]     = csr_val_[k];
        }
    }

    // 每列内按行索引排序（保证有序）
    for (size_t j = 0; j < cols_; ++j) {
        size_t start = csc_col_ptr_[j];
        size_t end   = csc_col_ptr_[j + 1];
        if (end - start <= 1) continue;

        std::vector<std::pair<size_t, double>> tmp(end - start);
        for (size_t k = start; k < end; ++k) {
            tmp[k - start] = {csc_row_idx_[k], csc_val_[k]};
        }
        std::sort(tmp.begin(), tmp.end());
        for (size_t k = 0; k < tmp.size(); ++k) {
            csc_row_idx_[start + k] = tmp[k].first;
            csc_val_[start + k]     = tmp[k].second;
        }
    }
}

// ── A^T * x ─────────────────────────────────────────────

Vector SparseMatrix::multiplyTranspose(const Vector& x) {
    if (x.size() != rows_) {
        throw std::invalid_argument(
            "multiplyTranspose: vector size " + std::to_string(x.size()) +
            " != matrix rows " + std::to_string(rows_));
    }

    // 确保 CSC 格式
    if (format_ != StorageFormat::CSC) {
        if (format_ != StorageFormat::CSR) toCSR();
        // 保存 CSC 但不改变主格式（保留 CSR 给普通 multiply）
        convertCSRtoCSC();
    }

    // y = A^T * x：对 CSC 而言，第 j 列 = A^T 的第 j 行
    // y[j] = Σ csc_val[k] * x[csc_row_idx[k]]  for k in col j
    size_t N = cols_;   // A^T 的行数 = A 的列数
    Vector y(N);

    const size_t* cptr = csc_col_ptr_.data();
    const size_t* ridx = csc_row_idx_.data();
    const double* val  = csc_val_.data();
    const double* xp   = x.data();

#ifdef _OPENMP
    if (N >= OMP_THRESHOLD) {
        #pragma omp parallel for schedule(static)
        for (size_t j = 0; j < N; ++j) {
            double sum = 0.0;
            for (size_t k = cptr[j]; k < cptr[j + 1]; ++k) {
                sum += val[k] * xp[ridx[k]];
            }
            y[j] = sum;
        }
    } else
#endif
    {
        for (size_t j = 0; j < N; ++j) {
            double sum = 0.0;
            for (size_t k = cptr[j]; k < cptr[j + 1]; ++k) {
                sum += val[k] * xp[ridx[k]];
            }
            y[j] = sum;
        }
    }

    return y;
}

// ── 输出 ────────────────────────────────────────────────

std::ostream& operator<<(std::ostream& os, const SparseMatrix& m) {
    os << "SparseMatrix(" << m.rows_ << "x" << m.cols_
       << ", nnz=" << m.nnz()
       << ", fmt=" << (m.format_ == StorageFormat::COO ? "COO" :
                      (m.format_ == StorageFormat::CSR ? "CSR" : "CSC")) << ")\n";

    // 稠密打印（仅适用于小矩阵）
    if (m.rows_ <= 16 && m.cols_ <= 16) {
        for (size_t i = 0; i < m.rows_; ++i) {
            os << "  [";
            for (size_t j = 0; j < m.cols_; ++j) {
                if (j > 0) os << ", ";
                double v = m.get(i, j);
                if (v == 0.0) os << " 0";
                else os << v;
            }
            os << "]\n";
        }
    }
    return os;
}

}  // namespace math
