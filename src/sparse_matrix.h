#pragma once

#include <vector>
#include <iostream>

#include "vector.h"

namespace math {

enum class StorageFormat { COO, CSR, CSC };

/**
 * 稀疏矩阵类，支持 COO 和 CSR 两种存储格式。
 *
 * COO: 三数组 (row, col, val)，适合构建阶段。
 * CSR: 行压缩格式，适合 SpMV 计算，缓存友好。
 *
 * 默认策略：构建时用 COO，计算前自动转换为 CSR。
 */
class SparseMatrix {
  public:
    // ── 构造 ────────────────────────────────────────────
    SparseMatrix() : rows_(0), cols_(0), format_(StorageFormat::COO) {}

    // 禁止隐式拷贝（大矩阵不应意外复制）
    SparseMatrix(const SparseMatrix&) = delete;
    SparseMatrix& operator=(const SparseMatrix&) = delete;

    // 允许移动
    SparseMatrix(SparseMatrix&&) noexcept = default;
    SparseMatrix& operator=(SparseMatrix&&) noexcept = default;

    // ── 静态工厂 ────────────────────────────────────────
    /**
     * 从 COO 数据构建。
     * @param rows  行数
     * @param cols  列数
     * @param row   行索引数组 (0-indexed)
     * @param col   列索引数组 (0-indexed)
     * @param val   对应值数组
     */
    [[nodiscard]] static SparseMatrix fromCOO(
        size_t rows, size_t cols,
        const std::vector<size_t>& row,
        const std::vector<size_t>& col,
        const std::vector<double>& val);

    /**
     * 从 CSR 数据构建。
     * @param rows     行数
     * @param cols     列数
     * @param row_ptr  行指针数组，长度 rows+1
     * @param col_idx  列索引数组，长度 nnz
     * @param val      值数组，长度 nnz
     */
    [[nodiscard]] static SparseMatrix fromCSR(
        size_t rows, size_t cols,
        const std::vector<size_t>& row_ptr,
        const std::vector<size_t>& col_idx,
        const std::vector<double>& val);

    /** N×N 单位矩阵 */
    [[nodiscard]] static SparseMatrix identity(size_t n);

    /** 对角矩阵 */
    [[nodiscard]] static SparseMatrix diagonal(const Vector& diag);

    // ── 查询 ────────────────────────────────────────────
    [[nodiscard]] size_t rows() const noexcept { return rows_; }
    [[nodiscard]] size_t cols() const noexcept { return cols_; }
    [[nodiscard]] size_t nnz() const noexcept;
    [[nodiscard]] StorageFormat format() const noexcept { return format_; }

    /** 获取单个元素（较慢，仅用于调试/验证） */
    [[nodiscard]] double get(size_t i, size_t j) const;

    // ── 格式转换 ────────────────────────────────────────
    /** 转换为 CSR（若已是 CSR 则无操作） */
    void toCSR();
    /** 转换为 COO（若已是 COO 则无操作） */
    void toCOO();
    /** 转换为 CSC（列压缩格式） */
    void toCSC();

    // ── 计算 ────────────────────────────────────────────
    /**
     * 稀疏矩阵-向量乘法：y = A * x
     * 如果当前格式是 COO，会自动先转换为 CSR。
     */
    [[nodiscard]] Vector multiply(const Vector& x);

    /**
     * 转置矩阵-向量乘法：y = A^T * x
     * 利用 CSC 存储高效实现（CSC 按列访问天然对应转置行访问）。
     * 如果当前不是 CSC，会自动先转换。
     */
    [[nodiscard]] Vector multiplyTranspose(const Vector& x);

    // ── 输出 ────────────────────────────────────────────
    friend std::ostream& operator<<(std::ostream& os, const SparseMatrix& m);

    // ── 内部访问（测试用） ──────────────────────────────
    const std::vector<size_t>& coo_row() const { return coo_row_; }
    const std::vector<size_t>& coo_col() const { return coo_col_; }
    const std::vector<double>& coo_val() const { return coo_val_; }
    const std::vector<size_t>& csr_row_ptr() const { return csr_row_ptr_; }
    const std::vector<size_t>& csr_col_idx() const { return csr_col_idx_; }
    const std::vector<double>& csr_val() const { return csr_val_; }
    const std::vector<size_t>& csc_col_ptr() const { return csc_col_ptr_; }
    const std::vector<size_t>& csc_row_idx() const { return csc_row_idx_; }
    const std::vector<double>& csc_val() const { return csc_val_; }

  private:
    size_t rows_;
    size_t cols_;
    StorageFormat format_;

    // COO 数据
    std::vector<size_t> coo_row_;
    std::vector<size_t> coo_col_;
    std::vector<double> coo_val_;

    // CSR 数据
    std::vector<size_t> csr_row_ptr_;
    std::vector<size_t> csr_col_idx_;
    std::vector<double> csr_val_;

    // CSC 数据
    std::vector<size_t> csc_col_ptr_;   // 列指针，长度 cols+1
    std::vector<size_t> csc_row_idx_;   // 行索引，长度 nnz
    std::vector<double> csc_val_;       // 值，长度 nnz

    // 内部转换实现
    void convertCOOtoCSR();
    void convertCSRtoCOO();
    void convertCSRtoCSC();   // CSR → CSC（核心路径）
};

}  // namespace math
