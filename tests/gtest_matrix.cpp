#include <gtest/gtest.h>
#include <sstream>
#include <stdexcept>
#include "sparse_matrix.h"
#include "vector.h"

using namespace math;

// ── COO 构建 ────────────────────────────────────────────
TEST(SparseMatrixTest, COOConstruction) {
    //  [2  1  0]
    //  [0  3  4]
    //  [5  0  6]
    auto m = SparseMatrix::fromCOO(
        3, 3,
        {0, 0, 1, 1, 2, 2},   // row
        {0, 1, 1, 2, 0, 2},   // col
        {2, 1, 3, 4, 5, 6});  // val

    EXPECT_EQ(m.rows(), 3);
    EXPECT_EQ(m.cols(), 3);
    EXPECT_EQ(m.nnz(), 6);
    EXPECT_EQ(m.format(), StorageFormat::COO);

    EXPECT_DOUBLE_EQ(m.get(0, 0), 2.0);
    EXPECT_DOUBLE_EQ(m.get(0, 1), 1.0);
    EXPECT_DOUBLE_EQ(m.get(0, 2), 0.0);  // 零元素
    EXPECT_DOUBLE_EQ(m.get(1, 1), 3.0);
    EXPECT_DOUBLE_EQ(m.get(1, 2), 4.0);
    EXPECT_DOUBLE_EQ(m.get(2, 0), 5.0);
    EXPECT_DOUBLE_EQ(m.get(2, 2), 6.0);
}

// ── CSR 构建 ────────────────────────────────────────────
TEST(SparseMatrixTest, CSRConstruction) {
    //  [2  1  0]
    //  [0  3  4]
    //  [5  0  6]
    // CSR:
    //   row_ptr = [0, 2, 4, 6]
    //   col_idx = [0, 1, 1, 2, 0, 2]
    //   val     = [2, 1, 3, 4, 5, 6]
    auto m = SparseMatrix::fromCSR(
        3, 3,
        {0, 2, 4, 6},         // row_ptr
        {0, 1, 1, 2, 0, 2},   // col_idx
        {2, 1, 3, 4, 5, 6});  // val

    EXPECT_EQ(m.rows(), 3);
    EXPECT_EQ(m.cols(), 3);
    EXPECT_EQ(m.nnz(), 6);
    EXPECT_EQ(m.format(), StorageFormat::CSR);

    EXPECT_DOUBLE_EQ(m.get(0, 0), 2.0);
    EXPECT_DOUBLE_EQ(m.get(0, 1), 1.0);
    EXPECT_DOUBLE_EQ(m.get(1, 1), 3.0);
    EXPECT_DOUBLE_EQ(m.get(2, 0), 5.0);
    EXPECT_DOUBLE_EQ(m.get(2, 2), 6.0);
}

// ── COO → CSR 转换 ──────────────────────────────────────
TEST(SparseMatrixTest, COOToCSR) {
    auto m = SparseMatrix::fromCOO(
        3, 3,
        {0, 0, 1, 1, 2, 2},
        {0, 1, 1, 2, 0, 2},
        {2, 1, 3, 4, 5, 6});

    EXPECT_EQ(m.format(), StorageFormat::COO);
    m.toCSR();
    EXPECT_EQ(m.format(), StorageFormat::CSR);

    // 转换后值不变
    EXPECT_DOUBLE_EQ(m.get(0, 0), 2.0);
    EXPECT_DOUBLE_EQ(m.get(0, 1), 1.0);
    EXPECT_DOUBLE_EQ(m.get(0, 2), 0.0);
    EXPECT_DOUBLE_EQ(m.get(1, 1), 3.0);
    EXPECT_DOUBLE_EQ(m.get(1, 2), 4.0);
    EXPECT_DOUBLE_EQ(m.get(2, 0), 5.0);
    EXPECT_DOUBLE_EQ(m.get(2, 2), 6.0);

    // 验证 CSR 结构
    std::vector<size_t> expected_row_ptr = {0, 2, 4, 6};
    EXPECT_EQ(m.csr_row_ptr(), expected_row_ptr);

    // 重复调用无操作
    m.toCSR();
    EXPECT_EQ(m.format(), StorageFormat::CSR);
}

// ── CSR → COO 转换 ──────────────────────────────────────
TEST(SparseMatrixTest, CSRToCOO) {
    auto m = SparseMatrix::fromCSR(
        3, 3,
        {0, 2, 4, 6},
        {0, 1, 1, 2, 0, 2},
        {2, 1, 3, 4, 5, 6});

    m.toCOO();
    EXPECT_EQ(m.format(), StorageFormat::COO);
    EXPECT_EQ(m.nnz(), 6);

    // 值不变
    EXPECT_DOUBLE_EQ(m.get(0, 0), 2.0);
    EXPECT_DOUBLE_EQ(m.get(1, 2), 4.0);
    EXPECT_DOUBLE_EQ(m.get(2, 0), 5.0);
}

// ── 往返转换一致性 ──────────────────────────────────────
TEST(SparseMatrixTest, RoundtripConversion) {
    auto m = SparseMatrix::fromCOO(
        4, 4,
        {0, 1, 2, 3, 0, 3},
        {0, 1, 2, 3, 3, 0},
        {1, 2, 3, 4, 5, 6});

    // 记录所有元素
    double original[4][4];
    for (size_t i = 0; i < 4; ++i)
        for (size_t j = 0; j < 4; ++j)
            original[i][j] = m.get(i, j);

    // COO → CSR → COO → CSR
    m.toCSR();
    m.toCOO();
    m.toCSR();

    for (size_t i = 0; i < 4; ++i)
        for (size_t j = 0; j < 4; ++j)
            EXPECT_NEAR(m.get(i, j), original[i][j], 1e-12);
}

// ── SpMV ────────────────────────────────────────────────
TEST(SparseMatrixTest, SpMV) {
    //  A = [2  1  0]     x = [1]     y = A*x = [2*1+1*2+0*3] = [ 4]
    //      [0  3  4]         [2]                [0*1+3*2+4*3]   [18]
    //      [5  0  6]         [3]                [5*1+0*2+6*3]   [23]
    auto A = SparseMatrix::fromCOO(
        3, 3,
        {0, 0, 1, 1, 2, 2},
        {0, 1, 1, 2, 0, 2},
        {2, 1, 3, 4, 5, 6});

    Vector x = {1.0, 2.0, 3.0};
    Vector y = A.multiply(x);  // 自动转为 CSR 再计算

    EXPECT_EQ(A.format(), StorageFormat::CSR);  // 已自动转换
    EXPECT_NEAR(y[0], 4.0, 1e-12);
    EXPECT_NEAR(y[1], 18.0, 1e-12);
    EXPECT_NEAR(y[2], 23.0, 1e-12);

    // 已是 CSR，再乘一次
    Vector y2 = A.multiply(x);
    EXPECT_TRUE(y.approx_equal(y2));
}

// ── 单位矩阵 ────────────────────────────────────────────
TEST(SparseMatrixTest, Identity) {
    auto I = SparseMatrix::identity(4);
    EXPECT_EQ(I.rows(), 4);
    EXPECT_EQ(I.cols(), 4);
    EXPECT_EQ(I.nnz(), 4);

    // I * x = x
    Vector x = {1.0, 2.0, 3.0, 4.0};
    Vector y = I.multiply(x);
    EXPECT_TRUE(y.approx_equal(x));
}

// ── 对角矩阵 ────────────────────────────────────────────
TEST(SparseMatrixTest, Diagonal) {
    Vector d = {2.0, 3.0, 5.0};
    auto D = SparseMatrix::diagonal(d);
    EXPECT_EQ(D.rows(), 3);
    EXPECT_EQ(D.cols(), 3);
    EXPECT_EQ(D.nnz(), 3);

    // D * x = element-wise d*x
    Vector x = {1.0, 2.0, 3.0};
    Vector y = D.multiply(x);
    Vector expected = {2.0, 6.0, 15.0};
    EXPECT_TRUE(y.approx_equal(expected));
}

// ── 边界情况 ────────────────────────────────────────────
TEST(SparseMatrixTest, EdgeCases) {
    // 全零矩阵（无非零元素）
    auto Z = SparseMatrix::fromCOO(3, 3, {}, {}, {});
    EXPECT_EQ(Z.nnz(), 0);
    Vector x = {1.0, 2.0, 3.0};
    Vector y = Z.multiply(x);
    EXPECT_TRUE(y.approx_equal(Vector::zeros(3)));

    // 1×1 矩阵
    auto S = SparseMatrix::fromCOO(1, 1, {0}, {0}, {7.0});
    Vector x1 = {3.0};
    Vector y1 = S.multiply(x1);
    EXPECT_NEAR(y1[0], 21.0, 1e-12);

    // 非方阵 2×3
    //  [1 0 2]
    //  [0 3 0]
    auto R = SparseMatrix::fromCOO(
        2, 3,
        {0, 0, 1},
        {0, 2, 1},
        {1, 2, 3});
    Vector xr = {1.0, 2.0, 3.0};
    Vector yr = R.multiply(xr);
    // [1*1+0*2+2*3, 0*1+3*2+0*3] = [7, 6]
    EXPECT_NEAR(yr[0], 7.0, 1e-12);
    EXPECT_NEAR(yr[1], 6.0, 1e-12);
}

// ── 移动语义 ────────────────────────────────────────────
TEST(SparseMatrixTest, MoveConstruction) {
    auto A = SparseMatrix::fromCOO(
        2, 2,
        {0, 1},
        {0, 1},
        {3.0, 7.0});

    SparseMatrix B = std::move(A);
    EXPECT_EQ(B.rows(), 2);
    EXPECT_EQ(B.cols(), 2);
    EXPECT_EQ(B.nnz(), 2);
    EXPECT_DOUBLE_EQ(B.get(0, 0), 3.0);
    EXPECT_DOUBLE_EQ(B.get(1, 1), 7.0);
}

TEST(SparseMatrixTest, MoveAssignment) {
    auto A = SparseMatrix::fromCOO(
        3, 3,
        {0, 1, 2},
        {0, 1, 2},
        {1.0, 2.0, 3.0});

    SparseMatrix B;
    B = std::move(A);
    EXPECT_EQ(B.rows(), 3);
    EXPECT_EQ(B.nnz(), 3);
    EXPECT_DOUBLE_EQ(B.get(2, 2), 3.0);
}

// ── get() 越界抛出异常 ──────────────────────────────────
TEST(SparseMatrixTest, GetOutOfRange) {
    auto A = SparseMatrix::fromCOO(
        3, 3,
        {0, 1, 2},
        {0, 1, 2},
        {1.0, 2.0, 3.0});

    EXPECT_THROW({ (void)A.get(3, 0); }, std::out_of_range);
    EXPECT_THROW({ (void)A.get(0, 3); }, std::out_of_range);
    EXPECT_THROW({ (void)A.get(5, 5); }, std::out_of_range);
}

// ── fromCOO 索引越界抛出 ──────────────────────────────
TEST(SparseMatrixTest, FromCOOInvalidIndex) {
    EXPECT_THROW(
        { (void)SparseMatrix::fromCOO(2, 2, {0, 2}, {0, 0}, {1.0, 2.0}); },
        std::out_of_range);
    EXPECT_THROW(
        { (void)SparseMatrix::fromCOO(2, 2, {0, 0}, {0, 2}, {1.0, 2.0}); },
        std::out_of_range);
}

// ── fromCOO/CSR 长度不匹配抛出 ──────────────────────────
TEST(SparseMatrixTest, FromCOOSizeMismatch) {
    EXPECT_THROW(
        { (void)SparseMatrix::fromCOO(2, 2, {0}, {0, 1}, {1.0, 2.0}); },
        std::invalid_argument);
}

TEST(SparseMatrixTest, FromCSRSizeMismatch) {
    // row_ptr 长度应为 rows+1=3，此处给4个
    EXPECT_THROW(
        { (void)SparseMatrix::fromCSR(2, 2, {0, 1, 2, 3}, {0, 1}, {1.0, 2.0}); },
        std::invalid_argument);
}

// ── CSC 幂等性 ──────────────────────────────────────────
TEST(SparseMatrixTest, CSCIdempotent) {
    auto A = SparseMatrix::fromCOO(
        3, 3,
        {0, 0, 1, 1, 2, 2},
        {0, 1, 1, 2, 0, 2},
        {2, 1, 3, 4, 5, 6});

    A.toCSC();
    EXPECT_EQ(A.format(), StorageFormat::CSC);
    size_t nnz1 = A.nnz();

    // 再次调用不改变状态
    A.toCSC();
    EXPECT_EQ(A.format(), StorageFormat::CSC);
    EXPECT_EQ(A.nnz(), nnz1);
    EXPECT_DOUBLE_EQ(A.get(0, 0), 2.0);
    EXPECT_DOUBLE_EQ(A.get(2, 2), 6.0);
}

// ── 非方阵的转置乘法 ────────────────────────────────────
TEST(SparseMatrixTest, NonSquareMultiplyTranspose) {
    // A 为 2×3:
    //  [1 0 2]
    //  [0 3 0]
    // A^T 为 3×2:
    //  [1 0]
    //  [0 3]
    //  [2 0]
    auto A = SparseMatrix::fromCOO(
        2, 3,
        {0, 0, 1},
        {0, 2, 1},
        {1.0, 2.0, 3.0});

    Vector x = {1.0, 2.0};  // 2D vector
    Vector y = A.multiplyTranspose(x);  // y = A^T * x, 结果 3D

    // y[0] = 1*1 + 0*2 = 1
    // y[1] = 0*1 + 3*2 = 6
    // y[2] = 2*1 + 0*2 = 2
    EXPECT_EQ(y.size(), 3);
    EXPECT_NEAR(y[0], 1.0, 1e-12);
    EXPECT_NEAR(y[1], 6.0, 1e-12);
    EXPECT_NEAR(y[2], 2.0, 1e-12);
}

// ── 流输出不崩溃 ────────────────────────────────────────
TEST(SparseMatrixTest, StreamOutput) {
    auto A = SparseMatrix::fromCOO(
        2, 2,
        {0, 1},
        {0, 1},
        {5.0, 9.0});

    std::ostringstream oss;
    oss << A;
    EXPECT_TRUE(oss.good());
    std::string s = oss.str();
    EXPECT_FALSE(s.empty());
}

// ── CSR 幂等性 ──────────────────────────────────────────
TEST(SparseMatrixTest, CSRIdempotent) {
    auto A = SparseMatrix::fromCOO(
        3, 3,
        {0, 1, 2},
        {0, 1, 2},
        {1.0, 2.0, 3.0});

    A.toCSR();
    EXPECT_EQ(A.format(), StorageFormat::CSR);
    A.toCSR();  // 再调一次
    EXPECT_EQ(A.format(), StorageFormat::CSR);
    EXPECT_EQ(A.nnz(), 3);
}