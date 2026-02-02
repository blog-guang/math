#include <gtest/gtest.h>
#include "sparse_matrix.h"
#include "vector.h"

using namespace math;

static SparseMatrix make3x3() {
    //  A = [[1, 2, 0],
    //       [3, 4, 5],
    //       [0, 6, 7]]
    return SparseMatrix::fromCOO(3, 3,
        {0,0, 1,1,1, 2,2},
        {0,1, 0,1,2, 1,2},
        {1,2, 3,4,5, 6,7});
}

// ── CSC 元素正确 ────────────────────────────────────────

TEST(CSCTest, Elements) {
    auto A = make3x3();
    A.toCSC();

    EXPECT_EQ(A.format(), StorageFormat::CSC);
    EXPECT_EQ(A.nnz(), 7);

    EXPECT_DOUBLE_EQ(A.get(0,0), 1.0);
    EXPECT_DOUBLE_EQ(A.get(0,1), 2.0);
    EXPECT_DOUBLE_EQ(A.get(0,2), 0.0);
    EXPECT_DOUBLE_EQ(A.get(1,0), 3.0);
    EXPECT_DOUBLE_EQ(A.get(1,1), 4.0);
    EXPECT_DOUBLE_EQ(A.get(1,2), 5.0);
    EXPECT_DOUBLE_EQ(A.get(2,0), 0.0);
    EXPECT_DOUBLE_EQ(A.get(2,1), 6.0);
    EXPECT_DOUBLE_EQ(A.get(2,2), 7.0);

    const auto& cp = A.csc_col_ptr();
    EXPECT_EQ(cp.size(), 4);
    EXPECT_EQ(cp[1] - cp[0], 2);  // col 0: 2 nnz
    EXPECT_EQ(cp[2] - cp[1], 3);  // col 1: 3 nnz
    EXPECT_EQ(cp[3] - cp[2], 2);  // col 2: 2 nnz
}

// ── multiplyTranspose y = A^T * x ──────────────────────

TEST(CSCTest, MultiplyTranspose) {
    auto A = make3x3();
    // A^T = [[1,3,0],[2,4,6],[0,5,7]]
    Vector x = {1.0, 2.0, 3.0};
    Vector y = A.multiplyTranspose(x);

    //   y[0] = 1*1 + 3*2 + 0*3 = 7
    //   y[1] = 2*1 + 4*2 + 6*3 = 28
    //   y[2] = 0*1 + 5*2 + 7*3 = 31
    EXPECT_TRUE(y.approx_equal(Vector{7.0, 28.0, 31.0}));
}

// ── 对称矩阵 A^T*x == A*x ─────────────────────────────

TEST(CSCTest, TransposeSymmetric) {
    std::vector<size_t> rows = {0,0, 1,1,1, 2,2,2, 3,3};
    std::vector<size_t> cols = {0,1, 0,1,2, 1,2,3, 2,3};
    std::vector<double> vals = {4,-1,-1,4,-1,-1,4,-1,-1,4};

    auto A1 = SparseMatrix::fromCOO(4, 4, rows, cols, vals);
    auto A2 = SparseMatrix::fromCOO(4, 4, rows, cols, vals);

    Vector x = {1.0, 2.0, 3.0, 4.0};
    Vector y_mul  = A1.multiply(x);
    Vector y_tran = A2.multiplyTranspose(x);

    EXPECT_TRUE(y_mul.approx_equal(y_tran, 1e-15));
}

// ── 单位矩阵 A^T = I ───────────────────────────────────

TEST(CSCTest, TransposeIdentity) {
    auto I = SparseMatrix::identity(5);
    Vector x = {1.0, -2.0, 3.0, -4.0, 5.0};
    Vector y = I.multiplyTranspose(x);
    EXPECT_TRUE(y.approx_equal(x));
}

// ── 大矩阵非对称转置正确性 ─────────────────────────────

TEST(CSCTest, TransposeLargeNonSymmetric) {
    size_t N = 1000;
    std::vector<size_t> rows, cols;
    std::vector<double> vals;
    for (size_t i = 0; i < N; ++i) {
        rows.push_back(i); cols.push_back(i); vals.push_back(4.0);
        if (i > 0)   { rows.push_back(i); cols.push_back(i-1); vals.push_back(-1.0); }
        if (i+1 < N) { rows.push_back(i); cols.push_back(i+1); vals.push_back(-2.0); }
    }
    auto A = SparseMatrix::fromCOO(N, N, rows, cols, vals);
    Vector x = Vector::random(N, 42);
    Vector y = A.multiplyTranspose(x);

    // A[i][i]=4, A[i][i-1]=-1(下), A[i][i+1]=-2(上)
    // A^T[i][j] = A[j][i]:
    //   A^T[i][i-1] = A[i-1][i] = -2 (A 的上对角)
    //   A^T[i][i+1] = A[i+1][i] = -1 (A 的下对角)
    // → y[0] = 4*x[0] - x[1]
    EXPECT_NEAR(y[0], 4.0*x[0] - 1.0*x[1], 1e-12);

    size_t mid = N / 2;
    EXPECT_NEAR(y[mid], -2.0*x[mid-1] + 4.0*x[mid] - 1.0*x[mid+1], 1e-12);

    EXPECT_NEAR(y[N-1], -2.0*x[N-2] + 4.0*x[N-1], 1e-12);
}

// ── CSC 转换后 nnz 不变 ─────────────────────────────────

TEST(CSCTest, NnzPreserved) {
    auto A = make3x3();
    size_t nnz_before = A.nnz();
    A.toCSC();
    EXPECT_EQ(A.nnz(), nnz_before);
}

// ── CSC 幂等性 ──────────────────────────────────────────

TEST(CSCTest, CSCIdempotent) {
    auto A = make3x3();
    A.toCSC();
    EXPECT_EQ(A.format(), StorageFormat::CSC);

    // 再调一次，状态不变
    A.toCSC();
    EXPECT_EQ(A.format(), StorageFormat::CSC);
    EXPECT_EQ(A.nnz(), 7);
    EXPECT_DOUBLE_EQ(A.get(1, 1), 4.0);
}

// ── 全零矩阵的转置乘法 ────────────────────────────────

TEST(CSCTest, TransposeZeroMatrix) {
    auto Z = SparseMatrix::fromCOO(3, 3, {}, {}, {});
    Vector x = {1.0, 2.0, 3.0};
    Vector y = Z.multiplyTranspose(x);
    EXPECT_TRUE(y.approx_equal(Vector::zeros(3)));
}

// ── 1×1 矩阵转置乘法 ──────────────────────────────────

TEST(CSCTest, Transpose1x1) {
    auto A = SparseMatrix::fromCOO(1, 1, {0}, {0}, {5.0});
    Vector x = {3.0};
    Vector y = A.multiplyTranspose(x);
    EXPECT_NEAR(y[0], 15.0, 1e-12);
}
