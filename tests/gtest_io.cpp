#include <gtest/gtest.h>
#include <filesystem>
#include <fstream>
#include "io/matrix_market.h"
#include "sparse_matrix.h"
#include "vector.h"

using namespace math;
using namespace math::io;
namespace fs = std::filesystem;

static fs::path test_dir() {
    fs::path dir = fs::temp_directory_path() / "math_gtest_io";
    fs::create_directories(dir);
    return dir;
}

// ── General 矩阵往返 ──────────────────────────────────

TEST(IOTest, WriteReadGeneral) {
    auto A = SparseMatrix::fromCOO(
        3, 3,
        {0, 0, 1, 1, 2, 2},
        {0, 1, 1, 2, 0, 2},
        {2, 1, 3, 4, 5, 6});

    fs::path path = test_dir() / "general.mtx";
    MatrixMarket::writeMatrix(path.string(), A, MMSymmetry::General);
    auto B = MatrixMarket::readMatrix(path.string());

    EXPECT_EQ(B.rows(), 3);
    EXPECT_EQ(B.cols(), 3);
    EXPECT_EQ(B.nnz(), 6);

    for (size_t i = 0; i < 3; ++i)
        for (size_t j = 0; j < 3; ++j)
            EXPECT_NEAR(A.get(i, j), B.get(i, j), 1e-12);

    Vector x = {1.0, 2.0, 3.0};
    EXPECT_TRUE(A.multiply(x).approx_equal(B.multiply(x)));
}

// ── Symmetric 矩阵往返 ─────────────────────────────────

TEST(IOTest, WriteReadSymmetric) {
    auto A = SparseMatrix::fromCOO(
        3, 3,
        {0, 0, 0, 1, 1, 1, 2, 2, 2},
        {0, 1, 2, 0, 1, 2, 0, 1, 2},
        {4, 2, 1, 2, 5, 3, 1, 3, 6});

    fs::path path = test_dir() / "symmetric.mtx";
    MatrixMarket::writeMatrix(path.string(), A, MMSymmetry::Symmetric);
    auto B = MatrixMarket::readMatrix(path.string());

    EXPECT_EQ(B.rows(), 3);
    EXPECT_EQ(B.cols(), 3);

    for (size_t i = 0; i < 3; ++i)
        for (size_t j = 0; j < 3; ++j)
            EXPECT_NEAR(A.get(i, j), B.get(i, j), 1e-12);
}

// ── 向量往返 ────────────────────────────────────────────

TEST(IOTest, WriteReadVector) {
    Vector v = {1.5, -2.7, 3.14159, 0.0, 42.0};

    fs::path path = test_dir() / "vector.mtx";
    MatrixMarket::writeVector(path.string(), v);
    Vector w = MatrixMarket::readVector(path.string());

    EXPECT_EQ(w.size(), v.size());
    EXPECT_TRUE(v.approx_equal(w));
}

// ── 手写标准格式读取 ────────────────────────────────────

TEST(IOTest, ReadStandardFormat) {
    fs::path path = test_dir() / "manual.mtx";
    {
        std::ofstream f(path);
        f << "%%MatrixMarket matrix coordinate real general\n";
        f << "% comment\n";
        f << "3 3 4\n";
        f << "1 1 10.0\n";
        f << "1 3 20.0\n";
        f << "2 2 30.0\n";
        f << "3 1 40.0\n";
    }

    auto A = MatrixMarket::readMatrix(path.string());
    EXPECT_EQ(A.rows(), 3);
    EXPECT_EQ(A.cols(), 3);
    EXPECT_EQ(A.nnz(), 4);
    EXPECT_NEAR(A.get(0, 0), 10.0, 1e-12);
    EXPECT_NEAR(A.get(0, 2), 20.0, 1e-12);
    EXPECT_NEAR(A.get(1, 1), 30.0, 1e-12);
    EXPECT_NEAR(A.get(2, 0), 40.0, 1e-12);
    EXPECT_DOUBLE_EQ(A.get(0, 1), 0.0);
}

// ── 非方阵 ──────────────────────────────────────────────

TEST(IOTest, NonSquareMatrix) {
    auto A = SparseMatrix::fromCOO(
        2, 4,
        {0, 0, 1, 1},
        {0, 3, 1, 2},
        {1, 2, 3, 4});

    fs::path path = test_dir() / "nonsquare.mtx";
    MatrixMarket::writeMatrix(path.string(), A, MMSymmetry::General);
    auto B = MatrixMarket::readMatrix(path.string());

    EXPECT_EQ(B.rows(), 2);
    EXPECT_EQ(B.cols(), 4);
    EXPECT_EQ(B.nnz(), 4);

    for (size_t i = 0; i < 2; ++i)
        for (size_t j = 0; j < 4; ++j)
            EXPECT_NEAR(A.get(i, j), B.get(i, j), 1e-12);
}

// ── 单元素矩阵 ──────────────────────────────────────────

TEST(IOTest, SingleElement) {
    auto A = SparseMatrix::fromCOO(1, 1, {0}, {0}, {99.0});

    fs::path path = test_dir() / "single.mtx";
    MatrixMarket::writeMatrix(path.string(), A, MMSymmetry::General);
    auto B = MatrixMarket::readMatrix(path.string());

    EXPECT_EQ(B.rows(), 1);
    EXPECT_EQ(B.cols(), 1);
    EXPECT_EQ(B.nnz(), 1);
    EXPECT_NEAR(B.get(0, 0), 99.0, 1e-12);
}

// ── 精度保持 ────────────────────────────────────────────

TEST(IOTest, Precision) {
    double pi = 3.141592653589793238;
    double e  = 2.718281828459045235;

    auto A = SparseMatrix::fromCOO(2, 2, {0, 1}, {0, 1}, {pi, e});

    fs::path path = test_dir() / "precision.mtx";
    MatrixMarket::writeMatrix(path.string(), A, MMSymmetry::General);
    auto B = MatrixMarket::readMatrix(path.string());

    EXPECT_NEAR(B.get(0, 0), pi, 1e-15);
    EXPECT_NEAR(B.get(1, 1), e,  1e-15);
}

// ── 读取不存在的文件抛出异常 ──────────────────────────

TEST(IOTest, ReadNonExistentFile) {
    EXPECT_THROW(
        { (void)MatrixMarket::readMatrix("/tmp/nonexistent_file_xyz_12345.mtx"); },
        std::runtime_error);
}

// ── 读取格式错误的文件抛出异常 ────────────────────────

TEST(IOTest, ReadMalformedHeader) {
    fs::path path = test_dir() / "malformed.mtx";
    {
        std::ofstream f(path);
        f << "This is not a MatrixMarket file\n";
        f << "garbage data\n";
    }

    EXPECT_THROW(
        { (void)MatrixMarket::readMatrix(path.string()); },
        std::runtime_error);
}

// ── 空矩阵（无非零元素）往返 ───────────────────────────

TEST(IOTest, EmptyMatrix) {
    auto A = SparseMatrix::fromCOO(3, 3, {}, {}, {});
    EXPECT_EQ(A.nnz(), 0);

    fs::path path = test_dir() / "empty.mtx";
    MatrixMarket::writeMatrix(path.string(), A, MMSymmetry::General);
    auto B = MatrixMarket::readMatrix(path.string());

    EXPECT_EQ(B.rows(), 3);
    EXPECT_EQ(B.cols(), 3);
    EXPECT_EQ(B.nnz(), 0);

    Vector x = {1.0, 2.0, 3.0};
    Vector y = B.multiply(x);
    EXPECT_TRUE(y.approx_equal(Vector::zeros(3)));
}

// ── 向量读取不存在文件抛出异常 ────────────────────────

TEST(IOTest, ReadVectorNonExistent) {
    EXPECT_THROW(
        { (void)MatrixMarket::readVector("/tmp/nonexistent_vector_xyz_12345.mtx"); },
        std::runtime_error);
}

// ── 负值和混合符号矩阵往返 ──────────────────────────────

TEST(IOTest, NegativeValues) {
    auto A = SparseMatrix::fromCOO(
        3, 3,
        {0, 0, 1, 1, 2, 2},
        {0, 1, 1, 2, 0, 2},
        {-2.0, 1.5, -3.7, 0.001, 100.0, -0.5});

    fs::path path = test_dir() / "negative.mtx";
    MatrixMarket::writeMatrix(path.string(), A, MMSymmetry::General);
    auto B = MatrixMarket::readMatrix(path.string());

    for (size_t i = 0; i < 3; ++i)
        for (size_t j = 0; j < 3; ++j)
            EXPECT_NEAR(A.get(i, j), B.get(i, j), 1e-12);
}
