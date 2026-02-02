#pragma once

#include <string>

#include "sparse_matrix.h"
#include "vector.h"

namespace math {
namespace io {

/**
 * MatrixMarket 文件格式读写。
 *
 * 支持格式：
 *   %%MatrixMarket matrix coordinate real general
 *   %%MatrixMarket matrix coordinate real symmetric
 *
 * 索引为 1-based（文件标准），内部转换为 0-based。
 *
 * 向量使用 array 格式：
 *   %%MatrixMarket matrix array real general
 *   N 1
 *   v1
 *   v2
 *   ...
 */

enum class MMSymmetry { General, Symmetric, Skew_Symmetric };

struct MMHeader {
    size_t rows;
    size_t cols;
    size_t nnz;
    MMSymmetry symmetry;
};

class MatrixMarket {
  public:
    /**
     * 从 .mtx 文件读取稀疏矩阵。
     * @param filename 文件路径
     * @return 读取的 SparseMatrix（COO 格式）
     * @throws std::runtime_error 文件打开或解析失败
     */
    [[nodiscard]] static SparseMatrix readMatrix(const std::string& filename);

    /**
     * 将稀疏矩阵写入 .mtx 文件。
     * @param filename 文件路径
     * @param matrix   待写矩阵（会临时转为 COO）
     * @param symmetry 对称性标记，默认 General
     */
    static void writeMatrix(const std::string& filename,
                            SparseMatrix& matrix,
                            MMSymmetry symmetry = MMSymmetry::General);

    /**
     * 从文件读取向量（array 格式）。
     */
    [[nodiscard]] static Vector readVector(const std::string& filename);

    /**
     * 将向量写入文件（array 格式）。
     */
    static void writeVector(const std::string& filename, const Vector& vec);

  private:
    static MMHeader parseHeader(std::istream& is);  // returns io::MMHeader
};

}  // namespace io
}  // namespace math
