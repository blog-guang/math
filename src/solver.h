#pragma once

#include <chrono>
#include <optional>
#include <string>

#include "preconditioner.h"
#include "sparse_matrix.h"
#include "vector.h"

namespace math {

// ── 数值稳定性常量 ──────────────────────────────────────
namespace detail {
    /// 零对角/奇异判定门槛
    constexpr double SINGULAR_TOL  = 1e-30;
    /// 迭代法内积退化（接近下溢范围）
    constexpr double BREAKDOWN_TOL = 1e-300;
    /// GMRES Arnoldi 正交化断裂
    constexpr double ARNOLDI_TOL   = 1e-17;
}

/** 求解结果 */
struct SolveResult {
    bool converged = false;
    int iterations = 0;
    double residual = 0.0;            // ||b - Ax||
    double relative_residual = 0.0;   // ||b - Ax|| / ||b||
    double elapsed_ms = 0.0;
};

/** 求解器配置参数 */
struct SolverConfig {
    double tol = 1e-10;              // 收敛容差（相对残差）
    int max_iter = 10000;            // 最大迭代次数
    int gmres_restart = 50;          // GMRES 重启长度 m
    Preconditioner* precond = nullptr; // 预条件器（nullptr = 无预条件）
    bool verbose = false;            // 是否打印迭代日志
};

/**
 * 求解器基类。
 * 所有迭代求解器继承此类，提供统一接口。
 */
class Solver {
  public:
    virtual ~Solver() = default;

    /**
     * 求解 Ax = b。
     * @param A      系数矩阵（会被转为 CSR）
     * @param b      右端向量
     * @param x      初始猜测 / 输出解向量
     * @param config 求解参数
     * @return       求解结果
     */
    virtual SolveResult solve(SparseMatrix& A,
                              const Vector& b,
                              Vector& x,
                              const SolverConfig& config) = 0;

    /** 求解器名称 */
    [[nodiscard]] virtual std::string name() const = 0;

  protected:
    /** 计算残差向量 r = b - A*x，返回 ||r|| */
    static double computeResidual(SparseMatrix& A, const Vector& b, const Vector& x);

    /** 从 CSR 矩阵提取对角线元素 */
    static Vector extractDiagonal(const SparseMatrix& A);

    // ── 驻点迭代求解器公共辅助 ────────────────────────────

    /** 驻点迭代求解器的前置准备结果 */
    struct StationarySetup {
        Vector diag;      ///< 对角线元素
        double b_norm;    ///< ||b||
    };

    /**
     * 驻点迭代求解器公共前置：验证维度 + CSR 转换 + 对角线提取 + 零对角检查。
     *
     * 若 b = 0，将 x 置零并返回 std::nullopt；调用者应直接返回零解结果。
     *
     * @throws std::invalid_argument  矩阵非方或向量维度不匹配
     * @throws std::runtime_error     存在零对角元素
     */
    static std::optional<StationarySetup>
    prepareStationary(SparseMatrix& A, const Vector& b, Vector& x,
                      const std::string& solver_name);

    /** 封装求解结果并计算耗时。b_norm = 0 时 relative_residual 返回 0。 */
    static SolveResult packResult(bool converged, int iterations,
                                  double residual, double b_norm,
                                  std::chrono::steady_clock::time_point t_start);

    /** Verbose 迭代日志：表头 */
    static void logHeader(const std::string& prefix = "");

    /** Verbose 迭代日志：单行 */
    static void logIter(int iter, double residual, double rel_residual);
};

}  // namespace math
