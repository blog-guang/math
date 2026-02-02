#pragma once
#include "preconditioner.h"
#include <vector>
#include <string>

namespace math {

/**
 * AMG (Algebraic Multigrid) 预条件器 — 代数多重网格法。
 *
 * 核心思想：通过多层级粗化（coarsening）将问题规模递减，
 * 在最粗层用直接法求解，再通过插值（prolongation）回到细层。
 * 每次 apply() 执行一个 V-cycle。
 *
 * Setup（build）流程：
 *   1. 强连接图 → Ruge-Stüben C/F 分割
 *   2. 构建插值算子 P（N_fine × N_coarse）
 *   3. 限制算子 R = P^T
 *   4. Galerkin 粗网格算子 A_c = R * A_f * P
 *   5. 递归粗化直到满足停止条件
 *
 * Solve（apply = 单次 V-cycle）：
 *   预光滑 → 残差限制 → 递归粗层修正 → 插值修正 → 后光滑
 */

// ── 配置参数 ────────────────────────────────────────────
struct AMGConfig {
    double strength_threshold = 0.25;   // 强连接阈值 θ
    int    max_levels         = 20;     // 最大层级数
    size_t coarse_threshold   = 64;     // 粗化停止：N <= 此值时停止
    int    nu_pre             = 2;      // 预光滑次数
    int    nu_post            = 2;      // 后光滑次数
    double jacobi_omega       = 2.0/3.0;// ω-Jacobi 松弛参数
};

// ── 层级数据 ────────────────────────────────────────────
struct AMGLevel {
    size_t N = 0;

    // A (CSR)
    std::vector<size_t> A_rp, A_ci;
    std::vector<double> A_val;

    // 对角线缓存
    std::vector<double> diag;

    // P: N × N_coarse (CSR) — 仅非最粗层
    std::vector<size_t> P_rp, P_ci;
    std::vector<double> P_val;

    // R = P^T: N_coarse × N (CSR) — 仅非最粗层
    std::vector<size_t> R_rp, R_ci;
    std::vector<double> R_val;

    size_t N_coarse = 0;
};

// ── AMG 预条件器类 ──────────────────────────────────────
class AMGPreconditioner : public Preconditioner {
public:
    explicit AMGPreconditioner(const AMGConfig& cfg = AMGConfig{});

    void build(const SparseMatrix& A) override;
    [[nodiscard]] Vector apply(const Vector& r) const override;
    [[nodiscard]] std::string name() const override { return "AMG"; }

    /** 返回层级数 */
    [[nodiscard]] size_t numLevels() const { return levels_.size(); }

private:
    AMGConfig config_;
    std::vector<AMGLevel> levels_;

    // ── Setup 阶段 ──
    /** 从 CSR 数据构建一层，递归调用直到停止条件 */
    void buildLevel(const std::vector<size_t>& rp,
                    const std::vector<size_t>& ci,
                    const std::vector<double>& val,
                    size_t N);

    /** 强连接图：strong[i] = list of strongly connected neighbors of i */
    static void computeStrong(const std::vector<size_t>& rp,
                              const std::vector<size_t>& ci,
                              const std::vector<double>& val,
                              size_t N, double theta,
                              std::vector<std::vector<size_t>>& strong);

    /** Ruge-Stüben C/F 分割 */
    static void rugeStubbenCF(const std::vector<std::vector<size_t>>& strong,
                              size_t N,
                              std::vector<bool>& isC,
                              std::vector<size_t>& coarse_map);

    /** 构建插值算子 P（CSR） */
    static void buildP(const std::vector<size_t>& A_rp,
                       const std::vector<size_t>& A_ci,
                       const std::vector<double>& A_val,
                       size_t N, size_t N_coarse,
                       const std::vector<std::vector<size_t>>& strong,
                       const std::vector<bool>& isC,
                       const std::vector<size_t>& coarse_map,
                       std::vector<size_t>& P_rp,
                       std::vector<size_t>& P_ci,
                       std::vector<double>& P_val);

    /** 转置 P → R（CSR） */
    static void transposeCSR(size_t M, size_t N_cols,
                             const std::vector<size_t>& P_rp,
                             const std::vector<size_t>& P_ci,
                             const std::vector<double>& P_val,
                             std::vector<size_t>& R_rp,
                             std::vector<size_t>& R_ci,
                             std::vector<double>& R_val);

    /** Galerkin 粗网格算子 A_c = R * A_f * P */
    static void galerkinCoarse(size_t N_fine, size_t N_coarse,
                               const std::vector<size_t>& A_rp,
                               const std::vector<size_t>& A_ci,
                               const std::vector<double>& A_val,
                               const std::vector<size_t>& P_rp,
                               const std::vector<size_t>& P_ci,
                               const std::vector<double>& P_val,
                               const std::vector<size_t>& R_rp,
                               const std::vector<size_t>& R_ci,
                               const std::vector<double>& R_val,
                               std::vector<size_t>& Ac_rp,
                               std::vector<size_t>& Ac_ci,
                               std::vector<double>& Ac_val);

    // ── Solve 阶段 ──
    /** V-cycle 递归求解 */
    void vcycle(size_t level, const Vector& rhs, Vector& x) const;

    /** ω-Jacobi 光滑 */
    static void smooth(const std::vector<size_t>& rp,
                       const std::vector<size_t>& ci,
                       const std::vector<double>& val,
                       const std::vector<double>& diag,
                       const Vector& rhs, Vector& x,
                       int iters, double omega);

    /** CSR矩阵-向量乘法 */
    static Vector csr_multiply(size_t N,
                               const std::vector<size_t>& rp,
                               const std::vector<size_t>& ci,
                               const std::vector<double>& val,
                               const Vector& x);

    /** 最粗层直接求解（密矩阵 LU + 部分主元） */
    static Vector directSolve(size_t N,
                              const std::vector<size_t>& rp,
                              const std::vector<size_t>& ci,
                              const std::vector<double>& val,
                              const Vector& rhs);
};

}  // namespace math
