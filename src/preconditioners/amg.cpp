#include "amg.h"
#include "solver.h"   // detail::SINGULAR_TOL

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace math {

// ════════════════════════════════════════════════════════
// Constructor
// ════════════════════════════════════════════════════════
AMGPreconditioner::AMGPreconditioner(const AMGConfig& cfg)
    : config_(cfg) {}

// ════════════════════════════════════════════════════════
// build — 入口
// ════════════════════════════════════════════════════════
void AMGPreconditioner::build(const SparseMatrix& A) {
    if (A.format() != StorageFormat::CSR) {
        throw std::runtime_error("AMG::build: matrix must be in CSR format");
    }
    if (A.rows() != A.cols()) {
        throw std::runtime_error("AMG::build: matrix must be square");
    }

    levels_.clear();
    levels_.reserve(config_.max_levels);  // 避免 push_back 重分配

    buildLevel(A.csr_row_ptr(), A.csr_col_idx(), A.csr_val(), A.rows());
}

// ════════════════════════════════════════════════════════
// buildLevel — 递归构建各层
// ════════════════════════════════════════════════════════
void AMGPreconditioner::buildLevel(const std::vector<size_t>& rp,
                                   const std::vector<size_t>& ci,
                                   const std::vector<double>& val,
                                   size_t N) {
    // 创建当前层级，拷贝矩阵数据
    AMGLevel lev;
    lev.N     = N;
    lev.A_rp  = rp;
    lev.A_ci  = ci;
    lev.A_val = val;

    // 缓存对角线
    lev.diag.resize(N, 0.0);
    for (size_t i = 0; i < N; ++i) {
        for (size_t k = rp[i]; k < rp[i+1]; ++k) {
            if (ci[k] == i) {
                lev.diag[i] = val[k];
                break;
            }
        }
    }

    // 停止条件：规模太小，直接保存为最粗层
    if (N <= config_.coarse_threshold) {
        lev.N_coarse = 0;
        levels_.push_back(std::move(lev));
        return;
    }

    // ── 1. 强连接图 ──
    std::vector<std::vector<size_t>> strong;
    computeStrong(rp, ci, val, N, config_.strength_threshold, strong);

    // ── 2. C/F 分割 ──
    std::vector<bool>   isC;
    std::vector<size_t> coarse_map;
    rugeStubbenCF(strong, N, isC, coarse_map);

    // 计算粗层规模
    size_t N_coarse = 0;
    for (size_t i = 0; i < N; ++i) {
        if (isC[i]) N_coarse++;
    }

    // 停止条件：粗化率太低或无粗点
    if (N_coarse == 0 || N_coarse >= (3 * N) / 4) {
        lev.N_coarse = 0;
        levels_.push_back(std::move(lev));
        return;
    }

    lev.N_coarse = N_coarse;

    // ── 3. 插值算子 P ──
    buildP(rp, ci, val, N, N_coarse, strong, isC, coarse_map,
           lev.P_rp, lev.P_ci, lev.P_val);

    // ── 4. 限制算子 R = P^T ──
    transposeCSR(N, N_coarse,
                 lev.P_rp, lev.P_ci, lev.P_val,
                 lev.R_rp, lev.R_ci, lev.R_val);

    // ── 5. Galerkin 粗网格算子 A_c = R * A_f * P ──
    std::vector<size_t> Ac_rp, Ac_ci;
    std::vector<double> Ac_val;
    galerkinCoarse(N, N_coarse,
                   rp, ci, val,
                   lev.P_rp, lev.P_ci, lev.P_val,
                   lev.R_rp, lev.R_ci, lev.R_val,
                   Ac_rp, Ac_ci, Ac_val);

    // 保存当前层级（push_back 之后 lev 被 move，不能再用）
    levels_.push_back(std::move(lev));

    // ── 6. 递归粗化 ──（层级数检查）
    if ((int)levels_.size() < config_.max_levels) {
        buildLevel(Ac_rp, Ac_ci, Ac_val, N_coarse);
    }
}

// ════════════════════════════════════════════════════════
// computeStrong — 强连接图
// ════════════════════════════════════════════════════════
void AMGPreconditioner::computeStrong(const std::vector<size_t>& rp,
                                      const std::vector<size_t>& ci,
                                      const std::vector<double>& val,
                                      size_t N, double theta,
                                      std::vector<std::vector<size_t>>& strong) {
    strong.resize(N);
    for (size_t i = 0; i < N; ++i) {
        // 找行 i 的最大非对角元绝对值
        double max_off = 0.0;
        for (size_t k = rp[i]; k < rp[i+1]; ++k) {
            if (ci[k] != i) {
                max_off = std::max(max_off, std::abs(val[k]));
            }
        }

        double thresh = theta * max_off;
        strong[i].clear();
        for (size_t k = rp[i]; k < rp[i+1]; ++k) {
            size_t j = ci[k];
            if (j != i && std::abs(val[k]) >= thresh) {
                strong[i].push_back(j);
            }
        }
    }
}

// ════════════════════════════════════════════════════════
// rugeStubbenCF — Ruge-Stüben C/F 分割
// ════════════════════════════════════════════════════════
void AMGPreconditioner::rugeStubbenCF(const std::vector<std::vector<size_t>>& strong,
                                      size_t N,
                                      std::vector<bool>& isC,
                                      std::vector<size_t>& coarse_map) {
    // 按强连接邻居数排序（降序）
    std::vector<size_t> order(N);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
        return strong[a].size() > strong[b].size();
    });

    // 状态：0=未决, 1=C, 2=F
    std::vector<int> state(N, 0);

    // 贪心遍历
    for (size_t idx = 0; idx < N; ++idx) {
        size_t i = order[idx];
        if (state[i] != 0) continue;  // 已决定

        // 标记为 C 点
        state[i] = 1;

        // 其所有未决强连接邻居标记为 F
        for (size_t j : strong[i]) {
            if (state[j] == 0) {
                state[j] = 2;
            }
        }
    }

    // 构建 isC 和 coarse_map
    isC.resize(N);
    coarse_map.resize(N, 0);
    size_t cnt = 0;
    for (size_t i = 0; i < N; ++i) {
        isC[i] = (state[i] == 1);
        if (isC[i]) {
            coarse_map[i] = cnt++;
        }
    }
}

// ════════════════════════════════════════════════════════
// buildP — 插值算子
// ════════════════════════════════════════════════════════
void AMGPreconditioner::buildP(const std::vector<size_t>& A_rp,
                               const std::vector<size_t>& A_ci,
                               const std::vector<double>& A_val,
                               size_t N, size_t N_coarse,
                               const std::vector<std::vector<size_t>>& strong,
                               const std::vector<bool>& isC,
                               const std::vector<size_t>& coarse_map,
                               std::vector<size_t>& P_rp,
                               std::vector<size_t>& P_ci,
                               std::vector<double>& P_val) {
    P_rp.resize(N + 1);
    P_rp[0] = 0;

    // 先计算每行的非零数
    std::vector<std::vector<std::pair<size_t, double>>> rows(N);

    for (size_t i = 0; i < N; ++i) {
        if (isC[i]) {
            // C 点：注入
            rows[i].emplace_back(coarse_map[i], 1.0);
        } else {
            // F 点：用强连接 C 邻居加权插值
            // 找强连接 C 邻居及对应权重
            std::vector<std::pair<size_t, double>> c_strong;  // (coarse_idx, |a_ij|)
            double sum_abs = 0.0;

            for (size_t j : strong[i]) {
                if (isC[j]) {
                    // 从 A 中找 a[i][j]
                    double aij = 0.0;
                    for (size_t k = A_rp[i]; k < A_rp[i+1]; ++k) {
                        if (A_ci[k] == j) {
                            aij = A_val[k];
                            break;
                        }
                    }
                    double w = std::abs(aij);
                    c_strong.emplace_back(coarse_map[j], w);
                    sum_abs += w;
                }
            }

            if (c_strong.empty()) {
                // 回退：扫描整行找所有 C 邻居
                for (size_t k = A_rp[i]; k < A_rp[i+1]; ++k) {
                    size_t j = A_ci[k];
                    if (j != i && isC[j]) {
                        double w = std::abs(A_val[k]);
                        c_strong.emplace_back(coarse_map[j], w);
                        sum_abs += w;
                    }
                }
            }

            if (sum_abs > 0.0) {
                // 按 coarse_map 排序（保证 CSR 列有序）
                std::sort(c_strong.begin(), c_strong.end());
                for (auto& [cidx, w] : c_strong) {
                    rows[i].emplace_back(cidx, w / sum_abs);
                }
            } else {
                // 极端回退：等权分配给所有 C 点（不应发生）
                // 只给自己的粗层对应点注入1（如果有的话）
                // 否则就不插值了
            }
        }

        P_rp[i + 1] = P_rp[i] + rows[i].size();
    }

    // 填充 P_ci, P_val
    size_t nnz = P_rp[N];
    P_ci.resize(nnz);
    P_val.resize(nnz);
    for (size_t i = 0; i < N; ++i) {
        size_t off = P_rp[i];
        for (size_t k = 0; k < rows[i].size(); ++k) {
            P_ci[off + k]  = rows[i][k].first;
            P_val[off + k] = rows[i][k].second;
        }
    }
}

// ════════════════════════════════════════════════════════
// transposeCSR — 转置 CSR 矩阵
// ════════════════════════════════════════════════════════
void AMGPreconditioner::transposeCSR(size_t M, size_t N_cols,
                                     const std::vector<size_t>& P_rp,
                                     const std::vector<size_t>& P_ci,
                                     const std::vector<double>& P_val,
                                     std::vector<size_t>& R_rp,
                                     std::vector<size_t>& R_ci,
                                     std::vector<double>& R_val) {
    // R 是 N_cols × M 的矩阵
    size_t nnz = P_rp[M];
    R_rp.assign(N_cols + 1, 0);
    R_ci.resize(nnz);
    R_val.resize(nnz);

    // 计数每列出现次数（即 R 的每行nnz）
    for (size_t k = 0; k < nnz; ++k) {
        R_rp[P_ci[k] + 1]++;
    }
    // 前缀和
    for (size_t j = 0; j < N_cols; ++j) {
        R_rp[j + 1] += R_rp[j];
    }

    // 用临时 offset 数组填充
    std::vector<size_t> offset(R_rp.begin(), R_rp.begin() + N_cols);
    for (size_t i = 0; i < M; ++i) {
        for (size_t k = P_rp[i]; k < P_rp[i+1]; ++k) {
            size_t j = P_ci[k];
            size_t pos = offset[j]++;
            R_ci[pos]  = i;       // 转置后列索引为原来的行索引
            R_val[pos] = P_val[k];
        }
    }

    // 每行按列索引排序
    for (size_t i = 0; i < N_cols; ++i) {
        size_t start = R_rp[i];
        size_t end   = R_rp[i + 1];
        if (end - start <= 1) continue;

        // 简单插入排序（每行通常很少元素）
        std::vector<std::pair<size_t, double>> tmp;
        tmp.reserve(end - start);
        for (size_t k = start; k < end; ++k) {
            tmp.emplace_back(R_ci[k], R_val[k]);
        }
        std::sort(tmp.begin(), tmp.end());
        for (size_t k = 0; k < tmp.size(); ++k) {
            R_ci[start + k]  = tmp[k].first;
            R_val[start + k] = tmp[k].second;
        }
    }
}

// ════════════════════════════════════════════════════════
// galerkinCoarse — A_c = R * A_f * P
// ════════════════════════════════════════════════════════
void AMGPreconditioner::galerkinCoarse(size_t N_fine, size_t N_coarse,
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
                                       std::vector<double>& Ac_val) {
    // 第一步：T = A_f * P （N_fine × N_coarse）
    // T[i][j] = sum_k A[i][k] * P[k][j]
    // 用 unordered_map 累加
    // T 用每行一个 map
    std::vector<std::unordered_map<size_t, double>> T(N_fine);
    for (size_t i = 0; i < N_fine; ++i) {
        for (size_t ka = A_rp[i]; ka < A_rp[i+1]; ++ka) {
            size_t k   = A_ci[ka];
            double aik = A_val[ka];
            for (size_t kp = P_rp[k]; kp < P_rp[k+1]; ++kp) {
                size_t j   = P_ci[kp];
                double pkj = P_val[kp];
                T[i][j] += aik * pkj;
            }
        }
    }

    // 第二步：A_c = R * T （N_coarse × N_coarse）
    // A_c[i][j] = sum_k R[i][k] * T[k][j]
    std::vector<std::unordered_map<size_t, double>> Ac_map(N_coarse);
    for (size_t i = 0; i < N_coarse; ++i) {
        for (size_t kr = R_rp[i]; kr < R_rp[i+1]; ++kr) {
            size_t k   = R_ci[kr];
            double rik = R_val[kr];
            for (auto& [j, tkj] : T[k]) {
                Ac_map[i][j] += rik * tkj;
            }
        }
    }

    // 转换为 CSR
    Ac_rp.resize(N_coarse + 1);
    Ac_rp[0] = 0;
    size_t total_nnz = 0;
    for (size_t i = 0; i < N_coarse; ++i) {
        total_nnz += Ac_map[i].size();
        Ac_rp[i + 1] = total_nnz;
    }
    Ac_ci.resize(total_nnz);
    Ac_val.resize(total_nnz);

    for (size_t i = 0; i < N_coarse; ++i) {
        // 收集并排序
        std::vector<std::pair<size_t, double>> entries(Ac_map[i].begin(), Ac_map[i].end());
        std::sort(entries.begin(), entries.end());
        size_t off = Ac_rp[i];
        for (size_t k = 0; k < entries.size(); ++k) {
            Ac_ci[off + k]  = entries[k].first;
            Ac_val[off + k] = entries[k].second;
        }
    }
}

// ════════════════════════════════════════════════════════
// apply — 单次 V-cycle
// ════════════════════════════════════════════════════════
Vector AMGPreconditioner::apply(const Vector& r) const {
    if (r.size() != levels_[0].N) {
        throw std::invalid_argument("AMG::apply: size mismatch");
    }
    Vector x(r.size());  // 零初始化
    vcycle(0, r, x);
    return x;
}

// ════════════════════════════════════════════════════════
// vcycle — V-cycle 递归
// ════════════════════════════════════════════════════════
void AMGPreconditioner::vcycle(size_t level, const Vector& rhs, Vector& x) const {
    const AMGLevel& lev = levels_[level];

    // 最粗层：直接求解
    if (level == levels_.size() - 1) {
        x = directSolve(lev.N, lev.A_rp, lev.A_ci, lev.A_val, rhs);
        return;
    }

    // 预光滑
    smooth(lev.A_rp, lev.A_ci, lev.A_val, lev.diag,
           rhs, x, config_.nu_pre, config_.jacobi_omega);

    // 残差 res = rhs - A*x
    Vector Ax = csr_multiply(lev.N, lev.A_rp, lev.A_ci, lev.A_val, x);
    Vector res = rhs - Ax;

    // 限制：coarse_rhs = R * res
    Vector coarse_rhs = csr_multiply(lev.N_coarse,
                                     lev.R_rp, lev.R_ci, lev.R_val, res);

    // 递归粗层修正
    Vector coarse_x(lev.N_coarse);
    vcycle(level + 1, coarse_rhs, coarse_x);

    // 插值修正：x += P * coarse_x
    Vector Px = csr_multiply(lev.N, lev.P_rp, lev.P_ci, lev.P_val, coarse_x);
    x += Px;

    // 后光滑
    smooth(lev.A_rp, lev.A_ci, lev.A_val, lev.diag,
           rhs, x, config_.nu_post, config_.jacobi_omega);
}

// ════════════════════════════════════════════════════════
// smooth — ω-Jacobi 光滑
// ════════════════════════════════════════════════════════
void AMGPreconditioner::smooth(const std::vector<size_t>& rp,
                               const std::vector<size_t>& ci,
                               const std::vector<double>& val,
                               const std::vector<double>& diag,
                               const Vector& rhs, Vector& x,
                               int iters, double omega) {
    size_t N = rhs.size();
    for (int it = 0; it < iters; ++it) {
        Vector Ax = csr_multiply(N, rp, ci, val, x);
        // x[i] += omega * (rhs[i] - Ax[i]) / diag[i]
        for (size_t i = 0; i < N; ++i) {
            if (std::abs(diag[i]) > detail::SINGULAR_TOL) {
                x[i] += omega * (rhs[i] - Ax[i]) / diag[i];
            }
        }
    }
}

// ════════════════════════════════════════════════════════
// csr_multiply — CSR 矩阵-向量乘法
// ════════════════════════════════════════════════════════
Vector AMGPreconditioner::csr_multiply(size_t N,
                                       const std::vector<size_t>& rp,
                                       const std::vector<size_t>& ci,
                                       const std::vector<double>& val,
                                       const Vector& x) {
    Vector y(N);
    for (size_t i = 0; i < N; ++i) {
        double s = 0.0;
        for (size_t k = rp[i]; k < rp[i+1]; ++k) {
            s += val[k] * x[ci[k]];
        }
        y[i] = s;
    }
    return y;
}

// ════════════════════════════════════════════════════════
// directSolve — 最粗层密矩阵 LU 求解（部分主元）
// ════════════════════════════════════════════════════════
Vector AMGPreconditioner::directSolve(size_t N,
                                      const std::vector<size_t>& rp,
                                      const std::vector<size_t>& ci,
                                      const std::vector<double>& val,
                                      const Vector& rhs) {
    // 构建密矩阵
    std::vector<double> A(N * N, 0.0);
    for (size_t i = 0; i < N; ++i) {
        for (size_t k = rp[i]; k < rp[i+1]; ++k) {
            A[i * N + ci[k]] = val[k];
        }
    }

    // 右端向量拷贝
    std::vector<double> b(N);
    for (size_t i = 0; i < N; ++i) b[i] = rhs[i];

    // 部分主元高斯消元
    for (size_t col = 0; col < N; ++col) {
        // 找最大主元行
        size_t max_row = col;
        double max_val = std::abs(A[col * N + col]);
        for (size_t row = col + 1; row < N; ++row) {
            double v = std::abs(A[row * N + col]);
            if (v > max_val) {
                max_val = v;
                max_row = row;
            }
        }

        // 交换行
        if (max_row != col) {
            for (size_t j = 0; j < N; ++j) {
                std::swap(A[col * N + j], A[max_row * N + j]);
            }
            std::swap(b[col], b[max_row]);
        }

        // 对角元小于奇异门槛，跳过消元
        if (std::abs(A[col * N + col]) < detail::SINGULAR_TOL) {
            continue;
        }

        // 消元
        double pivot = A[col * N + col];
        for (size_t row = col + 1; row < N; ++row) {
            double factor = A[row * N + col] / pivot;
            A[row * N + col] = 0.0;
            for (size_t j = col + 1; j < N; ++j) {
                A[row * N + j] -= factor * A[col * N + j];
            }
            b[row] -= factor * b[col];
        }
    }

    // 回代
    std::vector<double> x(N, 0.0);
    for (size_t i = N; i > 0; ) {
        --i;
        double s = b[i];
        for (size_t j = i + 1; j < N; ++j) {
            s -= A[i * N + j] * x[j];
        }
        if (std::abs(A[i * N + i]) > detail::SINGULAR_TOL) {
            x[i] = s / A[i * N + i];
        }
    }

    // 转为 Vector
    Vector result(N);
    for (size_t i = 0; i < N; ++i) result[i] = x[i];
    return result;
}

}  // namespace math
