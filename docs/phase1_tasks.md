# Phase 1 详细任务清单

## S1.1: 实现 QMR 求解器 ⏳

**背景**：Quasi-Minimal Residual (QMR) 是一种适用于非对称矩阵的 Krylov 方法，通过双正交 Lanczos 过程构建。相比 BiCGSTAB，QMR 在某些病态矩阵上更稳定。

**参考文献**：
- Freund & Nachtigal (1991): "QMR: a quasi-minimal residual method for non-Hermitian linear systems"
- Templates for the Solution of Linear Systems, Section 2.3.8

**实现步骤**：

### 1. 算法伪代码

```
QMR(A, b, x0, tol, max_iter, precond=M):
    r0 = b - A*x0
    v_tilde = r0
    y = M^{-1} * v_tilde
    rho = ||y||
    w_tilde = r0
    z = M^{-T} * w_tilde
    xi = ||z||
    gamma = 1, eta = -1
    
    for i = 1, 2, ..., max_iter:
        if rho == 0 or xi == 0:
            breakdown
        
        v = v_tilde / rho
        y = y / rho
        w = w_tilde / xi
        z = z / xi
        
        delta = z^T * y
        if delta == 0:
            breakdown
        
        y_tilde = M^{-1} * A * y
        z_tilde = M^{-T} * A^T * z
        
        if i > 1:
            p = y_tilde - (xi * delta / epsilon) * p
            q = z_tilde - (rho * delta_bar / epsilon) * q
        else:
            p = y_tilde
            q = z_tilde
        
        p_tilde = A * p
        epsilon = q^T * p_tilde
        if epsilon == 0:
            breakdown
        
        beta = epsilon / delta
        v_tilde = p_tilde - beta * v
        y = M^{-1} * v_tilde
        rho_next = ||y||
        
        w_tilde = A^T * q - beta_bar * w
        z = M^{-T} * w_tilde
        xi_next = ||z||
        
        # Update solution
        theta = rho_next / (gamma * |beta|)
        gamma_next = 1 / sqrt(1 + theta^2)
        eta = -eta * rho * gamma_next^2 / (beta * gamma^2)
        
        if i == 1:
            d = eta * p
            s = eta * p_tilde
        else:
            d = eta * p + (theta_old * gamma_next)^2 * d
            s = eta * p_tilde + (theta_old * gamma_next)^2 * s
        
        x = x + d
        r = r - s
        
        # Check convergence
        res = ||r|| / ||b||
        if res < tol:
            return x, CONVERGED
        
        rho = rho_next
        xi = xi_next
        gamma = gamma_next
        theta_old = theta
    
    return x, NOT_CONVERGED
```

### 2. 文件结构

创建文件：
- `src/solvers/qmr.h`
- `src/solvers/qmr.cpp`
- `tests/gtest_qmr.cpp`

### 3. 类接口设计

```cpp
// src/solvers/qmr.h
#pragma once
#include "solver.h"

namespace math {

/**
 * Quasi-Minimal Residual (QMR) solver for non-symmetric systems.
 * 
 * QMR minimizes the residual norm over the Krylov subspace using a
 * bi-orthogonal Lanczos process. More stable than BiCGSTAB for some
 * ill-conditioned problems, at the cost of requiring A^T * v products.
 * 
 * Supports left preconditioning: (M^{-1} A) x = M^{-1} b
 * 
 * Reference: Freund & Nachtigal (1991)
 */
class QMRSolver : public Solver {
public:
    QMRSolver() = default;
    ~QMRSolver() override = default;

    /**
     * Solve Ax = b using QMR.
     * 
     * @param A       Coefficient matrix (must be square)
     * @param b       Right-hand side vector
     * @param x       Solution vector (input: initial guess, output: solution)
     * @param config  Solver configuration (tol, max_iter, preconditioner)
     * @return        Solve result (converged, iterations, residual)
     */
    [[nodiscard]] SolveResult solve(SparseMatrix& A,
                                     const Vector& b,
                                     Vector& x,
                                     SolverConfig& config) override;

    [[nodiscard]] std::string name() const override { return "QMR"; }

private:
    // Helper: Apply preconditioner if available, otherwise return copy
    Vector applyPrecond(SparseMatrix& A, const Vector& r, SolverConfig& config);
};

}  // namespace math
```

### 4. 实现骨架

```cpp
// src/solvers/qmr.cpp
#include "qmr.h"
#include <cmath>
#include <stdexcept>

namespace math {

SolveResult QMRSolver::solve(SparseMatrix& A,
                              const Vector& b,
                              Vector& x,
                              SolverConfig& config) {
    size_t n = A.rows();
    if (A.cols() != n || b.size() != n || x.size() != n) {
        throw std::invalid_argument("QMR: dimension mismatch");
    }

    double b_norm = b.norm();
    if (b_norm < 1e-30) {
        x = Vector::zeros(n);
        return {true, 0, 0.0, 0.0, 0.0};
    }

    // TODO: 实现完整 QMR 算法
    // 1. 初始化向量 v_tilde, w_tilde, y, z, p, q, d, s
    // 2. 主循环：Lanczos 双正交化 + 残差最小化
    // 3. 收敛检查
    // 4. 处理 breakdown 情况

    return {false, config.max_iter, 0.0, 1.0, 0.0};  // placeholder
}

Vector QMRSolver::applyPrecond(SparseMatrix& A,
                                const Vector& r,
                                SolverConfig& config) {
    if (config.precond) {
        return config.precond->apply(r);
    }
    return r;  // No preconditioning
}

}  // namespace math
```

### 5. 单元测试

```cpp
// tests/gtest_qmr.cpp
#include <gtest/gtest.h>
#include "solvers/qmr.h"
#include "sparse_matrix.h"
#include "vector.h"

using namespace math;

// 构造测试矩阵：三对角非对称
static SparseMatrix makeNonSymTridiag(size_t N) {
    std::vector<size_t> rows, cols;
    std::vector<double> vals;
    for (size_t i = 0; i < N; ++i) {
        rows.push_back(i); cols.push_back(i); vals.push_back(4.0);
        if (i > 0)   { rows.push_back(i); cols.push_back(i-1); vals.push_back(-1.5); }
        if (i+1 < N) { rows.push_back(i); cols.push_back(i+1); vals.push_back(-0.5); }
    }
    return SparseMatrix::fromCOO(N, N, rows, cols, vals);
}

TEST(QMRTest, SmallProblem) {
    size_t n = 10;
    auto A = makeNonSymTridiag(n);
    Vector b(n, 1.0);
    Vector x(n, 0.0);

    SolverConfig cfg;
    cfg.tol = 1e-8;
    cfg.max_iter = 1000;

    QMRSolver qmr;
    auto result = qmr.solve(A, b, x, cfg);

    EXPECT_TRUE(result.converged);
    EXPECT_LT(result.relative_residual, 1e-8);

    // Verify residual
    Vector r = b - A.multiply(x);
    EXPECT_LT(r.norm() / b.norm(), 1e-7);
}

TEST(QMRTest, CompareWithGMRES) {
    // TODO: 比较 QMR 和 GMRES 在同一矩阵上的表现
}

TEST(QMRTest, WithPreconditioner) {
    // TODO: 测试预条件 QMR
}
```

### 6. 添加到 CMakeLists.txt

```cmake
# 在 src/CMakeLists.txt 或主 CMakeLists.txt 中添加
add_library(math STATIC
    ...
    src/solvers/qmr.cpp
    ...
)

# 在 tests/CMakeLists.txt 中添加
add_executable(gtest_qmr gtest_qmr.cpp)
target_link_libraries(gtest_qmr math gtest_main)
add_test(NAME qmr_tests COMMAND gtest_qmr)
```

### 7. 验收标准

- [ ] QMR 在 10 个测试矩阵上收敛
- [ ] 与参考实现（SciPy qmr）迭代次数误差 < 10%
- [ ] 单元测试覆盖率 > 90%
- [ ] 代码通过 clang-tidy 检查
- [ ] 文档注释完整（Doxygen 可生成）

---

## S1.2: 实现 TFQMR 求解器

*（类似结构，待填充）*

---

## S1.3: 实现 CGS 求解器

*（类似结构，待填充）*

---

## S1.4: 实现 IDR(s) 求解器

*（类似结构，待填充）*

---

## S1.5: 实现 Chebyshev 迭代

*（类似结构，待填充）*

---

## S1.7: 扩展 benchmark 到 50+ 矩阵

### 矩阵来源

从 SuiteSparse Matrix Collection 下载：

**SPD 矩阵**（20 个）：
- Structural: bcsstk*, NASA*, shuttle*
- FEM: apache*, nd*, thread*

**非对称矩阵**（20 个）：
- CFD: raefsky*, wang*, venkat*
- Circuit: memchip*, rajat*, circuit*

**近奇异/病态矩阵**（10 个）：
- LNS*
- Harwell-Boeing illc*, orsreg*

### 下载脚本

```bash
#!/bin/bash
# scripts/download_suitesparse.sh

BASE_URL="https://suitesparse-collection-website.herokuapp.com"
MATRICES=(
    "HB/bcsstk14"
    "HB/bcsstk15"
    # ... 50 个矩阵列表
)

mkdir -p test_matrices
for mat in "${MATRICES[@]}"; do
    wget "$BASE_URL/MM/$mat.tar.gz"
    tar -xzf "$(basename $mat).tar.gz" -C test_matrices/
done
```

### Benchmark 输出格式

```
Matrix                  Solver      N      nnz     Iter    Time(ms)  Residual    Status
----------------------  ----------  -----  ------  ------  --------  ----------  ------
bcsstk14_spd_1806       CG          1806   63454   142     45.2      3.45e-09    ✓
bcsstk14_spd_1806       QMR         1806   63454   201     78.3      5.12e-09    ✓
sherman1_sym_1000       GMRES       1000   3750    781     322.7     8.76e-09    ✓
sherman1_sym_1000       QMR         1000   3750    645     245.1     7.23e-09    ✓
```

---

## S1.8: 性能回归测试

### CI 集成

```yaml
# .github/workflows/performance.yml
name: Performance Regression

on: [push, pull_request]

jobs:
  benchmark:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Build
        run: |
          cmake -B build -DCMAKE_BUILD_TYPE=Release
          cmake --build build -j$(nproc)
      - name: Run benchmark
        run: ./build/benchmark/benchmark_mm > results.txt
      - name: Compare with baseline
        run: python scripts/compare_performance.py baseline.txt results.txt
      - name: Upload results
        uses: actions/upload-artifact@v3
        with:
          name: benchmark-results
          path: results.txt
```

### 性能回归检测

如果新版本在任何矩阵上性能下降 > 20%，CI 失败并警告。

---

**任务跟踪**：
- S1.1 QMR: ⏳ 进行中
- S1.2 TFQMR: 🔲 待开始
- S1.3 CGS: 🔲 待开始
- S1.4 IDR(s): 🔲 待开始
- S1.5 Chebyshev: 🔲 待开始
- S1.7 Benchmark: 🔲 待开始
- S1.8 CI: 🔲 待开始
