# 大规模稀疏线性系统求解器 — 开发文档

## 目录
1. [项目概述](#1-项目概述)
2. [数学背景](#2-数学背景)
3. [稀疏矩阵存储格式](#3-稀疏矩阵存储格式)
4. [迭代求解方法](#4-迭代求解方法)
5. [预条件器](#5-预条件器)
6. [项目架构设计](#6-项目架构设计)
7. [收敛判据与参数配置](#7-收敛判据与参数配置)
8. [矩阵文件 I/O](#8-矩阵文件-io)
9. [性能优化策略](#9-性能优化策略)
10. [构建系统 (CMake)](#10-构建系统-cmake)
11. [测试与验证](#11-测试与验证)
12. [开发路线图](#12-开发路线图)

---

## 1. 项目概述

**目标：** 开发一个高性能、模块化的大规模稀疏线性方程组求解器，求解形如 **Ax = b** 的线性系统。

**核心特性：**
- 支持多种稀疏矩阵存储格式（CSR / CSC / COO）
- 实现多种经典迭代求解算法
- 支持预条件加速
- OpenMP 并行化支持
- 标准矩阵文件 I/O（MatrixMarket 格式）
- 可配置的收敛策略和参数

**适用场景：**
- 有限元/有限差分离散后的工程仿真
- 计算流体力学 (CFD)
- 结构力学分析
- 大规模科学计算

---

## 2. 数学背景

### 问题定义
求解 **Ax = b**，其中：
- A：N×N 稀疏矩阵（非零元素远少于 N²）
- x：未知向量（待求）
- b：右端向量（已知）

### 为什么用迭代法？
- 直接法（如 LU 分解）对大规模稀疏矩阵内存和计算开销极大
- 迭代法每轮仅需执行稀疏矩阵-向量乘法（SpMV），复杂度 O(nnz)
- 配合预条件器可大大加快收敛

### 矩阵分类与方法选择
| 矩阵特性 | 推荐方法 |
|----------|----------|
| 对称正定 (SPD) | CG (共轭梯度法) |
| 对称不定 | MINRES |
| 非对称 | GMRES / BiCGSTAB |
| 对角占优 | Jacobi / Gauss-Seidel / SOR |

---

## 3. 稀疏矩阵存储格式

### 3.1 COO (Coordinate)
最简单的格式，适合矩阵构建阶段。

```
row[]: 行索引数组
col[]: 列索引数组
val[]: 对应值数组
nnz:   非零元素个数
```

### 3.2 CSR (Compressed Sparse Row) ⭐ 主要格式
行压缩格式，SpMV 计算最友好，缓存命中率高。

```
row_ptr[N+1]: 第 i 行的非零元素起始位置为 row_ptr[i]
col_idx[nnz]: 列索引
val[nnz]:     非零值
```

**SpMV 核心循环：**
```cpp
for (int i = 0; i < N; i++) {
    double sum = 0.0;
    for (int j = row_ptr[i]; j < row_ptr[i+1]; j++) {
        sum += val[j] * x[col_idx[j]];
    }
    y[i] = sum;
}
```

### 3.3 CSC (Compressed Sparse Column)
与 CSR 对称，适合按列访问（如转置运算）。

### 格式转换路径
```
COO  →  CSR  (构建阶段转换，之后主要用 CSR)
CSR  ↔  CSC  (按需转换)
```

---

## 4. 迭代求解方法

### 4.1 经典迭代法（适用于对角占优矩阵）

#### Jacobi 迭代
```
x_k+1[i] = (b[i] - Σ a[i][j]*x_k[j], j≠i) / a[i][i]
```
- 优点：完全可并行
- 缺点：收敛慢，要求严格对角占优

#### Gauss-Seidel 迭代
```
x_k+1[i] = (b[i] - Σ a[i][j]*x[j]) / a[i][i]
           (已更新的用新值，未更新的用旧值)
```
- 优点：收敛速度是 Jacobi 的 2x
- 缺点：不易并行

#### SOR (Successive Over-Relaxation)
```
x_k+1[i] = (1-ω)*x_k[i] + ω * GS_update[i]
```
- ω ∈ (0, 2)，最优 ω 需根据矩阵谱半径计算
- 适当选取 ω 可显著加速收敛

### 4.2 Krylov 子空间方法（大规模系统首选）

#### CG (Conjugate Gradient) — 对称正定矩阵专用
```
算法：
  r0 = b - A*x0
  p0 = r0
  for k = 0, 1, 2, ...
      α_k = (r_k^T r_k) / (p_k^T A p_k)
      x_{k+1} = x_k + α_k * p_k
      r_{k+1} = r_k - α_k * A * p_k
      β_k = (r_{k+1}^T r_{k+1}) / (r_k^T r_k)
      p_{k+1} = r_{k+1} + β_k * p_k
      if ||r_{k+1}|| < tol: break
```
- 理论收敛步数 ≤ N，实际远少于此
- 核心运算：SpMV + 向量加减 + 内积

#### GMRES (Generalized Minimal Residual) — 非对称矩阵
- 在 Krylov 子空间中找使残差最小的解
- 采用重启策略 GMRES(m) 控制内存
- 适用于一般非对称矩阵

#### BiCGSTAB (Bi-Conjugate Gradient Stabilized)
- 适用于非对称矩阵
- 比 GMRES 内存占用低
- 收敛行为稳定

---

## 5. 预条件器

预条件器将 Ax=b 转化为 M⁻¹Ax = M⁻¹b，使条件数降低、收敛加速。

### 5.1 Jacobi 预条件器（对角预条件）
```
M = diag(A)
M⁻¹ * v：逐元素除以对角线值
```
- 极轻量，几乎无额外开销
- 适作为基础预条件

### 5.2 ILU(0) — 不完全 LU 分解
- 对 A 做 LU 分解，但只保留与 A 相同非零结构的元素
- 预条件效果远优于 Jacobi
- 构建开销 O(nnz)，存储 2×nnz

### 5.3 Block Jacobi
- 将矩阵分块，每块独立做 LU 分解
- 适合并行化

---

## 6. 项目架构设计

```
math/
├── CMakeLists.txt
├── DEV.md                  ← 本文档
├── include/
│   ├── sparse_matrix.h     # 稀疏矩阵抽象（COO/CSR/CSC）
│   ├── vector.h            # 向量运算封装
│   ├── solver.h            # 求解器基类接口
│   ├── solvers/
│   │   ├── jacobi.h
│   │   ├── gauss_seidel.h
│   │   ├── sor.h
│   │   ├── cg.h
│   │   ├── gmres.h
│   │   └── bicgstab.h
│   ├── preconditioner.h    # 预条件器接口
│   ├── preconditioners/
│   │   ├── jacobi_precond.h
│   │   └── ilu0.h
│   └── io/
│       └── matrix_market.h # MatrixMarket 文件读写
├── src/
│   ├── sparse_matrix.cpp
│   ├── vector.cpp
│   ├── solvers/
│   │   ├── jacobi.cpp
│   │   ├── gauss_seidel.cpp
│   │   ├── sor.cpp
│   │   ├── cg.cpp
│   │   ├── gmres.cpp
│   │   └── bicgstab.cpp
│   ├── preconditioners/
│   │   ├── jacobi_precond.cpp
│   │   └── ilu0.cpp
│   └── io/
│       └── matrix_market.cpp
├── examples/
│   └── solve_example.cpp   # 使用示例
└── tests/
    ├── test_matrix.cpp
    ├── test_cg.cpp
    ├── test_gmres.cpp
    └── test_precond.cpp
```

### 核心接口设计

```cpp
// 求解器统一接口
class Solver {
public:
    virtual SolveResult solve(const SparseMatrix& A,
                              const Vector& b,
                              Vector& x,
                              const SolverConfig& config) = 0;
};

// 返回结果
struct SolveResult {
    bool converged;
    int iterations;
    double residual;      // 最终残差 ||b - Ax||
    double elapsed_ms;
};

// 配置参数
struct SolverConfig {
    double tol = 1e-10;       // 收敛容差
    int max_iter = 10000;     // 最大迭代次数
    Preconditioner* precond = nullptr;
    int gmres_restart = 50;   // GMRES 重启参数
    double sor_omega = 1.5;   // SOR 松弛因子
};
```

---

## 7. 收敛判据与参数配置

### 收敛条件（任选）
| 类型 | 公式 | 说明 |
|------|------|------|
| 绝对残差 | \|\|r_k\|\| < tol | 残差绝对值 |
| 相对残差 | \|\|r_k\|\| / \|\|b\|\| < tol | **最常用** |
| 相对变化 | \|\|x_k - x_{k-1}\|\| / \|\|x_k\|\| < tol | 解的变化率 |

### 关键参数
| 参数 | 典型值 | 说明 |
|------|--------|------|
| tol | 1e-6 ~ 1e-10 | 收敛容差 |
| max_iter | 1000 ~ 10000 | 防止不收敛死循环 |
| sor_omega | 1.0 ~ 1.9 | SOR 松弛因子 |
| gmres_m | 20 ~ 100 | GMRES 重启长度 |

---

## 8. 矩阵文件 I/O

### MatrixMarket 格式 (.mtx)
业内标准稀疏矩阵交换格式。

```
%%MatrixMarket matrix coordinate real general
% 注释行
6 6 12          ← 行数 列数 非零元素数
1 1 2.0         ← 行 列 值 (1-indexed)
1 2 -1.0
2 1 -1.0
...
```

**读写接口：**
```cpp
SparseMatrix MatrixMarket::read(const std::string& filename);
void MatrixMarket::write(const std::string& filename, const SparseMatrix& mat);
```

---

## 9. 性能优化策略

### 9.1 内存访问优化
- CSR 格式保证行访问连续，SpMV 缓存友好
- 矩阵行重序（如 RCM 算法）降低带宽

### 9.2 OpenMP 并行化
- **SpMV 并行：** 每个线程处理若干行（天然无数据依赖）
- **向量运算并行：** 加法、内积、缩放均可并行
- **注意：** Gauss-Seidel 本质有序依赖，需用红黑染色并行

```cpp
#pragma omp parallel for schedule(static)
for (int i = 0; i < N; i++) {
    double sum = 0.0;
    for (int j = row_ptr[i]; j < row_ptr[i+1]; j++)
        sum += val[j] * x[col_idx[j]];
    y[i] = sum;
}
```

### 9.3 SIMD 向量化
- 编译器自动向量化（-O3 -march=native）
- 关键循环结构需足够简单以配合自动向量化

### 9.4 编译优化标志
```
-O3 -march=native -fopenmp -funroll-loops
```

---

## 10. 构建系统 (CMake)

```cmake
cmake_minimum_required(VERSION 3.20)
project(SparseSolver LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenMP)

add_library(math STATIC
    src/sparse_matrix.cpp
    src/vector.cpp
    src/solvers/jacobi.cpp
    src/solvers/gauss_seidel.cpp
    src/solvers/sor.cpp
    src/solvers/cg.cpp
    src/solvers/gmres.cpp
    src/solvers/bicgstab.cpp
    src/preconditioners/jacobi_precond.cpp
    src/preconditioners/ilu0.cpp
    src/io/matrix_market.cpp
)

target_include_directories(math PUBLIC include/)
if(OpenMP_CXX_FOUND)
    target_link_libraries(math PUBLIC OpenMP::OpenMP_CXX)
endif()

# 示例
add_executable(solve_example examples/solve_example.cpp)
target_link_libraries(solve_example math)

# 测试
enable_testing()
add_executable(tests tests/test_matrix.cpp tests/test_cg.cpp tests/test_gmres.cpp)
target_link_libraries(tests math)
add_test(NAME solver_tests COMMAND tests)
```

---

## 11. 测试与验证

### 单位测试
| 测试项 | 方法 |
|--------|------|
| 矩阵构建与转换 | COO→CSR→CSC 往返一致性 |
| SpMV 正确性 | 与稠密矩阵乘法对比 |
| 求解器收敛 | 已知解的小规模系统验证 |
| 预条件器 | 验证 M⁻¹A 条件数降低 |

### 性能基准测试
- 生成不同规模（1K / 10K / 100K / 1M）的标准测试矩阵
- 记录迭代次数、耗时、内存占用
- 对比各求解器在同一矩阵上的表现

### 标准测试矩阵
- **Poisson 方程离散矩阵**（对称正定，结构规律）
- **随机对称正定矩阵**
- **非对称不定矩阵**

---

## 12. 开发路线图

| 阶段 | 内容 | 优先级 |
|------|------|--------|
| Phase 1 | 项目骨架 + CMake + 基础向量运算 | 🔴 必须 |
| Phase 2 | 稀疏矩阵：COO / CSR 存储 + SpMV | 🔴 必须 |
| Phase 3 | MatrixMarket I/O | 🔴 必须 |
| Phase 4 | Jacobi / Gauss-Seidel / SOR 求解器 | 🔴 必须 |
| Phase 5 | CG 求解器（对称正定核心） | 🔴 必须 |
| Phase 6 | Jacobi 预条件器 | 🟡 高优先 |
| Phase 7 | GMRES 求解器 | 🟡 高优先 |
| Phase 8 | BiCGSTAB 求解器 | 🟡 高优先 |
| Phase 9 | ILU(0) 预条件器 | 🟢 中优先 |
| Phase 10 | OpenMP 并行化 | 🟢 中优先 |
| Phase 11 | 性能基准测试套件 | 🟢 中优先 |
| Phase 12 | CSC 格式 + 高级优化 | 🔵 后续 |

---

*文档版本：v1.0 — 2026-01-31*
