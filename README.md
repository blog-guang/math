# Math — C++ Sparse Iterative Solver Library

A high-performance C++17 library for solving large sparse linear systems **Ax = b** using iterative methods. Supports symmetric positive definite (SPD) and non-symmetric matrices, with pluggable preconditioners and Matrix Market I/O.

## Features

- **3 iterative solvers**: CG, GMRES, BiCGSTAB
- **4 preconditioners**: ILU0, AMG, AMGCL (via [amgcl](https://github.com/ddemidov/amgcl)), Jacobi
- **Sparse matrix formats**: COO input → CSR/CSC internal storage
- **Matrix Market I/O**: Read/write `.mtx` files (coordinate format)
- **OpenMP** parallelization support
- **12 standard test matrices** from [NIST Matrix Market](https://math.nist.gov/MatrixMarket/)

## Project Structure

```
math/
├── src/                        # Library source
│   ├── sparse_matrix.{h,cpp}   # Sparse matrix (COO/CSR/CSC)
│   ├── vector.{h,cpp}          # Dense vector with BLAS-style ops
│   ├── solver.h                # Solver base class + config
│   ├── preconditioner.h        # Preconditioner interface
│   ├── solvers/                # CG, GMRES, BiCGSTAB
│   ├── preconditioners/        # ILU0, AMG, AMGCL, Jacobi
│   └── io/                     # Matrix Market reader/writer
├── tests/                      # 13 Google Test unit test suites
│   └── CMakeLists.txt
├── benchmark/                  # Performance benchmarks
│   ├── benchmark_matrix_market.cpp   # Per-matrix solver comparison
│   └── CMakeLists.txt
├── examples/                   # Usage examples
├── test_matrices/              # 12 .mtx files (SPD, non-symmetric, fluid)
├── third_party/                # GoogleTest, AMGCL (git submodules)
└── CMakeLists.txt
```

## Build

**Requirements:** C++17 compiler (g++ / clang++), CMake ≥ 3.20, OpenMP (optional)

```bash
git clone --recurse-submodules https://github.com/blog-guang/math.git
cd math
mkdir build && cd build
cmake ..
cmake --build . -j$(nproc)
```

## Quick Start

```cpp
#include "sparse_matrix.h"
#include "solvers/cg.h"
#include "preconditioners/ilu0.h"
#include "io/matrix_market.h"

using namespace math;

int main() {
    // 从 Matrix Market 文件读取矩阵
    SparseMatrix A = math::io::MatrixMarket::readMatrix("test_matrices/bcsstk06_spd_420.mtx");
    size_t n = A.rows();

    // 构造右端向量
    Vector b(n, 1.0);
    Vector x(n, 0.0);   // 初始猜测

    // 配置求解器
    SolverConfig cfg;
    cfg.tol      = 1e-10;
    cfg.max_iter = 5000;

    // ILU0 预条件 CG 求解
    ILU0Preconditioner ilu0;
    ilu0.build(A);
    cfg.precond = &ilu0;

    CGSolver cg;
    auto result = cg.solve(A, b, x, cfg);

    printf("converged=%d  iterations=%d  residual=%.2e\n",
           result.converged, result.iterations, result.relative_residual);
    return 0;
}
```

## 求解器选择指南

| 矩阵类型 | 推荐求解器 | 推荐预条件器 |
|---|---|---|
| SPD（对称正定） | CG | ILU0（首选）/ AMG |
| 非对称（一般） | GMRES / BiCGSTAB | ILU0 / AMGCL |
| 非对称（高难度） | GMRES | AMGCL |
| 结构奇异（零对角） | — | 需要行置换预处理 |

## Benchmark 结果

对 12 个 Matrix Market 标准矩阵的求解器对比（5000 iter 上限）：

### SPD 矩阵 — CG 系列

| 矩阵 (N) | CG | CG+ILU0 | CG+AMG |
|---|---|---|---|
| bcsstk01 (48) | 147 iter · 0.1 ms ✓ | **16 iter · 0.0 ms** ✓ | 1 iter · 0.1 ms ✓ |
| bcsstk03 (112) | 658 iter · 0.9 ms ✓ | **15 iter · 0.0 ms** ✓ | 71 iter · 3.7 ms ✓ |
| bcsstk06 (420) | 4536 iter · 55 ms ✓ | **44 iter · 1.3 ms** ✓ | 142 iter · 29.7 ms ✓ |
| bcsstk08 (1074) | ✗ 未收敛 | **27 iter · 1.8 ms** ✓ | ✗ |
| bcsstk13 (2003) | ✗ | ✗ | ✗ | **CG+AMGCL** ✓ (1 iter)

### 非对称矩阵 — GMRES / BiCGSTAB 系列

| 矩阵 (N) | GMRES | BiCGSTAB | +ILU0 | +AMGCL |
|---|---|---|---|---|
| sherman1 (1000) | 781 · 300 ms ✓ | 431 · 9.9 ms ✓ | 42 · 3.7 ms ✓ | **1 · 0.6 ms** ✓ |
| sherman2 (1080) | ✗ | ✗ | ✗ | **1 · 3.3 ms** ✓ |
| sherman4 (1104) | 123 · 26.9 ms ✓ | 93 · 2.2 ms ✓ | ✗ | **1 · 0.4 ms** ✓ |
| sherman5 (3312) | ✗ | **3157 · 316 ms** ✓ | ✗ | ✗ |

> **AMGCL 预条件 GMRES** 是非对称矩阵的最强组合，几乎所有矩阵 1 iter 即收敛。
> **ILU0** 加速比 10~100x，但对部分矩阵会退化。
> **LNS 系列**（结构奇异）当前无法求解，需行置换预处理。

## 运行 Benchmark

```bash
cd build

# 全部矩阵
./benchmark/benchmark_mm

# 过滤特定矩阵（支持子串匹配）
./benchmark/benchmark_mm sherman
./benchmark/benchmark_mm bcsstk
./benchmark/benchmark_mm poisson
```

## 单元测试

```bash
cd build
ctest --output-on-failure
# 或单独运行某个测试
./tests/gtest_cg
./tests/gtest_all_matrices   # 遍历全部 12 个矩阵
```

共 13 个 Google Test 测试套件，覆盖：向量运算、矩阵操作、CSC 转换、各求解器、预条件器、Matrix Market I/O、OpenMP 并行。

## Test Matrix 集合

`test_matrices/` 包含 26 个标准矩阵，来源：

| 集合 | 矩阵 | 类型 | 规模 | 应用领域 |
|---|---|---|---|---|
| BCSSTK | bcsstk01/03/06/08/13 | SPD | 48 ~ 2003 | 结构工程刚度矩阵 |
| Sherman | sherman1 ~ sherman5 | 对称/非对称 | 1000 ~ 3312 | 油田模拟 (Sherman Challenge) |
| LNS | lns131/511/3937 | 非对称（零对角） | 131 ~ 3937 | 流体力学 |

## 依赖

| 依赖 | 版本 | 用途 |
|---|---|---|
| C++ 编译器 | C++17 | — |
| CMake | ≥ 3.20 | 构建系统 |
| OpenMP | — | 并行加速（可选） |
| [GoogleTest](https://github.com/google/googletest) | submodule | 单元测试框架 |
| [AMGCL](https://github.com/ddemidov/amgcl) | submodule | AMG 预条件器 |

## License

This project is open source. See the repository for license details.
