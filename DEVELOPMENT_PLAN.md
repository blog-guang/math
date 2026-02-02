# Math Library — 大规模线性方程组迭代求解器开发计划

**版本：** 2.0  
**日期：** 2026-02-02  
**状态：** 规划阶段

---

## 执行摘要

本文档规划了 Math 库从当前原型（v1.0）向生产级大规模稀疏线性方程组求解器（v2.0）的演进路径。当前实现已具备核心求解器（CG、GMRES、BiCGSTAB）、预条件器（ILU0、AMG、AMGCL）和重排序算法（RCM），经过 12 个 Matrix Market 标准矩阵的验证。v2.0 目标是构建工业级求解器框架，支持并行计算、自适应算法选择、GPU 加速和多语言绑定。

**关键里程碑：**
- Phase 1 (2 周): 求解器族扩展 + 性能基准
- Phase 2 (3 周): 预条件器增强 + 并行化
- Phase 3 (2 周): 自适应框架 + Python 绑定
- Phase 4 (1 周): 文档、示例和发布准备

**预期成果：** 一个能处理千万级自由度、支持多核/GPU 并行、自动选择最优算法的生产级稀疏线性求解器库。

---

## 1. 当前状态评估 (v1.0)

### 1.1 已实现功能

| 组件 | 实现 | 状态 | 覆盖率 |
|---|---|---|---|
| **核心求解器** | CG, GMRES, BiCGSTAB | ✓ | 80% |
| **预条件器** | Jacobi, ILU0, AMG, AMGCL | ✓ | 75% |
| **重排序** | RCM, diagonal permutation | ✓ | 90% |
| **矩阵格式** | COO, CSR, CSC | ✓ | 85% |
| **I/O** | Matrix Market 读写 | ✓ | 95% |
| **测试** | 13 个单元测试套件 | ✓ | 70% |
| **示例** | 4 个应用案例 | ✓ | 60% |
| **文档** | README + 注释 | ✓ | 50% |

### 1.2 性能基准 (当前)

| 矩阵类型 | 规模 | 方法 | 迭代 | 时间 | 状态 |
|---|---|---|---|---|---|
| bcsstk13 (SPD) | 2003 | CG+AMGCL | 1 | 6ms | ✓ |
| sherman1 (非对称) | 1000 | GMRES+AMGCL | 1 | 1.2ms | ✓ |
| lns131 (近奇异) | 131 | GMRES+ILU0+Reg | 3 | 0.1ms | ✓* |

*需要正则化

### 1.3 识别的问题

1. **求解器单一**：仅 3 个 Krylov 方法，缺少 QMR/CGS/TFQMR/IDR(s)
2. **预条件器局限**：ILU0 对病态矩阵失效，缺少 ILUT/SSOR/Polynomial
3. **无并行支持**：纯串行，未利用多核/SIMD/GPU
4. **手动调参**：需人工选择求解器和参数，缺少自适应机制
5. **收敛诊断薄弱**：无历史追踪、残差图表、失败分析
6. **扩展性差**：添加新求解器需修改多处代码
7. **内存效率**：大矩阵时内存拷贝频繁
8. **用户接口**：仅 C++ API，无 Python/Julia/MATLAB 绑定

---

## 2. 架构设计 (v2.0)

### 2.1 分层架构

```
┌─────────────────────────────────────────────────────────────┐
│                    User Interface Layer                       │
│  C++ API │ Python Bindings │ CLI Tool │ Config Files         │
├─────────────────────────────────────────────────────────────┤
│                  Adaptive Solver Framework                    │
│  Auto-Selection │ Param Tuning │ Convergence Monitor         │
├─────────────────────────────────────────────────────────────┤
│                   Solver Registry Layer                       │
│  Krylov: CG│GMRES│BiCGSTAB│QMR│CGS│TFQMR│IDR(s)│Chebyshev  │
│  Stationary: Jacobi│GS│SOR│SSOR                              │
│  Direct (fallback): SuperLU│UMFPACK (optional)               │
├─────────────────────────────────────────────────────────────┤
│                 Preconditioner Registry                       │
│  ILU: ILU0│ILUT│ILUTP                                        │
│  AMG: AMG│AMGCL│SA-AMG                                       │
│  Polynomial: Jacobi│Chebyshev│SPAI                           │
│  Domain-specific: Block│Schwarz│Schur                        │
├─────────────────────────────────────────────────────────────┤
│                  Matrix Operations Layer                      │
│  Sparse: COO│CSR│CSC│BSR│ELL│HYB                            │
│  Dense (small blocks): BLAS/LAPACK interface                 │
│  Reordering: RCM│AMD│METIS│Nested Dissection                │
├─────────────────────────────────────────────────────────────┤
│                   Parallel Execution Layer                    │
│  Threading: OpenMP │ TBB │ std::thread                       │
│  Distributed: MPI (Phase 5, optional)                        │
│  GPU: CUDA/HIP/SYCL (Phase 5, optional)                      │
├─────────────────────────────────────────────────────────────┤
│                     Utilities Layer                           │
│  Vector ops │ Timer │ Logger │ Profiler │ Memory pool        │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 核心设计原则

1. **插件化**：求解器和预条件器通过注册表动态加载
2. **零拷贝**：矩阵格式按需转换，避免不必要的内存分配
3. **RAII**：资源管理自动化，无手动 delete
4. **策略模式**：收敛准则、停止条件、参数调整可配置
5. **观察者模式**：迭代过程通过回调通知（用于监控/可视化）
6. **工厂模式**：统一的求解器/预条件器创建接口

### 2.3 接口示例 (v2.0)

#### 2.3.1 简单用例 (自动求解)

```cpp
#include <math/solver.h>

// 自动选择最优方法
auto result = math::solve(A, b, x);  // 一行搞定
```

#### 2.3.2 高级用例 (手动配置)

```cpp
#include <math/solver_builder.h>

auto solver = math::SolverBuilder()
    .matrix(A)
    .solver("gmres")              // 或 "auto"
    .preconditioner("amgcl")      // 或 "auto"
    .tolerance(1e-8)
    .maxIterations(5000)
    .restart(100)                 // GMRES restart
    .threads(8)                   // OpenMP threads
    .monitor([](const IterInfo& info) {
        std::cout << "iter " << info.iteration 
                  << " res " << info.residual << "\n";
    })
    .build();

auto result = solver.solve(b, x);
```

#### 2.3.3 Python 绑定

```python
import pymath

A = pymath.read_matrix("A.mtx")
b = np.ones(A.rows())

# 自动求解
x, info = pymath.solve(A, b, tol=1e-8)

# 手动配置
solver = pymath.GMRESSolver(
    precond=pymath.AMGCL(),
    tol=1e-10,
    restart=150
)
x, info = solver.solve(A, b)
print(f"Converged: {info.converged}, iters: {info.iterations}")
```

---

## 3. 开发路线图

### Phase 1: 求解器族扩展 + 性能基准 (2 周)

**目标**：补齐常用 Krylov 方法，建立全面的性能基准。

#### 任务列表

| ID | 任务 | 优先级 | 工作量 | 负责 |
|---|---|---|---|---|
| S1.1 | 实现 QMR 求解器 | P1 | 3d | - |
| S1.2 | 实现 TFQMR 求解器 | P1 | 3d | - |
| S1.3 | 实现 CGS 求解器 | P2 | 2d | - |
| S1.4 | 实现 IDR(s) 求解器 | P2 | 4d | - |
| S1.5 | 实现 Chebyshev 迭代 | P2 | 2d | - |
| S1.6 | 恢复 GS / SOR（可选） | P3 | 1d | - |
| S1.7 | 扩展 benchmark 到 50+ 矩阵 | P1 | 2d | - |
| S1.8 | 建立性能回归测试 | P1 | 2d | - |

**交付物**：
- 8 个求解器（CG, GMRES, BiCGSTAB, QMR, TFQMR, CGS, IDR(s), Chebyshev）
- 50+ 矩阵的性能数据库
- 自动化回归测试脚本

**验收标准**：
- 所有新求解器通过单元测试
- Benchmark 覆盖 SPD、非对称、近奇异三类矩阵
- 与参考实现（PETSc/Trilinos）的迭代次数误差 < 5%

---

### Phase 2: 预条件器增强 + 并行化 (3 周)

**目标**：提升病态矩阵求解能力，引入并行计算。

#### 任务列表

| ID | 任务 | 优先级 | 工作量 | 负责 |
|---|---|---|---|---|
| P2.1 | 实现 ILUT (threshold ILU) | P1 | 4d | - |
| P2.2 | 实现 SSOR 预条件器 | P1 | 2d | - |
| P2.3 | 实现 Polynomial 预条件 | P2 | 3d | - |
| P2.4 | 实现 Block Jacobi | P2 | 3d | - |
| P2.5 | 集成 METIS/AMD 重排序 | P2 | 3d | - |
| P2.6 | OpenMP 并行 SpMV | P1 | 3d | - |
| P2.7 | OpenMP 并行预条件器 | P1 | 4d | - |
| P2.8 | 内存池优化 | P2 | 2d | - |
| P2.9 | SIMD 优化（AVX2/NEON） | P3 | 4d | - |

**交付物**：
- 4 个新预条件器（ILUT, SSOR, Polynomial, Block Jacobi）
- METIS/AMD 集成
- OpenMP 并行版本（SpMV + precond）
- 并行加速比测试报告

**验收标准**：
- ILUT 在病态矩阵上迭代次数减少 > 50%
- OpenMP 在 8 核上加速比 > 4x
- 内存使用减少 > 20%（通过内存池）

---

### Phase 3: 自适应框架 + Python 绑定 (2 周)

**目标**：实现自动算法选择，提供 Python 接口。

#### 任务列表

| ID | 任务 | 优先级 | 工作量 | 负责 |
|---|---|---|---|---|
| A3.1 | 矩阵特征探测（SPD/对称/带宽/条件数） | P1 | 3d | - |
| A3.2 | 求解器/预条件器匹配规则库 | P1 | 4d | - |
| A3.3 | 自适应参数调整（tolerance/restart） | P2 | 3d | - |
| A3.4 | 收敛失败时的策略切换 | P1 | 3d | - |
| A3.5 | pybind11 Python 绑定 | P1 | 4d | - |
| A3.6 | NumPy/SciPy 互操作 | P1 | 2d | - |
| A3.7 | Jupyter 演示 notebook | P2 | 1d | - |

**交付物**：
- `math::solve()` 自动求解接口
- Python `pymath` 包（PyPI 可安装）
- 5 个 Jupyter notebook 示例

**验收标准**：
- 自动求解在 90% 测试矩阵上选择正确算法
- Python 绑定性能开销 < 5%
- Notebook 覆盖：快速入门、性能对比、自定义预条件器、并行计算、收敛分析

---

### Phase 4: 文档 + 示例 + 发布 (1 周)

**目标**：完善文档，准备公开发布。

#### 任务列表

| ID | 任务 | 优先级 | 工作量 | 负责 |
|---|---|---|---|---|
| D4.1 | API 参考文档（Doxygen） | P1 | 2d | - |
| D4.2 | 用户手册（安装/快速入门/进阶） | P1 | 2d | - |
| D4.3 | 开发者指南（贡献/架构/添加求解器） | P2 | 1d | - |
| D4.4 | 性能调优指南 | P2 | 1d | - |
| D4.5 | 10 个应用示例 | P1 | 2d | - |
| D4.6 | CI/CD 配置（GitHub Actions） | P1 | 1d | - |
| D4.7 | 包管理配置（Conan/vcpkg） | P2 | 1d | - |

**交付物**：
- 完整文档站点（ReadTheDocs 或 GitHub Pages）
- 10 个领域应用示例（结构工程、流体、电磁、金融等）
- 自动化测试/发布流水线

**验收标准**：
- 文档覆盖率 > 90%
- 示例可独立编译运行
- CI 覆盖 Linux/macOS/Windows

---

### Phase 5: 高级特性 (可选, 4 周)

仅在 v2.0 基础需求完成后，根据用户反馈和资源情况启动。

| 特性 | 工作量 | 优先级 |
|---|---|---|
| GPU 加速（CUDA/HIP） | 2 周 | P2 |
| MPI 分布式求解 | 2 周 | P2 |
| 块稀疏矩阵格式（BSR） | 1 周 | P3 |
| 非线性求解器（Newton-Krylov） | 2 周 | P3 |
| 时间步进框架（ODE/DAE） | 3 周 | P3 |
| Julia/MATLAB 绑定 | 1 周 | P3 |

---

## 4. 技术规范

### 4.1 编码标准

- **语言**：C++17 (考虑 C++20 concepts)
- **风格**：Google C++ Style Guide
- **测试**：Google Test, 覆盖率 > 80%
- **文档**：Doxygen + Markdown
- **构建**：CMake 3.20+, 支持 FetchContent
- **依赖管理**：git submodule 或 Conan

### 4.2 性能目标

| 指标 | v1.0 | v2.0 目标 |
|---|---|---|
| 矩阵规模 | < 10K | > 10M |
| SpMV 性能 | 1 GFlops | > 10 GFlops (多核) |
| 内存占用 | 矩阵 × 5 | 矩阵 × 2 |
| 并行效率 | N/A | > 60% (8 核) |
| 启动开销 | ~ 100ms | < 10ms |

### 4.3 兼容性

- **平台**：Linux (主), macOS, Windows (MinGW / MSVC)
- **编译器**：GCC 9+, Clang 10+, MSVC 2019+
- **架构**：x86-64, ARM64 (M1/M2)
- **依赖**：
  - 必需：OpenBLAS / MKL (可选)
  - 可选：METIS, SuiteSparse, AMGCL

### 4.4 质量保证

- **CI**：每次 PR 运行全部测试 + 性能回归检查
- **Sanitizers**：ASan, UBSan, TSan (线程安全)
- **Static Analysis**：clang-tidy, cppcheck
- **Fuzzing**：矩阵输入 fuzzing（libFuzzer）
- **Coverage**：Codecov 报告，目标 > 85%

---

## 5. 资源规划

### 5.1 人力需求

| 角色 | 技能要求 | 投入 |
|---|---|---|
| 核心开发者 | C++17, 数值算法, 线性代数 | 全职 × 1 |
| 并行计算工程师 | OpenMP, SIMD, GPU (可选) | 兼职 × 1 |
| Python 开发者 | pybind11, NumPy, 打包 | 兼职 × 1 |
| 文档工程师 | 技术写作, Doxygen, Markdown | 兼职 × 1 |

### 5.2 时间表

```
Week 1-2:   Phase 1 (求解器扩展)
Week 3-5:   Phase 2 (预条件器 + 并行)
Week 6-7:   Phase 3 (自适应 + Python)
Week 8:     Phase 4 (文档 + 发布)
Week 9-12:  Phase 5 (可选高级特性)
```

**里程碑检查点**：
- Week 2: 求解器族完成，benchmark 数据可用
- Week 5: 并行版本性能达标
- Week 7: Python 包可用，自动求解可用
- Week 8: v2.0 正式发布

### 5.3 风险评估

| 风险 | 概率 | 影响 | 缓解措施 |
|---|---|---|---|
| 并行加速比不达标 | 中 | 高 | 提前进行性能剖析，优化热点 |
| ILUT 收敛性不稳定 | 中 | 中 | 参考 PETSc 实现，增加 dropping 策略 |
| Python 绑定性能开销大 | 低 | 中 | 使用 buffer protocol，减少拷贝 |
| METIS 许可证冲突 | 低 | 低 | 提供可选依赖，或用 AMD 替代 |
| 内存泄漏 | 中 | 高 | 使用 RAII，ASan/Valgrind 检查 |

---

## 6. 成功度量

### 6.1 定量指标

| 指标 | 基线 (v1.0) | 目标 (v2.0) |
|---|---|---|
| 求解器数量 | 3 | 8+ |
| 预条件器数量 | 4 | 8+ |
| 测试矩阵数量 | 12 | 50+ |
| 代码覆盖率 | 70% | 85% |
| 文档覆盖率 | 50% | 90% |
| 性能提升（多核） | 1x | 5x+ |
| 用户数量 | 0 | 100+ (GitHub stars) |

### 6.2 定性目标

- **易用性**：新用户 < 10 分钟完成首次求解
- **鲁棒性**：自动求解在常见矩阵上成功率 > 90%
- **可扩展性**：添加新求解器 < 1 天
- **社区活跃度**：月均 issue/PR > 5

---

## 7. 下一步行动

### 7.1 立即启动 (本周)

1. ✅ 创建 `dev/phase1` 分支
2. ✅ 设置 GitHub Projects 看板
3. ✅ 编写 QMR 求解器原型
4. ⏳ 下载 SuiteSparse Matrix Collection 子集（50 个矩阵）

### 7.2 本月目标

- 完成 Phase 1 所有任务
- 发布 v1.1-alpha（包含 QMR/TFQMR/CGS）
- 建立 CI 流水线

### 7.3 联系与协作

- **项目主页**：https://github.com/blog-guang/math
- **文档**：https://math.readthedocs.io (待建)
- **讨论**：GitHub Discussions
- **问题跟踪**：GitHub Issues

---

## 附录 A: 参考实现对比

| 特性 | Math v2.0 | PETSc | Trilinos | Eigen | SciPy |
|---|---|---|---|---|---|
| 语言 | C++17 | C | C++ | C++ | Python |
| 求解器 | 8+ | 30+ | 50+ | 10+ | 15+ |
| 并行 | OpenMP | MPI | MPI+OpenMP | OpenMP | - |
| GPU | (Phase 5) | CUDA | CUDA | - | - |
| 易用性 | ★★★★★ | ★★★ | ★★ | ★★★★ | ★★★★★ |
| 性能 | ★★★★ | ★★★★★ | ★★★★★ | ★★★ | ★★★ |
| 文档 | ★★★★ | ★★★★ | ★★★ | ★★★★★ | ★★★★ |

**定位**：介于 Eigen（简单但功能少）和 PETSc（强大但复杂）之间，强调易用性和现代 C++ 设计。

---

## 附录 B: 术语表

| 术语 | 定义 |
|---|---|
| **SPD** | Symmetric Positive Definite（对称正定） |
| **Krylov 方法** | 基于 Krylov 子空间的迭代法（CG, GMRES, BiCGSTAB 等） |
| **预条件器** | 变换矩阵 M ≈ A^{-1}，加速迭代收敛 |
| **SpMV** | Sparse Matrix-Vector Multiplication（稀疏矩阵向量乘） |
| **ILU** | Incomplete LU Factorization（不完全 LU 分解） |
| **AMG** | Algebraic Multigrid（代数多重网格） |
| **RCM** | Reverse Cuthill-McKee（反向 Cuthill-McKee 重排序） |
| **Saddle-point** | 鞍点系统（来自流体/电磁等混合问题） |

---

**文档版本控制**：
- v1.0 (2026-02-02): 初始版本
- 下次审查：Phase 1 完成后

**批准签字**：
- 技术负责人：_________________  日期：________
- 项目经理：    _________________  日期：________
