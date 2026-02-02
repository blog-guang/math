# Math Library Roadmap

快速参考版本，详细规划见 [DEVELOPMENT_PLAN.md](DEVELOPMENT_PLAN.md)

---

## 当前状态：v1.0 ✓

**已完成**：
- ✅ 3 个 Krylov 求解器：CG, GMRES, BiCGSTAB
- ✅ 4 个预条件器：Jacobi, ILU0, AMG, AMGCL
- ✅ 重排序算法：RCM, diagonal permutation
- ✅ Matrix Market I/O
- ✅ 12 个测试矩阵验证
- ✅ 13 个单元测试套件

**性能亮点**：
- bcsstk13 (2003×2003 SPD): CG+AMGCL 1 iter, 6ms
- sherman1 (1000×1000): GMRES+AMGCL 1 iter, 1.2ms
- lns 系列: RCM 带宽缩减 2.6x~19.5x

---

## Phase 1: 求解器族扩展 (Week 1-2)

### 目标
补齐常用 Krylov 方法，建立全面性能基准。

### 任务
- [ ] **QMR** 求解器 (3d)
- [ ] **TFQMR** 求解器 (3d)
- [ ] **CGS** 求解器 (2d)
- [ ] **IDR(s)** 求解器 (4d)
- [ ] **Chebyshev** 迭代 (2d)
- [ ] 扩展 benchmark 到 50+ 矩阵 (2d)
- [ ] 性能回归测试 (2d)

### 交付
- 8 个求解器全覆盖
- 50+ 矩阵性能数据库
- 自动化回归测试脚本

---

## Phase 2: 预条件器 + 并行 (Week 3-5)

### 目标
病态矩阵求解能力提升，多核并行加速。

### 任务
- [ ] **ILUT** (threshold ILU) (4d)
- [ ] **SSOR** 预条件器 (2d)
- [ ] **Polynomial** 预条件 (3d)
- [ ] **Block Jacobi** (3d)
- [ ] 集成 **METIS/AMD** 重排序 (3d)
- [ ] **OpenMP** 并行 SpMV (3d)
- [ ] **OpenMP** 并行预条件器 (4d)
- [ ] 内存池优化 (2d)

### 交付
- 4 个新预条件器
- METIS/AMD 集成
- OpenMP 并行版本（8 核加速 > 4x）

---

## Phase 3: 自适应 + Python (Week 6-7)

### 目标
自动算法选择，Python 生态集成。

### 任务
- [ ] 矩阵特征探测（SPD/对称/带宽/条件数） (3d)
- [ ] 求解器/预条件器匹配规则库 (4d)
- [ ] 自适应参数调整 (3d)
- [ ] 收敛失败时策略切换 (3d)
- [ ] **pybind11** Python 绑定 (4d)
- [ ] NumPy/SciPy 互操作 (2d)
- [ ] Jupyter notebook 示例 (1d)

### 交付
- `math::solve()` 自动求解接口
- Python `pymath` 包（PyPI 可装）
- 5 个 Jupyter notebook

---

## Phase 4: 文档 + 发布 (Week 8)

### 目标
完善文档，公开发布 v2.0。

### 任务
- [ ] API 参考文档（Doxygen） (2d)
- [ ] 用户手册 (2d)
- [ ] 开发者指南 (1d)
- [ ] 性能调优指南 (1d)
- [ ] 10 个应用示例 (2d)
- [ ] CI/CD 配置 (1d)
- [ ] 包管理配置（Conan/vcpkg） (1d)

### 交付
- 完整文档站点
- 10 个领域应用示例
- 自动化 CI/CD 流水线

---

## Phase 5: 高级特性 (可选, Week 9-12)

根据用户反馈和资源情况，按需启动：

- [ ] **GPU 加速**（CUDA/HIP） (2 周)
- [ ] **MPI** 分布式求解 (2 周)
- [ ] **块稀疏矩阵**（BSR） (1 周)
- [ ] **非线性求解器**（Newton-Krylov） (2 周)
- [ ] **时间步进框架**（ODE/DAE） (3 周)
- [ ] Julia/MATLAB 绑定 (1 周)

---

## 里程碑检查点

- **Week 2**: ✓ 求解器族完成，benchmark 数据可用
- **Week 5**: ✓ 并行版本性能达标
- **Week 7**: ✓ Python 包可用，自动求解可用
- **Week 8**: 🚀 **v2.0 正式发布**

---

## 快速链接

- [详细开发计划](DEVELOPMENT_PLAN.md)
- [架构设计文档](docs/architecture.md) *(待创建)*
- [贡献指南](CONTRIBUTING.md) *(待创建)*
- [性能基准](benchmarks/README.md) *(待创建)*

---

**进度跟踪**：
- 总任务：48 项
- 已完成：0 项 (0%)
- 进行中：0 项
- 待开始：48 项

**最后更新**：2026-02-02
