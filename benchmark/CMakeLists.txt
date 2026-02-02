# ---------------------------------------------------------------------------
# Benchmarks (standalone executables, not part of ctest)
# ---------------------------------------------------------------------------

# Matrix Market 矩阵 benchmark
# 用法: cd <build>; ./benchmark_mm [filter]
#   例如: ./benchmark_mm sherman   — 只跑 sherman 系列
add_executable(benchmark_mm  benchmark_matrix_market.cpp)
target_link_libraries(benchmark_mm math)
