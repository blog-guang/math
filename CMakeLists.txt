cmake_minimum_required(VERSION 3.20)
project(Math LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---------------------------------------------------------------------------
# Compiler flags
# ---------------------------------------------------------------------------
add_compile_options(-O2)
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    add_compile_options(-O3 -march=native)
endif()

# ---------------------------------------------------------------------------
# OpenMP
# ---------------------------------------------------------------------------
find_package(OpenMP)

# ---------------------------------------------------------------------------
# Static library: math
# ---------------------------------------------------------------------------
add_library(math STATIC
    src/vector.cpp
    src/sparse_matrix.cpp
    src/io/matrix_market.cpp
    src/solvers/solver_base.cpp
    src/solvers/cg.cpp
    src/solvers/gmres.cpp
    src/solvers/bicgstab.cpp
    src/reorder.cpp
    src/preconditioners/jacobi_precond.cpp
    src/preconditioners/ilu0.cpp
    src/preconditioners/amg.cpp
    src/preconditioners/amgcl_wrapper.cpp
)

target_include_directories(math PUBLIC
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/third_party/amgcl
)

if(OpenMP_CXX_FOUND)
    target_link_libraries(math PUBLIC OpenMP::OpenMP_CXX)
endif()

# ---------------------------------------------------------------------------
# Example
# ---------------------------------------------------------------------------
add_executable(solve_example examples/solve_example.cpp)
target_link_libraries(solve_example math)

add_executable(bcsstk13_solve examples/bcsstk13_solve.cpp)
target_link_libraries(bcsstk13_solve math)

add_executable(bcsstk_amgcl_all examples/bcsstk_amgcl_all.cpp)
target_link_libraries(bcsstk_amgcl_all math)

add_executable(lns_reorder_solve examples/lns_reorder_solve.cpp)
target_link_libraries(lns_reorder_solve math)






# ---------------------------------------------------------------------------
# Google Test (子目录测试需要先获取）
# ---------------------------------------------------------------------------
include(FetchContent)
set(FETCHCONTENT_QUIET OFF)
FetchContent_Declare(
  googletest
  SOURCE_DIR ${PROJECT_SOURCE_DIR}/third_party/googletest
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# ---------------------------------------------------------------------------
# 子目录
# ---------------------------------------------------------------------------
enable_testing()
add_subdirectory(tests)
add_subdirectory(benchmark)
