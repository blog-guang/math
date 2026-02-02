# ---------------------------------------------------------------------------
# Unit Tests (Google Test)
# ---------------------------------------------------------------------------

set(GTEST_TARGETS
    gtest_vector
    gtest_matrix
    gtest_cg
    gtest_gmres
    gtest_bicgstab
    gtest_precond
    gtest_ilu0
    gtest_amg
    gtest_amgcl
    gtest_io
    gtest_csc
    gtest_openmp
    gtest_all_matrices
)

foreach(target ${GTEST_TARGETS})
    add_executable(${target} ${target}.cpp)
    target_link_libraries(${target} math gtest_main)
endforeach()

# ── ctest 注册 ──────────────────────────────────────────

add_test(NAME vector_tests      COMMAND gtest_vector)
add_test(NAME matrix_tests      COMMAND gtest_matrix)
add_test(NAME cg_tests          COMMAND gtest_cg)
add_test(NAME gmres_tests       COMMAND gtest_gmres)
add_test(NAME bicgstab_tests    COMMAND gtest_bicgstab)
add_test(NAME precond_tests     COMMAND gtest_precond)
add_test(NAME ilu0_tests        COMMAND gtest_ilu0)
add_test(NAME amg_tests         COMMAND gtest_amg)
add_test(NAME amgcl_tests       COMMAND gtest_amgcl)
add_test(NAME io_tests          COMMAND gtest_io)
add_test(NAME csc_tests         COMMAND gtest_csc)
add_test(NAME openmp_tests      COMMAND gtest_openmp)
add_test(NAME all_matrices_tests COMMAND gtest_all_matrices
         WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
