/**
 * bcsstk_amgcl_all.cpp
 * 用 CG+AMGCL 对全部 bcsstk 系列矩阵求解，验证通用性。
 */
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>
#include <algorithm>

#include "io/matrix_market.h"
#include "preconditioners/amgcl_wrapper.h"
#include "solvers/cg.h"
#include "sparse_matrix.h"
#include "vector.h"

using namespace math;
using namespace math::io;
namespace fs = std::filesystem;

int main() {
    fs::path dir = "test_matrices";
    if (!fs::is_directory(dir)) dir = "../test_matrices";
    if (!fs::is_directory(dir)) { puts("test_matrices/ not found"); return 1; }

    std::vector<std::string> files;
    for (auto& e : fs::directory_iterator(dir)) {
        std::string name = e.path().filename().stem().string();
        if (name.find("bcsstk") != std::string::npos)
            files.push_back(name);
    }
    std::sort(files.begin(), files.end());

    printf("%-32s %6s %6s %10s %12s  %s\n",
           "Matrix", "N", "Iter", "ms", "||r||/||b||", "Status");
    printf("%-32s %6s %6s %10s %12s  %s\n",
           "--------------------------------", "------", "------", "----------", "------------", "------");

    int total = 0, passed = 0;
    CGSolver cg;

    for (auto& name : files) {
        fs::path filepath = dir / (name + ".mtx");
        SparseMatrix A = MatrixMarket::readMatrix(filepath.string());
        size_t n = A.rows();
        A.toCSR();

        Vector b(n);
        for (size_t i = 0; i < n; ++i) b[i] = std::sin(static_cast<double>(i + 1));
        double b_norm = b.norm();

        AMGCLPreconditioner amgcl;
        amgcl.build(A);

        SolverConfig cfg;
        cfg.tol      = 1e-8;
        cfg.max_iter = 10000;
        cfg.precond  = &amgcl;

        Vector x(n, 0.0);
        auto t0 = std::chrono::steady_clock::now();
        auto result = cg.solve(A, b, x, cfg);
        auto t1 = std::chrono::steady_clock::now();

        Vector r = b - A.multiply(x);
        double rel_res = (b_norm > 0.0) ? r.norm() / b_norm : r.norm();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        bool ok = result.converged && rel_res < 1e-4;

        printf("%-32s %6zu %6d %10.1f %12.2e  %s\n",
               name.c_str(), n, result.iterations, ms, rel_res,
               ok ? "✓ OK" : "✗ FAIL");

        ++total;
        if (ok) ++passed;
    }

    printf("\n%d/%d passed\n", passed, total);
    return (passed == total) ? 0 : 1;
}
