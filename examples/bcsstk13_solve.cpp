/**
 * bcsstk13_solve.cpp
 *
 * Industry-standard approaches for solving the challenging bcsstk13 matrix.
 * This matrix is a 2003x2003 SPD generalized eigenvalue problem from fluid flow,
 * known for having an extremely high condition number that stumps basic iterative methods.
 */

#include <iostream>
#include <fstream>
#include <chrono>

#include "io/matrix_market.h"
#include "preconditioners/amgcl_wrapper.h"
#include "solvers/cg.h"
#include "solvers/gmres.h"
#include "solvers/bicgstab.h"
#include "vector.h"

using namespace math;
using namespace math::io;

int main() {
    std::cout << "Solving bcsstk13_spd_2003 (2003x2003, fluid flow generalized eigenvalue)\n";
    std::cout << "This is a notoriously difficult matrix with extremely high condition number.\n\n";

    // Load the matrix and ensure it's in CSR format for AMGCL
    SparseMatrix A = MatrixMarket::readMatrix("test_matrices/bcsstk13_spd_2003.mtx");
    A.toCSR();
    size_t n = A.rows();
    std::cout << "Matrix loaded: " << n << "x" << n << ", nnz = " << A.nnz() << "\n";

    // Create RHS vector (using sine pattern for numerical stability)
    Vector b(n);
    for (size_t i = 0; i < n; ++i) {
        b[i] = std::sin(static_cast<double>(i + 1));
    }
    double b_norm = b.norm();
    std::cout << "||b|| = " << b_norm << "\n\n";

    // Initial guess
    Vector x(n, 0.0);

    SolverConfig cfg;
    cfg.tol = 1e-8;           // Standard tolerance
    cfg.max_iter = 10000;     // Higher iteration limit for difficult problems

    // ── Method 1: CG with AMGCL (default settings) ──
    std::cout << "Method 1: CG + AMGCL (default settings)\n";
    {
        AMGCLPreconditioner amgcl;
        try {
            amgcl.build(A);
            cfg.precond = &amgcl;

            CGSolver cg;
            auto t0 = std::chrono::steady_clock::now();
            auto result = cg.solve(A, b, x, cfg);
            auto t1 = std::chrono::steady_clock::now();
            double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

            std::cout << "  CG+AMGCL: ";
            if (result.converged) {
                std::cout << "✓ converged in " << result.iterations << " iters, " << ms << " ms\n";
                
                // Verify residual independently
                Vector r = b - A.multiply(x);
                double res_norm = r.norm();
                double rel_res = (b_norm > 0.0) ? res_norm / b_norm : res_norm;
                std::cout << "  Verified residual: ||r||=" << res_norm << ", ||r||/||b||=" << rel_res << "\n";
            } else {
                std::cout << "✗ failed to converge after " << result.iterations << " iters\n";
                std::cout << "  Final residual: " << result.relative_residual << "\n";
            }
        } catch (const std::exception& e) {
            std::cout << "  ✗ AMGCL build failed: " << e.what() << "\n";
        }
        cfg.precond = nullptr;  // Reset
    }

    // ── Method 2: GMRES with AMGCL (more robust for tough problems) ──
    std::cout << "\nMethod 2: GMRES + AMGCL (more robust)\n";
    {
        Vector x_gmres(n, 0.0);  // Fresh initial guess
        AMGCLPreconditioner amgcl;
        try {
            amgcl.build(A);
            cfg.precond = &amgcl;
            cfg.gmres_restart = 100;  // Larger restart for difficult problems

            GMRESSolver gmres;
            auto t0 = std::chrono::steady_clock::now();
            auto result = gmres.solve(A, b, x_gmres, cfg);
            auto t1 = std::chrono::steady_clock::now();
            double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

            std::cout << "  GMRES+AMGCL: ";
            if (result.converged) {
                std::cout << "✓ converged in " << result.iterations << " iters, " << ms << " ms\n";
                
                Vector r = b - A.multiply(x_gmres);
                double res_norm = r.norm();
                double rel_res = (b_norm > 0.0) ? res_norm / b_norm : res_norm;
                std::cout << "  Verified residual: ||r||=" << res_norm << ", ||r||/||b||=" << rel_res << "\n";
                
                x = x_gmres;  // Use this solution
            } else {
                std::cout << "✗ failed to converge after " << result.iterations << " iters\n";
                std::cout << "  Final residual: " << result.relative_residual << "\n";
            }
        } catch (const std::exception& e) {
            std::cout << "  ✗ AMGCL build failed: " << e.what() << "\n";
        }
        cfg.precond = nullptr;
    }

    // ── Method 3: BiCGSTAB with AMGCL ──
    std::cout << "\nMethod 3: BiCGSTAB + AMGCL\n";
    {
        Vector x_bicgstab(n, 0.0);  // Fresh initial guess
        AMGCLPreconditioner amgcl;
        try {
            amgcl.build(A);
            cfg.precond = &amgcl;

            BiCGSTABSolver bicgstab;
            auto t0 = std::chrono::steady_clock::now();
            auto result = bicgstab.solve(A, b, x_bicgstab, cfg);
            auto t1 = std::chrono::steady_clock::now();
            double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

            std::cout << "  BiCGSTAB+AMGCL: ";
            if (result.converged) {
                std::cout << "✓ converged in " << result.iterations << " iters, " << ms << " ms\n";
                
                Vector r = b - A.multiply(x_bicgstab);
                double res_norm = r.norm();
                double rel_res = (b_norm > 0.0) ? res_norm / b_norm : res_norm;
                std::cout << "  Verified residual: ||r||=" << res_norm << ", ||r||/||b||=" << rel_res << "\n";
                
                x = x_bicgstab;  // Use this solution
            } else {
                std::cout << "✗ failed to converge after " << result.iterations << " iters\n";
                std::cout << "  Final residual: " << result.relative_residual << "\n";
            }
        } catch (const std::exception& e) {
            std::cout << "  ✗ AMGCL build failed: " << e.what() << "\n";
        }
        cfg.precond = nullptr;
    }

    std::cout << "\nSolution summary:\n";
    Vector r_final = b - A.multiply(x);
    double final_res = r_final.norm();
    double final_rel_res = (b_norm > 0.0) ? final_res / b_norm : final_res;
    std::cout << "Final solution residual: ||r||=" << final_res << ", ||r||/||b||=" << final_rel_res << "\n";

    // Optionally save solution
    std::ofstream out("bcsstk13_solution.txt");
    out << "# Solution vector x for bcsstk13_spd_2003\n";
    for (size_t i = 0; i < n; ++i) {
        out << x[i] << "\n";
    }
    out.close();
    std::cout << "Solution saved to bcsstk13_solution.txt\n";

    return 0;
}