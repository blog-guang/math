#!/usr/bin/env python3
"""
Download matrices from SuiteSparse Matrix Collection using ssget utility
Since direct downloading may be restricted, let's create a test matrix generator instead
"""

import os
import numpy as np
from scipy import sparse
import random

def generate_test_matrices():
    """Generate various types of test matrices that mimic real-world scenarios"""
    
    # Create directory for test matrices
    matrix_dir = "test_matrices"
    os.makedirs(matrix_dir, exist_ok=True)
    
    print("Generating diverse test matrices...")
    
    matrices = []
    
    # 1. Symmetric Positive Definite matrices
    # Small (100x100)
    n = 100
    A = sparse.random(n, n, density=0.1, format='coo')
    A = A + A.T  # Make symmetric
    A = A @ A.T  # Ensure positive definite
    A = A + sparse.diags([n], [0], shape=(n, n))  # Ensure diagonal dominance
    matrices.append(("spd_small_100x100.mtx", A.tocsr()))
    
    # Medium (1000x1000)
    n = 1000
    A = sparse.random(n, n, density=0.05, format='coo')
    A = A + A.T  # Make symmetric
    A = A @ A.T  # Ensure positive definite
    A = A + sparse.diags([n], [0], shape=(n, n))  # Ensure diagonal dominance
    matrices.append(("spd_medium_1000x1000.mtx", A.tocsr()))
    
    # 2. Non-symmetric matrices
    # Small non-symmetric
    n = 100
    A = sparse.random(n, n, density=0.1, format='csr')
    # Add some asymmetry
    A_upper = sparse.triu(A, k=1)
    A_lower = sparse.tril(A, k=-1)
    A = A_upper - A_lower  # Create non-symmetric matrix
    A = A + sparse.diags([n]*n, [0], shape=(n, n))  # Ensure diagonal dominance
    matrices.append(("nonsym_small_100x100.mtx", A))
    
    # Medium non-symmetric
    n = 1000
    A = sparse.random(n, n, density=0.05, format='csr')
    A_upper = sparse.triu(A, k=1)
    A_lower = sparse.tril(A, k=-1)
    A = A_upper - A_lower  # Create non-symmetric matrix
    A = A + sparse.diags([n]*n, [0], shape=(n, n))  # Ensure diagonal dominance
    matrices.append(("nonsym_medium_1000x1000.mtx", A))
    
    # 3. Indefinite matrices
    n = 500
    A = sparse.random(n, n, density=0.1, format='coo')
    A = A + A.T  # Make symmetric
    # Make indefinite by ensuring some negative eigenvalues
    D = sparse.diags(np.random.choice([-1, 1], size=n), format='csr')
    A = D + A
    matrices.append(("indef_medium_500x500.mtx", A.tocsr()))
    
    # 4. Nearly singular matrices (ill-conditioned)
    n = 200
    A = sparse.random(n, n, density=0.1, format='coo')
    A = A + A.T  # Make symmetric
    A = A @ A.T  # Ensure positive definite
    # Add very small values to diagonal to make nearly singular
    small_val = 1e-15
    A = A + sparse.diags([small_val]*n, [0], shape=(n, n))
    matrices.append(("nearly_singular_200x200.mtx", A.tocsr()))
    
    # 5. Diagonally dominant matrices
    n = 300
    A = sparse.random(n, n, density=0.1, format='csr')
    # Make strongly diagonally dominant
    row_sums = np.array(A.sum(axis=1)).flatten()
    diag_vals = np.abs(row_sums) + 10  # Ensure diagonal dominance
    A = A + sparse.diags(diag_vals, [0], shape=(n, n))
    matrices.append(("diag_dom_300x300.mtx", A))
    
    # 6. Grid/Stencil matrices (like finite difference)
    # 2D 5-point stencil (like Poisson but as generated matrix)
    def create_2d_5point_stencil(n):
        """Create a 2D 5-point stencil matrix"""
        N = n * n
        row, col, data = [], [], []
        
        for i in range(n):
            for j in range(n):
                idx = i * n + j
                
                # Center
                row.append(idx)
                col.append(idx)
                data.append(4.0)  # Diagonal entry
                
                # Left neighbor
                if j > 0:
                    row.append(idx)
                    col.append(idx - 1)
                    data.append(-1.0)
                
                # Right neighbor
                if j < n - 1:
                    row.append(idx)
                    col.append(idx + 1)
                    data.append(-1.0)
                
                # Top neighbor
                if i > 0:
                    row.append(idx)
                    col.append(idx - n)
                    data.append(-1.0)
                
                # Bottom neighbor
                if i < n - 1:
                    row.append(idx)
                    col.append(idx + n)
                    data.append(-1.0)
        
        return sparse.coo_matrix((data, (row, col)), shape=(N, N))
    
    A = create_2d_5point_stencil(20)  # 400x400 matrix
    matrices.append(("poisson_2d_400x400.mtx", A.tocsr()))
    
    # 7. Random rectangular matrix converted to square by A^TA
    m, n = 800, 600
    A_rect = sparse.random(m, n, density=0.05, format='csr')
    A_square = A_rect.T @ A_rect  # This creates a symmetric positive semi-definite matrix
    # Add identity to make positive definite
    A_square = A_square + sparse.eye(n, format='csr') * 0.1
    matrices.append(("rect_to_square_600x600.mtx", A_square))
    
    # 8. Convection-diffusion type (non-symmetric)
    def create_convection_diffusion_1d(n, eps=0.01, conv_coeff=1.0):
        """Create a 1D convection-diffusion matrix"""
        h = 1.0 / (n + 1)
        
        # Diffusion term: central differences for -eps*u''
        diff_diag = sparse.diags([2*eps/(h*h)]*n, [0], shape=(n, n))
        diff_offdiag1 = sparse.diags([-eps/(h*h)]*(n-1), [1], shape=(n, n))
        diff_offdiag2 = sparse.diags([-eps/(h*h)]*(n-1), [-1], shape=(n, n))
        
        # Convection term: central differences for conv_coeff*u'
        conv_offdiag1 = sparse.diags([conv_coeff/(2*h)]*(n-1), [1], shape=(n, n))
        conv_offdiag2 = sparse.diags([-conv_coeff/(2*h)]*(n-1), [-1], shape=(n, n))
        
        A = diff_diag + diff_offdiag1 + diff_offdiag2 + conv_offdiag1 + conv_offdiag2
        return A.tocsr()
    
    A = create_convection_diffusion_1d(500)
    matrices.append(("conv_diff_1d_500x500.mtx", A))
    
    # 9. Semiconductor-type matrix
    def create_semiconductor_1d(n, diff_coeff=0.1, field=0.5, mob=0.1):
        """Create a simplified semiconductor equation matrix"""
        h = 1.0 / (n + 1)
        
        # Diffusion part
        diff_diag = sparse.diags([2*diff_coeff/(h*h)]*n, [0], shape=(n, n))
        diff_offdiag1 = sparse.diags([-diff_coeff/(h*h)]*(n-1), [1], shape=(n, n))
        diff_offdiag2 = sparse.diags([-diff_coeff/(h*h)]*(n-1), [-1], shape=(n, n))
        
        # Field/convection part
        conv_offdiag1 = sparse.diags([mob*field/(2*h)]*(n-1), [1], shape=(n, n))
        conv_offdiag2 = sparse.diags([-mob*field/(2*h)]*(n-1), [-1], shape=(n, n))
        
        A = diff_diag + diff_offdiag1 + diff_offdiag2 + conv_offdiag1 + conv_offdiag2
        # Add diagonal dominance
        A = A + sparse.eye(n, format='csr') * 0.1
        return A.tocsr()
    
    A = create_semiconductor_1d(400)
    matrices.append(("semiconductor_1d_400x400.mtx", A))
    
    # 10. Wide band matrix
    def create_wide_band(n, bandwidth=20):
        """Create a matrix with wide band structure"""
        A = sparse.lil_matrix((n, n))
        
        for i in range(n):
            # Add entries in the band around diagonal
            start = max(0, i - bandwidth)
            end = min(n, i + bandwidth + 1)
            
            for j in range(start, end):
                if i == j:
                    A[i, j] = 2 * bandwidth + 2  # Diagonal dominance
                else:
                    A[i, j] = -1.0 / (abs(i-j) + 1)  # Weaker off-diagonal
        
        return A.tocsr()
    
    A = create_wide_band(600, 30)
    matrices.append(("wide_band_600x600.mtx", A))
    
    # Write all matrices in Matrix Market format
    from scipy.io import mmwrite
    
    for filename, matrix in matrices:
        filepath = os.path.join(matrix_dir, filename)
        mmwrite(filepath, matrix)
        print(f"Generated: {filename} - Shape: {matrix.shape}, NNZ: {matrix.nnz}")
    
    print(f"\nGenerated {len(matrices)} diverse test matrices in {matrix_dir}")
    print("These matrices cover various types:")
    print("- Symmetric Positive Definite (small, medium)")
    print("- Non-symmetric (small, medium)")
    print("- Indefinite")
    print("- Nearly singular")
    print("- Diagonally dominant")
    print("- Grid/stencil (Poisson-like)")
    print("- Rectangular-derived")
    print("- Convection-diffusion type")
    print("- Semiconductor type")
    print("- Wide band")

if __name__ == "__main__":
    generate_test_matrices()