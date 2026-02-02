#!/usr/bin/env python3
"""
Download matrices from Matrix Market for sparse solver testing
"""

import os
import requests
from urllib.parse import urljoin
import time
import sys

def download_matrix(url, dest_dir, filename):
    """Download a single matrix file"""
    filepath = os.path.join(dest_dir, filename)
    
    if os.path.exists(filepath):
        print(f"File {filename} already exists, skipping...")
        return True
        
    try:
        print(f"Downloading {filename} from {url}...")
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        with open(filepath, 'wb') as f:
            f.write(response.content)
        
        print(f"Successfully downloaded {filename}")
        return True
    except Exception as e:
        print(f"Failed to download {filename}: {str(e)}")
        return False

def main():
    # Create directory for test matrices
    matrix_dir = "test_matrices"
    os.makedirs(matrix_dir, exist_ok=True)
    
    # Matrix Market collection URLs (UCIrvine repository)
    # Using the sparse matrix collections that are commonly used for testing
    matrices_to_download = [
        # Symmetric positive definite matrices
        {
            'url': 'https://math.nist.gov/pub/MatrixMarket2/Harwell-Boeing/psadmit/s1rmq4m1.mtx.gz',
            'filename': 's1rmq4m1_spd.mtx.gz'
        },
        {
            'url': 'https://math.nist.gov/pub/MatrixMarket2/Harwell-Boeing/psadmit/sherman5.mtx.gz',
            'filename': 'sherman5_spd.mtx.gz'
        },
        {
            'url': 'https://math.nist.gov/pub/MatrixMarket2/Harwell-Boeing/psadmit/fs_183_1.mtx.gz',
            'filename': 'fs_183_1_spd.mtx.gz'
        },
        
        # Non-symmetric matrices
        {
            'url': 'https://math.nist.gov/pub/MatrixMarket2/Harwell-Boeing/counterx/impcol_a.mtx.gz',
            'filename': 'impcol_a_nonsym.mtx.gz'
        },
        {
            'url': 'https://math.nist.gov/pub/MatrixMarket2/Harwell-Boeing/misc/onetone1.mtx.gz',
            'filename': 'onetone1_nonsym.mtx.gz'
        },
        {
            'url': 'https://math.nist.gov/pub/MatrixMarket2/Harwell-Boeing/linear/can_24.mtx.gz',
            'filename': 'can_24_nonsym.mtx.gz'
        },
        
        # Different sizes - Small
        {
            'url': 'https://math.nist.gov/pub/MatrixMarket2/Harwell-Boeing/counterx/ash219.mtx.gz',
            'filename': 'ash219_small.mtx.gz'
        },
        {
            'url': 'https://math.nist.gov/pub/MatrixMarket2/Harwell-Boeing/linear/dwt_59.mtx.gz',
            'filename': 'dwt_59_small.mtx.gz'
        },
        
        # Different sizes - Medium
        {
            'url': 'https://math.nist.gov/pub/MatrixMarket2/Harwell-Boeing/linear/fs_183_6.mtx.gz',
            'filename': 'fs_183_6_medium.mtx.gz'
        },
        {
            'url': 'https://math.nist.gov/pub/MatrixMarket2/Harwell-Boeing/linear/lund_a.mtx.gz',
            'filename': 'lund_a_medium.mtx.gz'
        },
        
        # Different sizes - Large
        {
            'url': 'https://math.nist.gov/pub/MatrixMarket2/Harwell-Boeing/linear/lns_511.mtx.gz',
            'filename': 'lns_511_large.mtx.gz'
        },
        {
            'url': 'https://math.nist.gov/pub/MatrixMarket2/Harwell-Boeing/linear/hood.mtx.gz',
            'filename': 'hood_large.mtx.gz'
        },
        
        # Matrices with different characteristics
        {
            'url': 'https://math.nist.gov/pub/MatrixMarket2/Harwell-Boeing/linear/bcsstk01.mtx.gz',
            'filename': 'bcsstk01_stiffness.mtx.gz'
        },
        {
            'url': 'https://math.nist.gov/pub/MatrixMarket2/Harwell-Boeing/linear/finan512.mtx.gz',
            'filename': 'finan512_finance.mtx.gz'
        },
        {
            'url': 'https://math.nist.gov/pub/MatrixMarket2/Harwell-Boeing/linear/raefsky4.mtx.gz',
            'filename': 'raefsky4_fluid.mtx.gz'
        }
    ]
    
    print("Starting Matrix Market downloads...")
    print(f"Matrices will be saved to: {matrix_dir}")
    print("="*60)
    
    success_count = 0
    total_count = len(matrices_to_download)
    
    for matrix_info in matrices_to_download:
        success = download_matrix(
            matrix_info['url'],
            matrix_dir,
            matrix_info['filename']
        )
        
        if success:
            success_count += 1
            # Be respectful to the server
            time.sleep(1)
    
    print("="*60)
    print(f"Download completed: {success_count}/{total_count} matrices downloaded successfully")
    
    # Also download some matrices from University of Florida Sparse Matrix Collection
    # These are accessed via a different approach since direct URLs might change
    print("\nAttempting to download from SuiteSparse collection...")
    
    # Alternative smaller matrices that are commonly used for testing
    alternative_matrices = [
        # Using Matrix Market direct format files
        {
            'url': 'https://math.nist.gov/MatrixMarket/mmio/matlab/test_mat.mtx',
            'filename': 'test_small.mtx'
        }
    ]
    
    for matrix_info in alternative_matrices:
        success = download_matrix(
            matrix_info['url'],
            matrix_dir,
            matrix_info['filename']
        )
        if success:
            success_count += 1
            time.sleep(1)
    
    print(f"Total matrices in test directory: {len([f for f in os.listdir(matrix_dir) if f.endswith('.mtx') or f.endswith('.mtx.gz')])}")
    print("Matrix Market download script completed.")

if __name__ == "__main__":
    main()