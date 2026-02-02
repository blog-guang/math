#pragma once

#include "sparse_matrix.h"
#include "vector.h"
#include <vector>

namespace math {

/**
 * Matrix reordering algorithms for improving iterative solver convergence.
 *
 * Typical pipeline for non-symmetric matrices with zero diagonals (e.g. LNS series):
 *   1. nonzeroDiagonalPerm()  → row permutation to eliminate zeros on diagonal
 *   2. rcm()                  → RCM reordering for bandwidth reduction (less ILU fill-in)
 *   3. Apply combined permutation, then ILU0 + GMRES
 *
 * Permutation convention:
 *   perm[i] = j  means "position i in the new ordering corresponds to index j in the original"
 *   Equivalently: newVec[i] = oldVec[perm[i]]   (applyPerm)
 *                 newMat[i][j] = oldMat[perm[i]][perm[j]]  (permuteSymmetric)
 */
class Reorder {
public:
    // ── Core reordering algorithms ──────────────────────────────────

    /**
     * Find a row permutation P such that diag(P*A) has no structural zeros.
     *
     * Uses maximum bipartite matching via DFS augmenting paths.
     * Time: O(n * nnz) worst case, typically much faster for sparse matrices.
     *
     * @return perm where perm[i] = original row assigned to diagonal position i.
     *         Empty vector if the matrix is structurally singular
     *         (no permutation can make all diagonal entries nonzero).
     */
    [[nodiscard]] static std::vector<size_t> nonzeroDiagonalPerm(const SparseMatrix& A);

    /**
     * Reverse Cuthill-McKee (RCM) reordering on the symmetrized sparsity pattern.
     *
     * Works on the graph of (A + A^T): treats the matrix as undirected.
     * Reduces bandwidth → less fill-in in subsequent ILU factorization.
     *
     * Algorithm:
     *   1. Pick a peripheral node (approximated via double-BFS) as start
     *   2. BFS, visiting neighbors sorted by ascending degree
     *   3. Reverse the resulting order (the "Reverse" in RCM)
     *
     * @return perm where perm[i] = original index at position i in the new ordering.
     */
    [[nodiscard]] static std::vector<size_t> rcm(const SparseMatrix& A);

    // ── Permutation application ─────────────────────────────────────

    /** Row permutation: result[i][j] = A[perm[i]][j] */
    [[nodiscard]] static SparseMatrix permuteRows(const SparseMatrix& A,
                                                  const std::vector<size_t>& perm);

    /** Symmetric permutation: result[i][j] = A[perm[i]][perm[j]] */
    [[nodiscard]] static SparseMatrix permuteSymmetric(const SparseMatrix& A,
                                                       const std::vector<size_t>& perm);

    /** Row+column permutation: result[i][j] = A[rowPerm[i]][colPerm[j]] */
    [[nodiscard]] static SparseMatrix permuteRowCol(const SparseMatrix& A,
                                                    const std::vector<size_t>& rowPerm,
                                                    const std::vector<size_t>& colPerm);

    // ── Vector / permutation utilities ──────────────────────────────

    /** Apply permutation to vector: result[i] = v[perm[i]] */
    [[nodiscard]] static Vector applyPerm(const std::vector<size_t>& perm, const Vector& v);

    /** Apply inverse permutation: result[perm[i]] = v[i] */
    [[nodiscard]] static Vector applyInvPerm(const std::vector<size_t>& perm, const Vector& v);

    /** Compute inverse permutation: inv[perm[i]] = i */
    [[nodiscard]] static std::vector<size_t> invertPerm(const std::vector<size_t>& perm);

    /** Compose permutations: result[i] = outer[inner[i]] */
    [[nodiscard]] static std::vector<size_t> composePerm(const std::vector<size_t>& outer,
                                                         const std::vector<size_t>& inner);

    // ── Diagnostics ─────────────────────────────────────────────────

    /** Count zero diagonal entries */
    static size_t countZeroDiag(const SparseMatrix& A);

    /** Compute bandwidth of a square matrix */
    static size_t bandwidth(const SparseMatrix& A);

private:
    // DFS augmenting path for bipartite matching
    static bool augment(size_t col,
                        const std::vector<std::vector<size_t>>& colToRows,
                        std::vector<int>& matchCol,
                        std::vector<int>& matchRow,
                        std::vector<bool>& visited);

    // BFS for RCM, returns ordering starting from 'start'
    static std::vector<size_t> bfsCM(size_t start,
                                     const std::vector<std::vector<size_t>>& adj,
                                     std::vector<bool>& visited);
};

}  // namespace math
