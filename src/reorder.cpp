#include "reorder.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <numeric>
#include <queue>
#include <unordered_set>

namespace math {

// ── Bipartite matching helpers ──────────────────────────────────────────────

/**
 * DFS augmenting path for maximum bipartite matching.
 * Tries to find an augmenting path starting from unmatched column `col`.
 *
 * matchCol[c] = row matched to column c (-1 if unmatched)
 * matchRow[r] = col matched to row r    (-1 if unmatched)
 */
bool Reorder::augment(size_t col,
                      const std::vector<std::vector<size_t>>& colToRows,
                      std::vector<int>& matchCol,
                      std::vector<int>& matchRow,
                      std::vector<bool>& visited) {
    for (size_t row : colToRows[col]) {
        if (visited[row]) continue;
        visited[row] = true;

        // If this row is unmatched, or we can find an alternative for its current match
        if (matchRow[row] < 0 ||
            augment(static_cast<size_t>(matchRow[row]), colToRows, matchCol, matchRow, visited)) {
            matchCol[col] = static_cast<int>(row);
            matchRow[row] = static_cast<int>(col);
            return true;
        }
    }
    return false;
}

// ── RCM helpers ─────────────────────────────────────────────────────────────

/**
 * Cuthill-McKee BFS from a given start node.
 * At each level, neighbors are visited in order of ascending degree.
 * Handles only the connected component containing `start`.
 */
std::vector<size_t> Reorder::bfsCM(size_t start,
                                   const std::vector<std::vector<size_t>>& adj,
                                   std::vector<bool>& visited) {
    std::vector<size_t> order;
    std::queue<size_t> q;

    q.push(start);
    visited[start] = true;

    while (!q.empty()) {
        size_t u = q.front();
        q.pop();
        order.push_back(u);

        // Collect unvisited neighbors and sort by degree (ascending)
        std::vector<size_t> neighbors;
        for (size_t v : adj[u]) {
            if (!visited[v]) {
                neighbors.push_back(v);
            }
        }
        std::sort(neighbors.begin(), neighbors.end(), [&](size_t a, size_t b) {
            return adj[a].size() < adj[b].size();
        });

        for (size_t v : neighbors) {
            if (!visited[v]) {
                visited[v] = true;
                q.push(v);
            }
        }
    }
    return order;
}

// ── Core algorithms ─────────────────────────────────────────────────────────

std::vector<size_t> Reorder::nonzeroDiagonalPerm(const SparseMatrix& A) {
    size_t n = A.rows();
    assert(A.rows() == A.cols());

    // Ensure we have COO data
    SparseMatrix Acopy = [&]() -> SparseMatrix {
        // We can't copy (deleted), so we re-read via COO accessors after toCOO
        // Actually we need a mutable reference. Let's just build colToRows from COO directly.
        return SparseMatrix(); // placeholder, won't be used
    }();
    (void)Acopy;

    // Build column → list of rows that have nonzero entries in that column
    // Use CSC format for efficient column access
    // But we can also just scan COO data
    std::vector<std::vector<size_t>> colToRows(n);

    // We need COO data. The matrix might be in CSR/CSC.
    // Use the const accessors - check what format data is available
    // Safest: use get() but that's O(nnz) per call.
    // Better: access internal arrays directly via the public accessors.

    // Try COO first, fall back to CSR
    if (!A.coo_row().empty()) {
        // COO data available
        for (size_t k = 0; k < A.coo_row().size(); ++k) {
            size_t r = A.coo_row()[k];
            size_t c = A.coo_col()[k];
            if (r < n && c < n && std::abs(A.coo_val()[k]) > 0.0) {
                colToRows[c].push_back(r);
            }
        }
    } else if (!A.csr_row_ptr().empty()) {
        // CSR data available - scan all entries
        for (size_t i = 0; i < n; ++i) {
            for (size_t k = A.csr_row_ptr()[i]; k < A.csr_row_ptr()[i + 1]; ++k) {
                size_t c = A.csr_col_idx()[k];
                if (c < n && std::abs(A.csr_val()[k]) > 0.0) {
                    colToRows[c].push_back(i);
                }
            }
        }
    } else {
        // No data available - return identity
        std::vector<size_t> perm(n);
        std::iota(perm.begin(), perm.end(), 0);
        return perm;
    }

    // Remove duplicate rows in each column's list
    for (auto& rows : colToRows) {
        std::sort(rows.begin(), rows.end());
        rows.erase(std::unique(rows.begin(), rows.end()), rows.end());
    }

    // Maximum bipartite matching: for each column, find a row to match
    // matchCol[col] = row matched to this column (-1 if unmatched)
    // matchRow[row] = col matched to this row    (-1 if unmatched)
    std::vector<int> matchCol(n, -1);
    std::vector<int> matchRow(n, -1);

    // Greedy pre-matching: assign diagonal entries first (fast path)
    for (size_t i = 0; i < n; ++i) {
        for (size_t row : colToRows[i]) {
            if (row == i && matchRow[row] < 0) {
                matchCol[i] = static_cast<int>(row);
                matchRow[row] = static_cast<int>(i);
                break;
            }
        }
    }

    // Augmenting path phase for unmatched columns
    for (size_t col = 0; col < n; ++col) {
        if (matchCol[col] >= 0) continue;  // Already matched
        std::vector<bool> visited(n, false);
        augment(col, colToRows, matchCol, matchRow, visited);
    }

    // Check if full matching was found
    std::vector<size_t> perm(n);
    bool fullMatch = true;
    for (size_t i = 0; i < n; ++i) {
        if (matchCol[i] < 0) {
            fullMatch = false;
            perm[i] = i;  // Fallback to identity for unmatched columns
        } else {
            perm[i] = static_cast<size_t>(matchCol[i]);
        }
    }

    if (!fullMatch) {
        return {};  // Structurally singular — no valid permutation exists
    }
    return perm;
}

std::vector<size_t> Reorder::rcm(const SparseMatrix& A) {
    size_t n = A.rows();
    assert(A.rows() == A.cols());

    // Build symmetric adjacency list from sparsity pattern of A + A^T
    std::vector<std::vector<size_t>> adj(n);

    auto addEdge = [&](size_t i, size_t j) {
        if (i != j && i < n && j < n) {
            adj[i].push_back(j);
            adj[j].push_back(i);
        }
    };

    if (!A.coo_row().empty()) {
        for (size_t k = 0; k < A.coo_row().size(); ++k) {
            addEdge(A.coo_row()[k], A.coo_col()[k]);
        }
    } else if (!A.csr_row_ptr().empty()) {
        for (size_t i = 0; i < n; ++i) {
            for (size_t k = A.csr_row_ptr()[i]; k < A.csr_row_ptr()[i + 1]; ++k) {
                addEdge(i, A.csr_col_idx()[k]);
            }
        }
    }

    // Deduplicate adjacency lists
    for (auto& neighbors : adj) {
        std::sort(neighbors.begin(), neighbors.end());
        neighbors.erase(std::unique(neighbors.begin(), neighbors.end()), neighbors.end());
    }

    std::vector<bool> visited(n, false);
    std::vector<size_t> order;
    order.reserve(n);

    // Process each connected component
    for (size_t startCandidate = 0; startCandidate < n; ++startCandidate) {
        if (visited[startCandidate]) continue;

        // Find a peripheral node via double-BFS heuristic:
        // 1. BFS from arbitrary node → find farthest node u
        // 2. BFS from u → find farthest node v
        // 3. Use v as start (it's approximately peripheral)
        size_t start = startCandidate;
        {
            // First BFS to find a far node
            std::vector<bool> tmpVisited(n, false);
            auto tmp = bfsCM(startCandidate, adj, tmpVisited);
            if (!tmp.empty()) start = tmp.back();  // Farthest node in BFS order

            // Second BFS from that node
            std::fill(tmpVisited.begin(), tmpVisited.end(), false);
            tmp = bfsCM(start, adj, tmpVisited);
            if (!tmp.empty()) start = tmp.back();  // Even farther — good peripheral node
        }

        // Now do the actual Cuthill-McKee BFS from the peripheral node
        auto cmOrder = bfsCM(start, adj, visited);

        // Reverse (the "Reverse" in RCM) and append
        std::reverse(cmOrder.begin(), cmOrder.end());
        for (size_t idx : cmOrder) {
            order.push_back(idx);
        }
    }

    assert(order.size() == n);
    return order;
}

// ── Permutation application ─────────────────────────────────────────────────

SparseMatrix Reorder::permuteRows(const SparseMatrix& A, const std::vector<size_t>& perm) {
    size_t n = A.rows();
    size_t m = A.cols();
    assert(perm.size() == n);

    // Build inverse permutation: invPerm[perm[i]] = i
    // We need: result[i][j] = A[perm[i]][j]
    // In COO: for each entry (r, c, v) in A, it appears as (invPerm[r], c, v) in result
    std::vector<size_t> invPerm(n);
    for (size_t i = 0; i < n; ++i) invPerm[perm[i]] = i;

    std::vector<size_t> newRows, newCols;
    std::vector<double> newVals;

    if (!A.coo_row().empty()) {
        size_t nnz = A.coo_row().size();
        newRows.resize(nnz);
        newCols = A.coo_col();
        newVals = A.coo_val();
        for (size_t k = 0; k < nnz; ++k) {
            newRows[k] = invPerm[A.coo_row()[k]];
        }
    } else if (!A.csr_row_ptr().empty()) {
        for (size_t i = 0; i < n; ++i) {
            for (size_t k = A.csr_row_ptr()[i]; k < A.csr_row_ptr()[i + 1]; ++k) {
                newRows.push_back(invPerm[i]);
                newCols.push_back(A.csr_col_idx()[k]);
                newVals.push_back(A.csr_val()[k]);
            }
        }
    }

    return SparseMatrix::fromCOO(n, m, newRows, newCols, newVals);
}

SparseMatrix Reorder::permuteSymmetric(const SparseMatrix& A, const std::vector<size_t>& perm) {
    size_t n = A.rows();
    assert(A.rows() == A.cols());
    assert(perm.size() == n);

    // result[i][j] = A[perm[i]][perm[j]]
    // In COO: entry (r, c, v) in A → (invPerm[r], invPerm[c], v) in result
    std::vector<size_t> invPerm(n);
    for (size_t i = 0; i < n; ++i) invPerm[perm[i]] = i;

    std::vector<size_t> newRows, newCols;
    std::vector<double> newVals;

    if (!A.coo_row().empty()) {
        size_t nnz = A.coo_row().size();
        newRows.resize(nnz);
        newCols.resize(nnz);
        newVals = A.coo_val();
        for (size_t k = 0; k < nnz; ++k) {
            newRows[k] = invPerm[A.coo_row()[k]];
            newCols[k] = invPerm[A.coo_col()[k]];
        }
    } else if (!A.csr_row_ptr().empty()) {
        for (size_t i = 0; i < n; ++i) {
            for (size_t k = A.csr_row_ptr()[i]; k < A.csr_row_ptr()[i + 1]; ++k) {
                newRows.push_back(invPerm[i]);
                newCols.push_back(invPerm[A.csr_col_idx()[k]]);
                newVals.push_back(A.csr_val()[k]);
            }
        }
    }

    return SparseMatrix::fromCOO(n, n, newRows, newCols, newVals);
}

SparseMatrix Reorder::permuteRowCol(const SparseMatrix& A,
                                    const std::vector<size_t>& rowPerm,
                                    const std::vector<size_t>& colPerm) {
    size_t n = A.rows();
    size_t m = A.cols();
    assert(rowPerm.size() == n);
    assert(colPerm.size() == m);

    // result[i][j] = A[rowPerm[i]][colPerm[j]]
    // Entry (r, c, v) → (invRowPerm[r], invColPerm[c], v)
    std::vector<size_t> invRowPerm(n), invColPerm(m);
    for (size_t i = 0; i < n; ++i) invRowPerm[rowPerm[i]] = i;
    for (size_t j = 0; j < m; ++j) invColPerm[colPerm[j]] = j;

    std::vector<size_t> newRows, newCols;
    std::vector<double> newVals;

    if (!A.coo_row().empty()) {
        size_t nnz = A.coo_row().size();
        newRows.resize(nnz);
        newCols.resize(nnz);
        newVals = A.coo_val();
        for (size_t k = 0; k < nnz; ++k) {
            newRows[k] = invRowPerm[A.coo_row()[k]];
            newCols[k] = invColPerm[A.coo_col()[k]];
        }
    } else if (!A.csr_row_ptr().empty()) {
        for (size_t i = 0; i < n; ++i) {
            for (size_t k = A.csr_row_ptr()[i]; k < A.csr_row_ptr()[i + 1]; ++k) {
                newRows.push_back(invRowPerm[i]);
                newCols.push_back(invColPerm[A.csr_col_idx()[k]]);
                newVals.push_back(A.csr_val()[k]);
            }
        }
    }

    return SparseMatrix::fromCOO(n, m, newRows, newCols, newVals);
}

// ── Vector / permutation utilities ──────────────────────────────────────────

Vector Reorder::applyPerm(const std::vector<size_t>& perm, const Vector& v) {
    size_t n = perm.size();
    Vector result(n);
    for (size_t i = 0; i < n; ++i) {
        result[i] = v[perm[i]];
    }
    return result;
}

Vector Reorder::applyInvPerm(const std::vector<size_t>& perm, const Vector& v) {
    size_t n = perm.size();
    Vector result(n);
    for (size_t i = 0; i < n; ++i) {
        result[perm[i]] = v[i];
    }
    return result;
}

std::vector<size_t> Reorder::invertPerm(const std::vector<size_t>& perm) {
    size_t n = perm.size();
    std::vector<size_t> inv(n);
    for (size_t i = 0; i < n; ++i) inv[perm[i]] = i;
    return inv;
}

std::vector<size_t> Reorder::composePerm(const std::vector<size_t>& outer,
                                         const std::vector<size_t>& inner) {
    size_t n = inner.size();
    std::vector<size_t> result(n);
    for (size_t i = 0; i < n; ++i) {
        result[i] = outer[inner[i]];
    }
    return result;
}

// ── Diagnostics ─────────────────────────────────────────────────────────────

size_t Reorder::countZeroDiag(const SparseMatrix& A) {
    size_t n = A.rows();
    size_t count = 0;
    for (size_t i = 0; i < n; ++i) {
        if (std::abs(A.get(i, i)) < 1e-30) ++count;
    }
    return count;
}

size_t Reorder::bandwidth(const SparseMatrix& A) {
    size_t bw = 0;

    if (!A.coo_row().empty()) {
        for (size_t k = 0; k < A.coo_row().size(); ++k) {
            size_t r = A.coo_row()[k];
            size_t c = A.coo_col()[k];
            size_t diff = (r > c) ? (r - c) : (c - r);
            bw = std::max(bw, diff);
        }
    } else if (!A.csr_row_ptr().empty()) {
        size_t n = A.rows();
        for (size_t i = 0; i < n; ++i) {
            for (size_t k = A.csr_row_ptr()[i]; k < A.csr_row_ptr()[i + 1]; ++k) {
                size_t c = A.csr_col_idx()[k];
                size_t diff = (i > c) ? (i - c) : (c - i);
                bw = std::max(bw, diff);
            }
        }
    }

    return bw;
}

}  // namespace math
