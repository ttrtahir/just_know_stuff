# Hankel Tensorization

**One-line summary**: Reshape a Hankel-structured sequence into a higher-order tensor so that low-rank tensor decompositions can exploit the underlying redundancy.

---

## ELI5

Imagine you have a long list of numbers that were recorded one after another (like temperature readings every hour). A **Hankel matrix** is a clever way to fold that list into a table so that every slanted diagonal contains the same number — this reveals hidden repeating patterns.

**Tensorization** takes that idea one step further: instead of folding the list into a flat table (matrix), you fold it into a *cube* (or even higher-dimensional box). This higher-dimensional structure lets powerful math tools — called *tensor decompositions* — find patterns that a flat matrix would miss.

Doing this in a **hierarchical manner** means you repeat the folding process level by level: first fold the raw sequence, then fold the result again, and so on — like origami that you keep folding until the paper reveals its hidden geometry.

---

## Prior Knowledge Required

- Basic linear algebra (vectors, matrices, rank)
- What a Hankel matrix is
- Idea of a tensor (multi-dimensional array)
- Singular Value Decomposition (SVD)

---

## Subtopics

| Subtopic | Folder | Status |
|----------|--------|--------|
| Hankel matrices & structured matrices | `1_Fundamentals/` | TODO |
| Tensor basics & notation | `1_Fundamentals/` | TODO |
| Hankel-to-tensor embedding | `2_Hankel_Tensorization/` | TODO |
| Hankel tensor properties & rank | `2_Hankel_Tensorization/` | TODO |
| Low-rank Hankel tensor decomposition | `2_Hankel_Tensorization/` | TODO |
| Hierarchical Tucker (HT) decomposition | `3_Hierarchical_Methods/` | TODO |
| Multi-level Hankel tensorization | `3_Hierarchical_Methods/` | TODO |
| Applications (signal processing, ML) | `3_Hierarchical_Methods/` | TODO |

---

## Key Concepts

**Hankel matrix** — An `(m × n)` matrix `H` where `H[i,j] = h[i+j]` for a sequence `h`. Every anti-diagonal holds a constant value.

**Tensorization** — The act of reshaping a vector or matrix into a tensor of higher order while preserving (or leveraging) its algebraic structure.

**Hankel tensor** — A tensor of order `d` built from a sequence `h` such that entry `T[i₁, i₂, ..., i_d] = h[i₁ + i₂ + ... + i_d]`.

**Hierarchical tensorization** — A multi-level process where the Hankel embedding is applied recursively, producing a tree-structured tensor representation suitable for hierarchical decompositions (e.g., Hierarchical Tucker).

**Tensor rank** — The minimum number of rank-1 tensors whose sum equals the given tensor; central to compression and learning.

---

## Comparison / Differences

| Representation | Structure exploited | Decomposition | Typical use |
|---------------|---------------------|---------------|-------------|
| Hankel matrix | 2-way anti-diagonal | SVD / HOSVD | 1-D signals |
| Hankel tensor (order d) | d-way anti-diagonal | CP / Tucker | Multi-channel signals |
| Hierarchical Hankel tensor | Multi-level anti-diagonal | Hierarchical Tucker | Long sequences, ML compression |

---

## Basic Implementation

```python
# TODO: build a Hankel matrix from a 1-D sequence
import numpy as np

def hankel_matrix(seq, n_rows):
    """Embed sequence into a Hankel matrix."""
    # TODO
    pass
```

---

## Sample Data

```
# TODO: 1-D signal example
seq = [1, 2, 3, 4, 5, 6, 7, 8]
# Expected Hankel matrix (4×5):
# [[1, 2, 3, 4, 5],
#  [2, 3, 4, 5, 6],
#  [3, 4, 5, 6, 7],
#  [4, 5, 6, 7, 8]]
```

---

## Proper Implementation

```python
# TODO: full Hankel tensorization pipeline
# 1. Build Hankel tensor of chosen order
# 2. Apply Tucker / hierarchical Tucker decomposition
# 3. Reconstruct and evaluate approximation error
```

---

## Resources

- https://en.wikipedia.org/wiki/Hankel_matrix
- https://arxiv.org/abs/1307.0401 — Kolda & Bader, "Tensor Decompositions and Applications"
- https://arxiv.org/abs/1004.3752 — Hackbusch & Kühn, "A new scheme for the tensor representation"
- https://www.sciencedirect.com/science/article/pii/S0024379515004668 — Hankel tensors: structure and computability
