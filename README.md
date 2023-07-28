# rSVD

A Rust library for computing an approximate singular value decomposition (SVD) of a matrix using randomized algorithms.

## Background

The singular value decomposition (SVD) is a matrix factorization technique that is useful for many applications in data analysis and scientific computing. However, computing the full SVD can be very computationally intensive for large matrices.

Randomized algorithms provide an efficient way to compute an approximate SVD that captures most of the action of the original matrix in significantly less time. The key ideas behind randomized SVD are:

1. Use a random test matrix to sample the range of the input matrix
2. Orthogonalize the samples to obtain a subspace that captures the range
3. Compute the SVD of the smaller projected matrix

This technique allows the SVD to be approximated in O(mn log k) time instead of the O(mn min(m,n)) operations required for the full SVD, where m and n are the dimensions of the input matrix and k is the target rank.

## Features

- Compute approximate rank-k SVD of a matrix in O(mn log k) time
- Flexible choice of target rank k and oversampling parameter p
- Uses BLAS/LAPACK via [ndarray-linalg](https://docs.rs/ndarray-linalg) for performance
- Pure Rust implementation

## Usage

The main function is `rsvd()`, which takes a matrix, target rank k, and oversampling parameter p. It returns the approximate SVD factors U, S, and V^T.

```rust
use ndarray::{Array2, array};
use rsvd::rsvd;

let a = array!([1., 2., 3.],
               [8., 9., 4.],
               [7., 6., 5.]);

let (u, s, vt) = rsvd(&a, 2, 0, None);
```

## Applications

Randomized SVD is useful for many applications that involve large data matrices:

- Dimensionality reduction and feature extraction
- Principal component analysis (PCA)
- Recommender systems (collaborative filtering)
- Image compression
- Text analysis (LSA)

By efficiently approximating the SVD to capture the top k singular values/vectors, randomized SVD allows these techniques to scale to massive datasets.

The SVD also provides a powerful tool for analyzing and understanding the properties of a dataset contained in the matrix. The singular values indicate the importance of each component, while the singular vectors identify the principal components.

## References

- [Nathan Halko, et al. Finding structure with randomness: Probabilistic algorithms for constructing approximate matrix decompositions. (2011)](https://arxiv.org/abs/0909.4061)
