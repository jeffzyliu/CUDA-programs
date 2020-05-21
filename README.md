# CUDA-programs
A collection of GPU and high-performance computation programs written in CUDA.

Contains the following programs:
1. Parallelized matrix multiplication, transpose, and parallel reduction for Frobenius Norms (done)
2. Newtonian gravitational n-body solver using explicit Euler time integration (done)
3. Sparse linear system solver for discretized Laplacian convolution using parallelized Jacobi iteration and Red-Black Gauss-Seidel method (WIP)

For program 1, I received **2nd highest speed in class** for simple matrix multiplication AB, 8th place for transpose multiplication A^TBA, and 8th place for F-norm parallel reduction

For program 2, I was only 14th in class. However, with 1 line of code changed to leverage CUDA's math library, which was what most other students did, my placement would have jumped to 5th place. My code was highly optimized except for using traditional square root math instead of CUDA's optimized rsqrt() function.

Results for program 3 have yet to be finished.
