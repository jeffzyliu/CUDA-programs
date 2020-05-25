# CUDA-programs
A collection of GPU and high-performance computation programs written in CUDA.

Contains the following programs:
1. Parallelized matrix multiplication, transpose, and parallel reduction for Frobenius Norms (done)
2. Newtonian gravitational n-body solver using explicit Euler time integration (done)
3. Sparse linear system solver for discretized Laplacian convolution using parallelized Jacobi iteration and Red-Black Gauss-Seidel method (WIP)

For program 1, I received **2nd highest speed in class** for simple matrix multiplication AB, 8th place for transpose multiplication A^TBA, and 8th place for F-norm parallel reduction

For program 2, I was only 14th in class. However, with 1 line of code changed to leverage CUDA's math library, which was what most other students did, my placement would have jumped to 5th place. My code was highly optimized except for using traditional square root math instead of CUDA's optimized rsqrt() function.

Results for program 3 have yet to be finished.

## More detailed intro:

### What is CUDA?

CUDA stands for Compute Unified Device Architecture, which is an ecosystem of hardware, firmware, and software APIs for programmers to do general-purpose GPU programming on an Nvidia GPU. I use CUDA here as CUDA C/C++, which is an extension to C/C++ that uses CUDA to write kernels, or GPU functions, alongside traditional CPU functions.

### Why use GPUs?

For many years, Moore's law has held true. We have seen exponential growth in CPU processing power, and exponential decay in cost for that processing power since their inception. However, Moore's law is dying. We're running into the laws of physics that prevent us from fitting more transistors on a chip without overheating. The current solution, then, to increase processing power (before new computing paradigms such as quantum computing can bridge tha gap), is to not use better cores, but to use more cores. This is the motivation behind parallel computing, and how modern supercomputers work today.

An extension of the parallel computing model lets us arrive at the concept of General-Purpose GPU programming. GPUs were initially (and still are) created for graphics processing, which is an inherently parallel computing problem. Thousands of pixel colors are calculated dozens of times a second to be fed to a screen. Instead of using a small number of CPU cores, computer engineers created a GPU containing **hundreds of thousands of much weaker cores to do many tasks in parallel in one computer chip**.

With the progression of GPU development and general-purpose GPU programming, we now have the processing power of supercomputers contained in something as small as a gaming laptop, capable of running **programs hundreds or sometimes thousands of times faster than an equivalently powerful CPU**. That sounds amazing, but obviously comes with some important catches.

### How is GPU programming different, and what are the constraints?

The laws of physics don't just decide to give up on us when we use a GPU. So to use this massively parallel computing paradigm, there are necessary concessions we need to make.

The topic gets really technical here, but in summary, in exchange for raw strength in floating-point and integer arithmetic, we have to trade away the sophisticated schedulers and memory caches that a CPU has. On a CPU, hyper-sophisticated hardware predicts things like if statements, going down both branches faster than it can actaully evluate which one is right, and hopping over to the correct one once it finishes evaluating the conditional. Memory accesses have multiple hierarchal layers, optimized to the max so that your RAM accesses are automatically as perfect as they can be. All of this is a product of decades of research, and makes our lives as CPU programmers very easy. But it comes at the cost of silicon and transistor space that is relegated to pure arithmetic horsepower on a GPU.

These extreme limitations balance our extreme gains in speed. That's why GPUs don't do everything--they simply cannot do the advanced serial tasks that our CPUs have evolved to master. This motivates new programming paradigms called "SIMD" and "MIMD": Single/multiple instruction, multiple data.

### SIMD

Because we lack schedulers on a GPU, many cores must operate in unison. That is, they must be issued the exact same machine code. And then to make them useful in parallel, we need them to operate on different data. **The challenge, then, is to somehow create a unified set of instructions that every single of the thousands of cores on a GPU can follow, but to have them retain a level of sophistication required for advanced computing**.

The other motivator for SIMD is memory access. Because we lack the CPU cache systems that make everything perfect, access to memory must be pefectly coordinated between GPU cores to maximize efficiency. SIMD also accomplishes this, if the programmer has the skill to align and coalese their memory fetches.

### GPU architecture

In addition to relying on SIMD, GPU architecture also has a few more catches. In computer engineering, there is something called the principle of locality, which I discussed in brief with memory access above. A corollary of the principle of locality is that the closer a memory source is to the arithmetic processor, the faster it can be. However, this comes at a cost of capacity. The closer a memory source is to the processor, the less capacity it can have as well. This creates a hierarchy of register, local, shared, and global memory, along with the CPU DRAM, that further complicates the navigation of SIMD.

**Together, navigating the SIMD paradigm and the GPU architecture are the heart and soul of GPU programming.**

### Vectorized programming and scientific computing

The SIMD model (and in more advanced contexts, MIMD can be created) is then perfect on vectorized operations, where a computation can be described as a set of parallel operations on vectors (arrays to us computer scientists but with math terminology). Coincidentally, linear algebra and differential equations are the fundamental processes of almost all things in scientific computing. So SIMD and GPGPU programming is perfectly suited to massive scientific computing applications where a CPU would take far too long to feasibly compute a result.

### My projects and their applications

#### Competition 1 - Linear Algebra

My first assignment here was just to implement some basic algorithms that are fundamental to linear algebra, and also for understanding the SIMD model and GPU architecture. The key thing about matrix multiplication is that it is perfectly parallelizable; given matrices A and B to multiply, all elements in result matrix C are completely independent from each other. This is great, since we can task every element in our C to a single GPU core, which will compute it with a dot product, a same instruction which satisfies SIMD.

Matrix transposing is a really easy problem, but it is a little bit tricky to do quickly without wasting time. I won't go into the details here, but it involves swapping the elements in shared memory and doing consistent reads and writes in row-major order.

Parallel reduction for a matrix norm is also important in linear algebra (for instance, calculating the error for a least-squares approximation). Parallel reduction techniques are also far too complex for me to describe in a README, but it also relies on using shared memory and detailed scheduling of the cores to not waste any GPU time.

I was able to get a massive speedup for competition 1, and I almost topped the class leaderboards.

#### Competition 2 - N-body problem

My second assignment was to iteratively solve a gravitational n-body problem. In this type of problem, a large number of physical objects are related by certain differential equations in physics. These can be a large variety of equations; perhaps they're electromagnetic forces, physical stress, momentum and torques, or in my case, Newtonian gravity.

Gravitational n-body problems use the gravitational differential equation relating mass and distance between two objects to the force of attraction. By F=mA, I can solve for acceleration, and then integrate twice to find position. Computers can't really do calculus, but they can do extremely accurate approximations through discrete integration methods. In this case, discrete Euler time integration means using Euler's method of adding up a ton of tangent lines at infinitesimal intervals to closely approximate an integral.

This problem is again fully parallelizable. Each body's net acceleration is the sum of all the forces acting on it by the other bodies, divided by its mass. Notice that this is an O(n^2) problem, since each time step, we need to calculate force for each particle, which relies on each of the other ones. There are ways to approximate this and reduce it to O(nlogn) by using octrees, but our task was to parallelize and speed up the simplest brute force method.

By parallelizing this, we can in theory reduce it to O(n) as long as we have O(n) cores, and the ability to arbitrarily summon many more cores for us. This actually does happen for smaller test cases. For example, the Tesla K80 accelerators on the Dartmouth supercomputing cluster (those are like 5 thousand bucks a piece, jeez) have twin GPUs of 2496 CUDA cores each. If we're using only one out of two GPUs, we can run up to 2496 particles in complete parallel, and simulating in O(2496) time. If we have significantly more particles, we still see a huge speedup (up to thousands of times faster) than the CPU.

I described my process, my successes, and my struggles with writing this code in a github gist [here](https://gist.github.com/jeffzyliu/6c37a9bc1a8aeab6e8985c1276f1796a#file-jeff-dali-somethingibuilt-md).

Ultimately, I optimized this to near perfection, but I did not use two techniques that other people did in the class. One of them was a memory organization thing that was not a huge deal. The other one was. There exists a function that did math like 10 times faster than just writing out the operations by hand which I did not use, and my placement suffered as a result. That's a lesson that small optimizations mean little compared to the central bottleneck of your program, which in this case was my reciprocal square root calculation, a calculation that's very computationally expensive without using CUDA libraries.

#### Competition 3 -- Poisson equation linear solver

The Poisson equation is a common scientific computing problem with many applications, especially in fluid dynamics. It is `Laplacian(phi) = f`, where the Laplace operator is the divergence of the gradient of a multivariable function, `f` is some given result, and `phi` is the variable that we want to solve for. Once again, computers can't really do real calculus, so we approximate the Laplace operator by discretizing it on a grid of many particles. In that case, the divergence of the gradient can be approximated as the sum of the discretized partial derivatives `del/delx` and `del/dely` (and more if you're doing 3d or higher-dimensional problems). Those discretized partial derivatives can be calculated as a convolution over the grid.

This convolution can be expressed as a sparse matrix multiplication, where most of the elements are 0 except for the select few that we care about in the convolution for each particle on the grid. We can solve this with many methods, except the problem with the Poisson equation is that we easily get to billions of entries in such a matrix, or far more. Regular computation paradigms 
