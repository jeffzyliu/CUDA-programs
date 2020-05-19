//////////////////////////////////////////////////////////////////////////
////This is the code implementation for GPU Premier League Round 3: sparse linear solver
//////////////////////////////////////////////////////////////////////////
#include <iostream>
#include <fstream>

#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/reduce.h>
// #include <thrust/fill.h>
// #include <thrust/sequence.h>
// #include <thrust/transform.h>
// #include <thrust/replace.h>
// #include <thrust/functional.h>
// #include <thrust/sort.h>
// #include <thrust/extrema.h>
// #include <thrust/inner_product.h>


//////////////////////////////////////////////////////////////////////////
////TODO 0: Please replace the following strings with your team name and author names
////Note: Please do not use space in the string, use "_" instead
//////////////////////////////////////////////////////////////////////////

namespace name
{
	std::string team="using_namespace_std;";
	std::string author_1="Jeff Liu";
};

//////////////////////////////////////////////////////////////////////////
////TODO: Read the following three CPU implementations for Jacobi, Gauss-Seidel, and Red-Black Gauss-Seidel carefully
////and understand the steps for these numerical algorithms
//////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////
////These are the global variables that define the domain of the problem to solver
////You will need to use these parameters or macros in your GPU implementations
//////////////////////////////////////////////////////////////////////////

const int n=1024;							////grid size, we will change this value to up to 256 to test your code
const int g=1;							////padding size
const int s=(n+2*g)*(n+2*g);			////array size
#define I(i,j) (i+g)*(n+2*g)+(j+g)		////2D coordinate -> array index
#define B(i,j) i<0||i>=n||j<0||j>=n		////check boundary
const bool verbose=false;				////set false to turn off print for x and residual
const double tolerance=1e-3;			////tolerance for the iterative solver

//////////////////////////////////////////////////////////////////////////
////The following are three sample implementations for CPU iterative solvers
void Jacobi_Solver(double* x,const double* b)
{
	double* buf=new double[s];
	memcpy(buf,x,sizeof(double)*s);
	double* xr=x;			////read buffer pointer
	double* xw=buf;			////write buffer pointer
	int iter_num=0;			////iteration number
	int max_num=1e5;		////max iteration number
	// int max_num = 20;
	double residual=10.0;	////residual

	do{
		////update x values using the Jacobi iterative scheme
		for(int i=0;i<n;i++){
			for(int j=0;j<n;j++){
				xw[I(i,j)]=(b[I(i,j)]+xr[I(i-1,j)]+xr[I(i+1,j)]+xr[I(i,j-1)]+xr[I(i,j+1)])/4.0;
				// std::cout << xw[I(i,j)] << ", \t";
				// printf("%.0lf, \t", xw[I(i,j)]);
			}
			// std::cout <<std::endl;
		}

		////calculate residual
		residual=0.0;
		for(int i=0;i<n;i++){
			for(int j=0;j<n;j++){
				residual+=pow(4.0*xw[I(i,j)]-xw[I(i-1,j)]-xw[I(i+1,j)]-xw[I(i,j-1)]-xw[I(i,j+1)]-b[I(i,j)],2);
			}
		}

		if(verbose)std::cout<<"res: "<<residual<<std::endl;

		////swap the buffers
		double* swap=xr;
		xr=xw;
		xw=swap;
		iter_num++;
	}while(residual>tolerance&&iter_num<max_num);

	x=xw;

	if(verbose){
		// std::cout<<"\n\nx for Jacobi:\n";
		for(int i=-1;i<=n;i++){
			for(int j=-1;j<=n;j++){
				printf("%.0lf, \t", xr[I(i,j)]);
			}
			std::cout<<std::endl;
		}	
	}

	
	// x=xr;
	// std::cout<<"Jacobi solver converges in "<<iter_num<<" iterations, with residual "<<residual<<std::endl;

	delete [] buf;
}

void Gauss_Seidel_Solver(double* x,const double* b)
{
	int iter_num=0;			////iteration number
	int max_num=1e5;		////max iteration number
	double residual=0.0;	////residual

	do{
		////update x values using the Gauss-Seidel iterative scheme
		for(int i=0;i<n;i++){
			for(int j=0;j<n;j++){
				x[I(i,j)]=(b[I(i,j)]+x[I(i-1,j)]+x[I(i+1,j)]+x[I(i,j-1)]+x[I(i,j+1)])/4.0;
			}
		}

		////calculate residual
		residual=0.0;
		for(int i=0;i<n;i++){
			for(int j=0;j<n;j++){
				residual+=pow(4.0*x[I(i,j)]-x[I(i-1,j)]-x[I(i+1,j)]-x[I(i,j-1)]-x[I(i,j+1)]-b[I(i,j)],2);
			}
		}

		if(verbose)std::cout<<"res: "<<residual<<std::endl;
		iter_num++;
	}while(residual>tolerance&&iter_num<max_num);	

	std::cout<<"Gauss-Seidel solver converges in "<<iter_num<<" iterations, with residual "<<residual<<std::endl;
}

void Red_Black_Gauss_Seidel_Solver(double* x,const double* b)
{
	int iter_num=0;			////iteration number
	int max_num=1e5;		////max iteration number
	double residual=0.0;	////residual

	do{
		////red G-S
		for(int i=0;i<n;i++){
			for(int j=0;j<n;j++){
				if((i+j)%2==0)		////Look at this line!
					x[I(i,j)]=(b[I(i,j)]+x[I(i-1,j)]+x[I(i+1,j)]+x[I(i,j-1)]+x[I(i,j+1)])/4.0;
			}
		}

		////black G-S
		for(int i=0;i<n;i++){
			for(int j=0;j<n;j++){
				if((i+j)%2==1)		////And this line!
					x[I(i,j)]=(b[I(i,j)]+x[I(i-1,j)]+x[I(i+1,j)]+x[I(i,j-1)]+x[I(i,j+1)])/4.0;
			}
		}

		////calculate residual
		residual=0.0;
		for(int i=0;i<n;i++){
			for(int j=0;j<n;j++){
				residual+=pow(4.0*x[I(i,j)]-x[I(i-1,j)]-x[I(i+1,j)]-x[I(i,j-1)]-x[I(i,j+1)]-b[I(i,j)],2);
			}
		}

		if(verbose)std::cout<<"res: "<<residual<<std::endl;
		iter_num++;
	}while(residual>tolerance&&iter_num<max_num);	

	std::cout<<"Red-Black Gauss-Seidel solver converges in "<<iter_num<<" iterations, with residual "<<residual<<std::endl;
}

//////////////////////////////////////////////////////////////////////////
////In this function, we are solving a Poisson equation -laplace(p)=b, with p=x^2+y^2 and b=4.
////The boundary conditions are set on the one-ring ghost cells of the grid
//////////////////////////////////////////////////////////////////////////

void Test_CPU_Solvers()
{
	double* x=new double[s];
	memset(x,0x0000,sizeof(double)*s);
	double* b=new double[s];
	// for(int i=-1;i<=n;i++){
	// 	for(int j=-1;j<=n;j++){
	// 		b[I(i,j)]=4.0;		////set the values for the right-hand side
	// 	}
	// }

	// //////////////////////////////////////////////////////////////////////////
	// ////test Jacobi
	// for(int i=-1;i<=n;i++){
	// 	for(int j=-1;j<=n;j++){
	// 		if(B(i,j))
	// 			x[I(i,j)]=(double)(i*i+j*j);	////set boundary condition for x
	// 	}
	// }
	/////////////////////////////////////////////////////////////////////////
	////initialize x and b
	for(int i=-1;i<=n;i++){
		for(int j=-1;j<=n;j++){
			// b[I(i,j)]=4.0;		////set the values for the right-hand side
			b[I(i,j)]=100*i+j;
		}
	}
	for(int i=-1;i<=n;i++){
		for(int j=-1;j<=n;j++){
			if(B(i,j))
				x[I(i,j)]=(double)(i*i+j*j); ////set boundary condition for x
			else
				x[I(i,j)]=100*i+j;	
		}
	}

	Jacobi_Solver(x,b);

	// if(verbose){
		// std::cout<<"\n\nx for Jacobi:\n";
		// for(int i=0;i<n;i++){
		// 	for(int j=0;j<n;j++){
		// 		std::cout<<x[I(i,j)]<<", ";
		// 	}
		// 	std::cout<<std::endl;
		// }	
	// }
	std::cout<<"\n\n";

	// //////////////////////////////////////////////////////////////////////////
	// ////test Gauss-Seidel
	// memset(x,0x0000,sizeof(double)*s);
	// for(int i=-1;i<=n;i++){
	// 	for(int j=-1;j<=n;j++){
	// 		if(B(i,j))
	// 			x[I(i,j)]=(double)(i*i+j*j);	////set boundary condition for x
	// 	}
	// }

	// Gauss_Seidel_Solver(x,b);

	// if(verbose){
	// 	std::cout<<"\n\nx for Gauss-Seidel:\n";
	// 	for(int i=0;i<n;i++){
	// 		for(int j=0;j<n;j++){
	// 			std::cout<<x[I(i,j)]<<", ";
	// 		}
	// 		std::cout<<std::endl;
	// 	}	
	// }
	// std::cout<<"\n\n";

	// //////////////////////////////////////////////////////////////////////////
	// ////test Red-Black Gauss-Seidel
	// memset(x,0x0000,sizeof(double)*s);
	// for(int i=-1;i<=n;i++){
	// 	for(int j=-1;j<=n;j++){
	// 		if(B(i,j))
	// 			x[I(i,j)]=(double)(i*i+j*j);	////set boundary condition for x
	// 	}
	// }

	// Red_Black_Gauss_Seidel_Solver(x,b);

	// if(verbose){
	// 	std::cout<<"\n\nx for Red-Black Gauss-Seidel:\n";
	// 	for(int i=0;i<n;i++){
	// 		for(int j=0;j<n;j++){
	// 			std::cout<<x[I(i,j)]<<", ";
	// 		}
	// 		std::cout<<std::endl;
	// 	}	
	// }
	// std::cout<<"\n\n";

	// //////////////////////////////////////////////////////////////////////////

	delete [] x;
	delete [] b;
}

//////////////////////////////////////////////////////////////////////////
////TODO 1: your GPU variables and functions start here
// #define blockX 16
// #define blockY 18
__global__ void GPU_Jacobi(double* x, double* ghost, double* b,
							double* x_out, double* ghost_out)
{
	// shared memory prep, include ghost regions
	__shared__ double shared_x[18][18];
	// __shared__ double result_x[16][16];

	// registers prep
	double my_b = 0;
	double my_res = 0;
	int finalwarp_idx = 0;

	int absoluteY = blockIdx.y*blockDim.x+threadIdx.y; // not blockDimy to allow for the overlap
	int thr_per_row = blockDim.x*gridDim.x;
	int absoluteX = blockIdx.x*blockDim.x+threadIdx.x;

	double top = 0;
	double left = 0;
	double right = 0;
	double bottom = 0;

	int adjust_y = 0;
	int adjust_x = 0;

	// int thr_per_block = blockDim.x*blockDim.x; // not y to allow for the overlap
	// int block_idx = gridDim.x*blockIdx.y + blockIdx.x;
	// int thread_idx = blockDim.x*threadIdx.y + threadIdx.x;
	
	// PHASE ONE: load 18x16 middle columns with aligned, coalesced fetch
	// shared_x[threadIdx.y][threadIdx.x+1] = x[block_idx*thr_per_block + thread_idx];
	// shared_x[threadIdx.y][threadIdx.x+1] = x[(blockIdx.y*blockDim.y+threadIdx.y)*blockDim.x + blockIdx.x*blockDim.x+threadIdx.x];
	shared_x[threadIdx.y][threadIdx.x+1] = x[absoluteY*thr_per_row + absoluteX];
	__syncthreads();

	// PHASE TWO: half-warps 0-15 fetch global b
	// while half-warps 16-17 fetch the ghost columns from ghost
	if (threadIdx.y < 16) {
		// my_b = b[I(absoluteY, absoluteX)];
		my_b = b[absoluteY*thr_per_row + absoluteX];
	} else {
		finalwarp_idx = threadIdx.y - 16;
		shared_x[threadIdx.x+1][finalwarp_idx*17] = ghost[n*(blockIdx.x*2 + finalwarp_idx*3) + 
			threadIdx.x + blockDim.x*blockIdx.y];
	}
	__syncthreads();
	// if (threadIdx.x + threadIdx.y == 0 && blockIdx.x == 0 && blockIdx.y == 0) {
	// 	for (int i = 0; i < 18; i++) {
	// 		for (int j = 0; j < 18; j++) {
	// 			printf("%.0lf, \t",shared_x[i][j]);
	// 		}
	// 		printf("\n");
	// 	}
	// 	// printf("thr/blk: %d\n", thr_per_block);
	// 	// printf("blockidx: %d\n", block_idx);
	// 	// printf("thridx: %d\n", thread_idx);
	// 	// printf("final idx: %d\n", block_idx*thr_per_block + thread_idx);
	// 	// printf("res: %.0lf\n", x[block_idx*thr_per_block + thread_idx]);
	// }
	// if (blockIdx.x == 0 && blockIdx.y == 0) {
	// 	printf("%d  %d, b: %.0lf\n", absoluteY, absoluteX, my_b);
	// }

	// PHASE THREE: half-warps 0-15 use adjusted indexes to fetch from shared into registers,
	// and calculate their results
	if (threadIdx.y < 16) {
		adjust_x = threadIdx.x+1;
		adjust_y = threadIdx.y+1;
		top = shared_x[adjust_y-1][adjust_x];
		left = shared_x[adjust_y][adjust_x-1];
		right = shared_x[adjust_y][adjust_x+1];
		bottom = shared_x[adjust_y+1][adjust_x];
		my_res = my_b + top + left + right + bottom;
		my_res /= 4;
	}
	__syncthreads();

	// PHASE FOUR: 
	// write output to shared with adjusted indexes (for phase five to use)
	if (threadIdx.y < 16) {
		shared_x[adjust_y][adjust_x] = my_res;
	}
	__syncthreads();

	// PHASE FIVE:
	// half-warps 0-16 write to global memory while 16-17 update ghost buffers
	if (threadIdx.y < 16) {
		x_out[(absoluteY+1)*thr_per_row + absoluteX] = my_res;
	} else {
		finalwarp_idx = threadIdx.y - 16;
		ghost_out[n*(blockIdx.x*2 + 1+finalwarp_idx) + threadIdx.x + blockDim.x*blockIdx.y] =
			shared_x[threadIdx.x+1][1+finalwarp_idx*15];
	}
	// if (threadIdx.x + threadIdx.y == 0 && blockIdx.x == 0 && blockIdx.y == 1) {
	// 	for (int i = 0; i < 18; i++) {
	// 		for (int j = 0; j < 18; j++) {
	// 			printf("%.0lf, \t",shared_x[i][j]);
	// 		}
	// 		printf("\n");
	// 	}
	// }
}

__global__ void GPU_Residual_Helper(double* x, double* ghost, double* b, double* res_out)
{
	// shared memory prep, include ghost regions
	__shared__ double shared_x[18][18];
	// __shared__ double residual[16][16];

	// registers prep
	double my_b = 0;
	double my_res = 0;
	int finalwarp_idx = 0;

	int absoluteY = blockIdx.y*blockDim.x+threadIdx.y; // not blockDimy to allow for the overlap
	int thr_per_row = blockDim.x*gridDim.x;
	int absoluteX = blockIdx.x*blockDim.x+threadIdx.x;

	double top = 0;
	double left = 0;
	double right = 0;
	double bottom = 0;
	double me = 0;

	int adjust_y = 0;
	int adjust_x = 0;

	// int thr_per_block = blockDim.x*blockDim.x; // not y to allow for the overlap
	// int block_idx = gridDim.x*blockIdx.y + blockIdx.x;
	// int thread_idx = blockDim.x*threadIdx.y + threadIdx.x;
	
	// PHASE ONE: load 18x16 middle columns with aligned, coalesced fetch
	// shared_x[threadIdx.y][threadIdx.x+1] = x[block_idx*thr_per_block + thread_idx];
	// shared_x[threadIdx.y][threadIdx.x+1] = x[(blockIdx.y*blockDim.y+threadIdx.y)*blockDim.x + blockIdx.x*blockDim.x+threadIdx.x];
	shared_x[threadIdx.y][threadIdx.x+1] = x[absoluteY*thr_per_row + absoluteX];
	__syncthreads();

	// PHASE TWO: half-warps 0-15 fetch global b
	// while half-warps 16-17 fetch the ghost columns from ghost
	if (threadIdx.y < 16) {
		// my_b = b[I(absoluteY, absoluteX)];
		my_b = b[absoluteY*thr_per_row + absoluteX];
	} else {
		finalwarp_idx = threadIdx.y - 16;
		shared_x[threadIdx.x+1][finalwarp_idx*17] = ghost[n*(blockIdx.x*2 + finalwarp_idx*3) + 
			threadIdx.x + blockDim.x*blockIdx.y];
	}
	__syncthreads();

	// PHASE THREE: half-warps 0-15 use adjusted indexes to fetch from shared into registers,
	// and calculate their results
	if (threadIdx.y < 16) {
		adjust_x = threadIdx.x+1;
		adjust_y = threadIdx.y+1;
		top = shared_x[adjust_y-1][adjust_x];
		left = shared_x[adjust_y][adjust_x-1];
		right = shared_x[adjust_y][adjust_x+1];
		bottom = shared_x[adjust_y+1][adjust_x];
		me = shared_x[adjust_y][adjust_x];
		my_res = 4*me - top - left - right - bottom - my_b;
	}
	__syncthreads();

	if (threadIdx.y < 16) {
		// residual[absoluteY][absoluteY] = my_res*my_res;
		// atomicAdd(&res_out[0], my_res*my_res);
		res_out[(absoluteY)*thr_per_row + absoluteX] = my_res*my_res;
	}
}


////Your implementations end here
//////////////////////////////////////////////////////////////////////////

std::ofstream out;

//////////////////////////////////////////////////////////////////////////
////GPU test function
void Test_GPU_Solver()
{
	double* x=new double[s];
	memset(x,0x0000,sizeof(double)*s);
	double* b=new double[s];

	//////////////////////////////////////////////////////////////////////////
	////initialize x and b
	for(int i=-1;i<=n;i++){
		for(int j=-1;j<=n;j++){
			// b[I(i,j)]=4.0;		////set the values for the right-hand side
			b[I(i,j)]=100*i+j;
		}
	}
	for(int i=-1;i<=n;i++){
		for(int j=-1;j<=n;j++){
			if(B(i,j))
				x[I(i,j)]=(double)(i*i+j*j); ////set boundary condition for x
			else
				x[I(i,j)]=100*i+j;	
		}
	}

	

	// std::cout<<"\nactual x:\n";
	// for(int i=-1;i<n+1;i++){
	// 	for(int j=-1;j<n+1;j++){
	// 		std::cout<<x[I(i,j)]<<", \t";
	// 	}
	// 	std::cout<<std::endl;
	// }
	// std::cout<<std::endl;
	// std::cout<<std::endl;
	// std::cout<<"\nactual b:\n";
	// for(int i=-1;i<n+1;i++){
	// 	for(int j=-1;j<n+1;j++){
	// 		std::cout<<b[I(i,j)]<<", \t";
	// 	}
	// 	std::cout<<std::endl;
	// }
	// std::cout<<std::endl;
	// std::cout<<std::endl;

	// reformat memory to avoid column access
	const int my_s = (n+2)*n;
	double* x_host = new double[my_s];
	for(int i = -1;i <= n;i++){
		for(int j = 0;j < n;j++){
			int this_i = i+1;
			x_host[this_i*n+j] = x[I(i,j)];
			// std::cout<<x_host[this_i*n+j]<<", \t";
		}
		// std::cout<<std::endl;
	}
	// reformat memory to avoid column access
	const int my_b = n*n;
	double* b_host = new double[my_b];
	for(int i = 0;i < n;i++){
		for(int j = 0;j < n;j++){
			b_host[i*n+j] = b[I(i,j)];
			// std::cout<<x_host[this_i*n+j]<<", \t";
		}
		// std::cout<<std::endl;
	}
	
	// for(int i = 0;i < n+2;i++){
	// 	for(int j = 0;j < n;j++){
	// 		std::cout<<x_host[i*n+j]<<", \t";
	// 	}
	// 	std::cout<<std::endl;
	// }
	// std::cout<<std::endl;
	// std::cout<<std::endl;

	// for(int i = 0;i < n;i++){
	// 	for(int j = 0;j < n;j++){
	// 		std::cout<<b_host[i*n+j]<<", \t";
	// 	}
	// 	std::cout<<std::endl;
	// }
	// std::cout<<std::endl;
	// std::cout<<std::endl;

	const int ghost_cols = n/8+2;
	double* ghost_host = new double[ghost_cols*n];
	for (int j = 0; j < n; j++) {
		ghost_host[j] = x[I(j,-1)];
		ghost_host[(ghost_cols-1)*n+j] = x[I(j,n)];
	}

	for (int j = 0; j < n; j += 16) {
		int col = j/8;
		for (int i = 0; i < n; i++) {
			// std::cout<<col+1<<" "<<col+2<<" "<<i<<", ";
			ghost_host[(col+1)*n+i] = x[I(i,j)];
			ghost_host[(col+2)*n+i] = x[I(i,j+15)];
		}
		// std::cout<<std::endl;
	}
	// std::cout<<std::endl;

	// for (int i = 0; i < ghost_cols; i++) {
	// 	for (int j = 0; j < n; j++) {
	// 		std::cout<<ghost_host[i*n+j]<<", \t";
	// 	}
	// 	std::cout<<std::endl;
	// }
	// std::cout<<std::endl;
	// std::cout<<std::endl;

	// double-buffered
	double* x_dev[2];
	cudaMalloc((void **)&x_dev[0], my_s*sizeof(double));
	cudaMemcpy(x_dev[0], x_host, my_s*sizeof(double), cudaMemcpyHostToDevice);
	cudaMalloc((void **)&x_dev[1], my_s*sizeof(double));
	cudaMemcpy(x_dev[1], x_host, my_s*sizeof(double), cudaMemcpyHostToDevice);

	double* ghost_dev[2];
	cudaMalloc((void **)&ghost_dev[0], ghost_cols*n*sizeof(double));
	cudaMemcpy(ghost_dev[0], ghost_host, ghost_cols*n*sizeof(double), cudaMemcpyHostToDevice);
	cudaMalloc((void **)&ghost_dev[1], ghost_cols*n*sizeof(double));
	cudaMemcpy(ghost_dev[1], ghost_host, ghost_cols*n*sizeof(double), cudaMemcpyHostToDevice);

	// single buffer
	double* b_dev;
	cudaMalloc((void **)&b_dev, my_b*sizeof(double));
	cudaMemcpy(b_dev, b_host, my_b*sizeof(double), cudaMemcpyHostToDevice);

	// thrust::device_ptr<double> res_dev=thrust::device_malloc<double>(n*n);
	thrust::device_vector<double> res_thrust(n*n, 0);
	double* res_raw;
	// cudaMalloc((void **)&res_dev, n*n*sizeof(double));
	// double* res_host;
	cudaEvent_t start,end;
	cudaEventCreate(&start);
	cudaEventCreate(&end);
	float gpu_time=0.0f;
	cudaDeviceSynchronize();
	cudaEventRecord(start);
	//////////////////////////////////////////////////////////////////////////
	////TODO 2: call your GPU functions here
	////Requirement: You need to copy data from the CPU arrays, conduct computations on the GPU, and copy the values back from GPU to CPU
	////The final positions should be stored in the same place as the CPU function, i.e., the array of x
	////The correctness of your simulation will be evaluated by the residual (<1e-3)
	//////////////////////////////////////////////////////////////////////////
	int block_size = 16;
	int grid_dim = n/block_size;
	// std::std::cout << block_size <<std::endl;
	// std::cout <<grid_dim <<std::endl;
	int src = 0;
	int dest = 1;

	int iter_num=0;			////iteration number
	// int max_num=1e5;		////max iteration number
	int max_num = 10;
	double residual=10.0;	////residual

	for (; residual > tolerance && iter_num < max_num; iter_num++) {
		// residual = 0.0;
		src = iter_num & 1;
		dest = (src + 1) & 1;
		GPU_Jacobi<<<dim3(grid_dim, grid_dim), dim3(block_size, block_size+2)>>>(x_dev[src], ghost_dev[src], b_dev, x_dev[dest], ghost_dev[dest]);
		res_raw=thrust::raw_pointer_cast(res_thrust.data());
		GPU_Residual_Helper<<<dim3(grid_dim, grid_dim), dim3(block_size, block_size+2)>>>(x_dev[dest], ghost_dev[dest], b_dev, res_raw);
		// residual = res_raw[0];
		// cudaMemcpy(res_host, res_raw, n*n*sizeof(double), cudaMemcpyDeviceToHost);
		// for (int i = 0; i < n*n; i++) {
		// 	residual += res_host[i];
		// }
		// thrust::copy(res_dev,res_dev+n*n,res_thrust.begin());
		residual = thrust::reduce(res_thrust.begin(),res_thrust.end(),(double)0,thrust::plus<double>());
		// std::cout <<typeid(residual).name()<< std::endl;
		if(verbose)std::cout<<"res: "<<residual<<std::endl;
	}

	// std::cout << "host "<< x_host[512] <<std::endl;
	// std::cout << "original "<<x[I(15,0)] <<std::endl;
	// std::cout << "original "<<I(15,0) <<std::endl;

	
	cudaMemcpy(x_host, x_dev[dest], my_s*sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(ghost_host, ghost_dev[dest], ghost_cols*n*sizeof(double), cudaMemcpyDeviceToHost);
	
	for (int j = 0; j < n; j++) {
		x[I(j,-1)] = ghost_host[j];
		x[I(j,n)] = ghost_host[(ghost_cols-1)*n+j];
	}

	for (int j = 0; j < n; j += 16) {
		int col = j/8;
		for (int i = 0; i < n; i++) {
			// std::cout<<col+1<<" "<<col+2<<" "<<i<<", ";
			x[I(i,j)] = ghost_host[(col+1)*n+i];
			x[I(i,j+15)] = ghost_host[(col+2)*n+i];
		}
		// std::cout<<std::endl;
	}

	for(int i = -1;i <= n;i++){
		for(int j = 0;j < n;j++){
			int this_i = i+1;
			x[I(i,j)] = x_host[this_i*n+j];
			// std::cout<<x_host[this_i*n+j]<<", \t";
		}
		// std::cout<<std::endl;
	}
	


	cudaEventRecord(end);
	cudaEventSynchronize(end);
	cudaEventElapsedTime(&gpu_time,start,end);
	printf("\nGPU runtime: %.4f ms\n",gpu_time);
	cudaEventDestroy(start);
	cudaEventDestroy(end);
	//////////////////////////////////////////////////////////////////////////

	//output x
	if(verbose){
		std::cout<<"\n\nx for your GPU solver:\n";
		for(int i=-1;i<=n;i++){
			for(int j=-1;j<=n;j++){
				// std::cout<<x[I(i,j)]<<", ";
				printf("%.0lf, \t", x[I(i,j)]);
			}
			std::cout<<std::endl;
		}	
	}
	std::cout<<std::endl;
	std::cout<<std::endl;

	////calculate residual
	residual=0.0;
	for(int i=0;i<n;i++){
		for(int j=0;j<n;j++){
			residual+=pow(4.0*x[I(i,j)]-x[I(i-1,j)]-x[I(i+1,j)]-x[I(i,j-1)]-x[I(i,j+1)]-b[I(i,j)],2);
		}
	}
	std::cout<<"\n\nresidual for your GPU solver: "<<residual<<std::endl;

	std::cout<<"GPU Jacobi solver converges in "<<iter_num<<" iterations, with residual "<<residual<<std::endl;

	// std::out<<"R0: "<<residual<<std::endl;
	// std::out<<"T1: "<<gpu_time<<std::endl;

	//////////////////////////////////////////////////////////////////////////

	delete [] x;
	delete [] b;
}

int main()
{
	if(name::team=="Team_X"){
		printf("\nPlease specify your team name and team member names in name::team and name::author to start.\n");
		return 0;
	}

	std::string file_name=name::team+"_competition_3_linear_solver.dat";
	out.open(file_name.c_str());
	
	if(out.fail()){
		printf("\ncannot open file %s to record results\n",file_name.c_str());
		return 0;
	}

	#ifdef CPU
	Test_CPU_Solvers();	////You may comment out this line to run your GPU solver only
	#endif //CPU

	#ifndef CPU
	Test_GPU_Solver();	////Test function for your own GPU implementation
	#endif

	return 0;
}
