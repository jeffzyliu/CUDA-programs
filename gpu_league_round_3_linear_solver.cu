//////////////////////////////////////////////////////////////////////////
////This is the code implementation for GPU Premier League Round 3: sparse linear solver
//////////////////////////////////////////////////////////////////////////
#include <iostream>
#include <fstream>
using namespace std;

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

const int n=32;							////grid size, we will change this value to up to 256 to test your code
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
	double residual=0.0;	////residual

	do{
		////update x values using the Jacobi iterative scheme
		for(int i=0;i<n;i++){
			for(int j=0;j<n;j++){
				xw[I(i,j)]=(b[I(i,j)]+xr[I(i-1,j)]+xr[I(i+1,j)]+xr[I(i,j-1)]+xr[I(i,j+1)])/4.0;
			}
		}

		////calculate residual
		residual=0.0;
		for(int i=0;i<n;i++){
			for(int j=0;j<n;j++){
				residual+=pow(4.0*xw[I(i,j)]-xw[I(i-1,j)]-xw[I(i+1,j)]-xw[I(i,j-1)]-xw[I(i,j+1)]-b[I(i,j)],2);
			}
		}

		if(verbose)cout<<"res: "<<residual<<endl;

		////swap the buffers
		double* swap=xr;
		xr=xw;
		xw=swap;
		iter_num++;
	}while(residual>tolerance&&iter_num<max_num);	

	x=xr;

	cout<<"Jacobi solver converges in "<<iter_num<<" iterations, with residual "<<residual<<endl;

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

		if(verbose)cout<<"res: "<<residual<<endl;
		iter_num++;
	}while(residual>tolerance&&iter_num<max_num);	

	cout<<"Gauss-Seidel solver converges in "<<iter_num<<" iterations, with residual "<<residual<<endl;
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

		if(verbose)cout<<"res: "<<residual<<endl;
		iter_num++;
	}while(residual>tolerance&&iter_num<max_num);	

	cout<<"Red-Black Gauss-Seidel solver converges in "<<iter_num<<" iterations, with residual "<<residual<<endl;
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
	for(int i=-1;i<=n;i++){
		for(int j=-1;j<=n;j++){
			b[I(i,j)]=4.0;		////set the values for the right-hand side
		}
	}

	//////////////////////////////////////////////////////////////////////////
	////test Jacobi
	for(int i=-1;i<=n;i++){
		for(int j=-1;j<=n;j++){
			if(B(i,j))
				x[I(i,j)]=(double)(i*i+j*j);	////set boundary condition for x
		}
	}

	Jacobi_Solver(x,b);

	if(verbose){
		cout<<"\n\nx for Jacobi:\n";
		for(int i=0;i<n;i++){
			for(int j=0;j<n;j++){
				cout<<x[I(i,j)]<<", ";
			}
			cout<<std::endl;
		}	
	}
	cout<<"\n\n";

	//////////////////////////////////////////////////////////////////////////
	////test Gauss-Seidel
	memset(x,0x0000,sizeof(double)*s);
	for(int i=-1;i<=n;i++){
		for(int j=-1;j<=n;j++){
			if(B(i,j))
				x[I(i,j)]=(double)(i*i+j*j);	////set boundary condition for x
		}
	}

	Gauss_Seidel_Solver(x,b);

	if(verbose){
		cout<<"\n\nx for Gauss-Seidel:\n";
		for(int i=0;i<n;i++){
			for(int j=0;j<n;j++){
				cout<<x[I(i,j)]<<", ";
			}
			cout<<std::endl;
		}	
	}
	cout<<"\n\n";

	//////////////////////////////////////////////////////////////////////////
	////test Red-Black Gauss-Seidel
	memset(x,0x0000,sizeof(double)*s);
	for(int i=-1;i<=n;i++){
		for(int j=-1;j<=n;j++){
			if(B(i,j))
				x[I(i,j)]=(double)(i*i+j*j);	////set boundary condition for x
		}
	}

	Red_Black_Gauss_Seidel_Solver(x,b);

	if(verbose){
		cout<<"\n\nx for Red-Black Gauss-Seidel:\n";
		for(int i=0;i<n;i++){
			for(int j=0;j<n;j++){
				cout<<x[I(i,j)]<<", ";
			}
			cout<<std::endl;
		}	
	}
	cout<<"\n\n";

	//////////////////////////////////////////////////////////////////////////

	delete [] x;
	delete [] b;
}

//////////////////////////////////////////////////////////////////////////
////TODO 1: your GPU variables and functions start here
// #define blockX 16
// #define blockY 18
__global__ void GPU_Jacobi(double* x, double* ghost, double* b)
{
	// shared memory prep, include ghost regions
	__shared__ double shared_x[18][18];
	// registers prep
	double my_b = 0;
	int absoluteY = blockIdx.y*blockDim.x+threadIdx.y; // not blockDimy to allow for the overlap
	int thr_per_row = blockDim.x*gridDim.x;
	int absoluteX = blockIdx.x*blockDim.x+threadIdx.x;
	// int thr_per_block = blockDim.x*blockDim.x; // not y to allow for the overlap
	// int block_idx = gridDim.x*blockIdx.y + blockIdx.x;
	// int thread_idx = blockDim.x*threadIdx.y + threadIdx.x;
	
	// PHASE ONE: load 18x16 middle columns with aligned, coalesced fetch
	// shared_x[threadIdx.y][threadIdx.x+1] = x[block_idx*thr_per_block + thread_idx];
	// shared_x[threadIdx.y][threadIdx.x+1] = x[(blockIdx.y*blockDim.y+threadIdx.y)*blockDim.x + blockIdx.x*blockDim.x+threadIdx.x];
	shared_x[threadIdx.y][threadIdx.x+1] = x[absoluteY*thr_per_row + absoluteX];
	__syncthreads();

	// PHASE TWO: half-warps 0-15 fetch global b, remember to add 1 to row/col in shared now
	// while half-warps 16-17 fetch the ghost columns from ghost
	if (threadIdx.y < 16) {
		// my_b = b[I(absoluteY, absoluteX)];
		my_b = b[absoluteY*thr_per_row + absoluteX];
	} else {
		int finalwarp_idx = threadIdx.y - 16;
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
}


////Your implementations end here
//////////////////////////////////////////////////////////////////////////

ofstream out;

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
				x[I(i,j)]=(double)(i*i+j*j);
			else
				x[I(i,j)]=100*i+j;	////set boundary condition for x
		}
	}

	cudaEvent_t start,end;
	cudaEventCreate(&start);
	cudaEventCreate(&end);
	float gpu_time=0.0f;
	cudaDeviceSynchronize();
	cudaEventRecord(start);

	// cout<<"\nactual x:\n";
	// for(int i=-1;i<n+1;i++){
	// 	for(int j=-1;j<n+1;j++){
	// 		cout<<x[I(i,j)]<<", \t";
	// 	}
	// 	cout<<std::endl;
	// }
	// cout<<std::endl;
	// cout<<std::endl;
	// cout<<"\nactual b:\n";
	// for(int i=-1;i<n+1;i++){
	// 	for(int j=-1;j<n+1;j++){
	// 		cout<<b[I(i,j)]<<", \t";
	// 	}
	// 	cout<<std::endl;
	// }
	// cout<<std::endl;
	// cout<<std::endl;

	// reformat memory to avoid column access
	const int my_s = (n+2)*n;
	double* x_host = new double[my_s];
	for(int i = -1;i <= n;i++){
		for(int j = 0;j < n;j++){
			int this_i = i+1;
			x_host[this_i*n+j] = x[I(i,j)];
			// cout<<x_host[this_i*n+j]<<", \t";
		}
		// cout<<std::endl;
	}
	// reformat memory to avoid column access
	const int my_b = n*n;
	double* b_host = new double[my_b];
	for(int i = 0;i < n;i++){
		for(int j = 0;j < n;j++){
			b_host[i*n+j] = b[I(i,j)];
			// cout<<x_host[this_i*n+j]<<", \t";
		}
		// cout<<std::endl;
	}
	
	// for(int i = 0;i < n+2;i++){
	// 	for(int j = 0;j < n;j++){
	// 		cout<<x_host[i*n+j]<<", \t";
	// 	}
	// 	cout<<std::endl;
	// }
	// cout<<std::endl;
	// cout<<std::endl;

	// for(int i = 0;i < n;i++){
	// 	for(int j = 0;j < n;j++){
	// 		cout<<b_host[i*n+j]<<", \t";
	// 	}
	// 	cout<<std::endl;
	// }
	// cout<<std::endl;
	// cout<<std::endl;

	const int ghost_cols = n/8+2;
	double* ghost_host = new double[ghost_cols*n];
	for (int j = 0; j < n; j++) {
		ghost_host[j] = x[I(j,-1)];
		ghost_host[(ghost_cols-1)*n+j] = x[I(j,n)];
	}

	for (int j = 0; j < n; j += 16) {
		int col = j/8;
		for (int i = 0; i < n; i++) {
			// cout<<col+1<<" "<<col+2<<" "<<i<<", ";
			ghost_host[(col+1)*n+i] = x[I(i,j)];
			ghost_host[(col+2)*n+i] = x[I(i,j+15)];
		}
		// cout<<std::endl;
	}
	// cout<<std::endl;

	// for (int i = 0; i < ghost_cols; i++) {
	// 	for (int j = 0; j < n; j++) {
	// 		cout<<ghost_host[i*n+j]<<", \t";
	// 	}
	// 	cout<<std::endl;
	// }
	// cout<<std::endl;
	// cout<<std::endl;

	double* x_dev = nullptr;
	cudaMalloc((void **)&x_dev, my_s*sizeof(double));
	cudaMemcpy(x_dev, x_host, my_s*sizeof(double), cudaMemcpyHostToDevice);
	double* b_dev = nullptr;
	cudaMalloc((void **)&b_dev, my_b*sizeof(double));
	cudaMemcpy(b_dev, b_host, my_b*sizeof(double), cudaMemcpyHostToDevice);
	double* ghost_dev = nullptr;
	cudaMalloc((void **)&ghost_dev, ghost_cols*n*sizeof(double));
	cudaMemcpy(ghost_dev, ghost_host, ghost_cols*n*sizeof(double), cudaMemcpyHostToDevice);


	//////////////////////////////////////////////////////////////////////////
	////TODO 2: call your GPU functions here
	////Requirement: You need to copy data from the CPU arrays, conduct computations on the GPU, and copy the values back from GPU to CPU
	////The final positions should be stored in the same place as the CPU function, i.e., the array of x
	////The correctness of your simulation will be evaluated by the residual (<1e-3)
	//////////////////////////////////////////////////////////////////////////
	int block_size = 16;
	int grid_dim = n/block_size;
	// cout << block_size <<endl;
	// cout <<grid_dim << endl;

	GPU_Jacobi<<<dim3(grid_dim, grid_dim), dim3(block_size, block_size+2)>>>(x_dev, ghost_dev, b_dev);

	// cout << "host "<< x_host[512] << endl;
	// cout << "original "<<x[I(15,0)] << endl;
	// cout << "original "<<I(15,0) << endl;
	cudaEventRecord(end);
	cudaEventSynchronize(end);
	cudaEventElapsedTime(&gpu_time,start,end);
	printf("\nGPU runtime: %.4f ms\n",gpu_time);
	cudaEventDestroy(start);
	cudaEventDestroy(end);
	//////////////////////////////////////////////////////////////////////////

	////output x
	// if(verbose){
	// 	cout<<"\n\nx for your GPU solver:\n";
	// 	for(int i=0;i<n;i++){
	// 		for(int j=0;j<n;j++){
	// 			cout<<x[I(i,j)]<<", ";
	// 		}
	// 		cout<<std::endl;
	// 	}	
	// }

	// ////calculate residual
	// double residual=0.0;
	// for(int i=0;i<n;i++){
	// 	for(int j=0;j<n;j++){
	// 		residual+=pow(4.0*x[I(i,j)]-x[I(i-1,j)]-x[I(i+1,j)]-x[I(i,j-1)]-x[I(i,j+1)]-b[I(i,j)],2);
	// 	}
	// }
	// cout<<"\n\nresidual for your GPU solver: "<<residual<<endl;

	// out<<"R0: "<<residual<<endl;
	// out<<"T1: "<<gpu_time<<endl;

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

	// Test_CPU_Solvers();	////You may comment out this line to run your GPU solver only
	Test_GPU_Solver();	////Test function for your own GPU implementation

	return 0;
}
