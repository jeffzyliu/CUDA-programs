//////////////////////////////////////////////////////////////////////////
////This is the code implementation for GPU Premier League Round 1
//////////////////////////////////////////////////////////////////////////
#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>

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

////This is a matrix class to carry out linear algebra operations on both GPU and CPU
////It is the same as the sample code I showed in class on Week 3. 

////NOTICE: You do not have to change the implementation in this class. 
////But if you do want to change part of it for performance reasons, please let us known by writting a submission note on Canvas.

class Matrix{
public:
    int m=0;							////number of rows
    int n=0;							////number of columns
	vector<float> elements_on_host;		////we use a std::vector for the element array on host
    float* elements_on_dev=0;			////we use a pointer for the element array on device
	bool on_host=true;

	////constructors
	__host__ Matrix(){}

	__host__ Matrix(const int _m,const int _n,bool _on_host=true)
	{
		on_host=_on_host;
		if(on_host)Resize_On_Host(_m,_n);
		else Resize_On_Device(_m,_n);
	}

	////destructor
	__host__ ~Matrix()
	{
		if(!on_host&&elements_on_dev!=0) cudaFree(elements_on_dev);		
	}

	////Resize on host or device
	__host__ void Resize_On_Host(const int _m,const int _n)
	{
		if(m==_m&&n==_n)return;
		m=_m;
		n=_n;
		elements_on_host.resize(m*n);
	}

	__host__ void Resize_On_Device(const int _m,const int _n)
	{
		if(m==_m&&n==_n)return;
		m=_m;
		n=_n;
		if(elements_on_dev!=0)cudaFree(elements_on_dev);
		cudaMalloc((void**)&elements_on_dev,m*n*sizeof(float));
	}

	////random access a matrix element
	inline __host__ float& operator() (const int i,const int j)
	{
		return elements_on_host[i*n+j];
	}

	inline __host__ const float& operator() (const int i,const int j) const
	{
		return elements_on_host[i*n+j];
	}

	////copy data with four cases (CPU->CPU, GPU->CPU, GPU->GPU, CPU->GPU)
	__host__ Matrix& operator= (const Matrix& mtx)
	{
		if(on_host&&mtx.on_host){
			Resize_On_Host(mtx.m,mtx.n);
			elements_on_host=mtx.elements_on_host;
		}
		else if(on_host&&!mtx.on_host){
			Resize_On_Host(mtx.m,mtx.n);
			cudaMemcpy(&elements_on_host[0],mtx.elements_on_dev,m*n*sizeof(float),cudaMemcpyDeviceToHost);
		}
		else if(!on_host&&!mtx.on_host){
			Resize_On_Device(mtx.m,mtx.n);
			cudaMemcpy(elements_on_dev,mtx.elements_on_dev,mtx.m*n*sizeof(float),cudaMemcpyDeviceToDevice);
		}
		else if(!on_host&&mtx.on_host){
			Resize_On_Device(mtx.m,mtx.n);
			cudaMemcpy(elements_on_dev,&mtx.elements_on_host[0],m*n*sizeof(float),cudaMemcpyHostToDevice);
		}
		return *this;
	}

	////print matrix elements on screen
	__host__ friend ostream & operator << (ostream &out,const Matrix &mtx)
	{
		if(!mtx.on_host)
			cout<<"Print for matrix on device is not supported."<<endl;

		for(int i=0;i<mtx.m;i++){
			for(int j=0;j<mtx.n;j++){
				out<<mtx(i,j)<<", ";
			}
			out<<std::endl;
		}
		return out;
	}
};

//////////////////////////////////////////////////////////////////////////
////Your tasks start!

////This is a sample implementation without using any memory hierarchy
////The function calculates C=A*B, with dimA=[Am,An], dimB=[Bm,Bn], dimC=[Am,bn], and An=Bm
__global__ void Matrix_Multiplication_AB_Kernel_Poorman(const float* Ae,const float* Be,float* Ce,const int Am,const int An,const int Bn)
{
	int i=blockIdx.x*blockDim.x+threadIdx.x;
	int j=blockIdx.y*blockDim.y+threadIdx.y;

	float val=0.f;
	for(int k=0;k<An;k++)
		val+=Ae[i*An+k]*Be[k*Bn+j];
	Ce[i*Bn+j]=val;
} 

//////////////////////////////////////////////////////////////////////////
////Task 1: implement your fast matrix-matrix multiplication in the following kernel function.
////The function parameters are the same as the sample function:
////The function calculates C=A*B, with dimA=[Am,An], dimB=[Bm,Bn], dimC=[Am,bn], and An=Bm
//////////////////////////////////////////////////////////////////////////

__global__ void Matrix_Multiplication_AB_Kernel_Your_Version(const float* Ae,const float* Be,float* Ce,const int Am,const int An,const int Bn)
{
	// initialize memory
	const int block_size = 32;
	const int num_tiles = An / block_size;
	__shared__ float a_shared[block_size][block_size];
	__shared__ float b_shared[block_size][block_size];
	__shared__ float c_shared[block_size][block_size];

	// calculate 1d index of correct item on A, B, C
	int thr_per_block = blockDim.y * blockDim.x;
	int c_idx = blockIdx.y * gridDim.x * thr_per_block + threadIdx.y * gridDim.x * blockDim.x + blockIdx.x * blockDim.x + threadIdx.x;
	c_shared[threadIdx.y][threadIdx.x] = 0; // set everything to zero just the first time
	int a_idx, b_idx;

	for (int tile = 0; tile < num_tiles; ++tile) {
		// want blockIdx.x to increment
		a_idx = blockIdx.y * num_tiles * thr_per_block + threadIdx.y * num_tiles * blockDim.x + tile * blockDim.x + threadIdx.x;
		// want blockIdx.y to increment
		b_idx = tile * gridDim.x * thr_per_block + threadIdx.y * gridDim.x * blockDim.x + blockIdx.x * blockDim.x + threadIdx.x;
		a_shared[threadIdx.y][threadIdx.x] = Ae[a_idx];
		b_shared[threadIdx.y][threadIdx.x] = Be[b_idx];
		__syncthreads();

		// lmao loop unrolling time my dudes
		c_shared[threadIdx.y][threadIdx.x] += a_shared[threadIdx.y][0] * b_shared[0][threadIdx.x];
		c_shared[threadIdx.y][threadIdx.x] += a_shared[threadIdx.y][1] * b_shared[1][threadIdx.x];
		c_shared[threadIdx.y][threadIdx.x] += a_shared[threadIdx.y][2] * b_shared[2][threadIdx.x];
		c_shared[threadIdx.y][threadIdx.x] += a_shared[threadIdx.y][3] * b_shared[3][threadIdx.x];
		c_shared[threadIdx.y][threadIdx.x] += a_shared[threadIdx.y][4] * b_shared[4][threadIdx.x];
		c_shared[threadIdx.y][threadIdx.x] += a_shared[threadIdx.y][5] * b_shared[5][threadIdx.x];
		c_shared[threadIdx.y][threadIdx.x] += a_shared[threadIdx.y][6] * b_shared[6][threadIdx.x];
		c_shared[threadIdx.y][threadIdx.x] += a_shared[threadIdx.y][7] * b_shared[7][threadIdx.x];
		c_shared[threadIdx.y][threadIdx.x] += a_shared[threadIdx.y][8] * b_shared[8][threadIdx.x];
		c_shared[threadIdx.y][threadIdx.x] += a_shared[threadIdx.y][9] * b_shared[9][threadIdx.x];
		c_shared[threadIdx.y][threadIdx.x] += a_shared[threadIdx.y][10] * b_shared[10][threadIdx.x];
		c_shared[threadIdx.y][threadIdx.x] += a_shared[threadIdx.y][11] * b_shared[11][threadIdx.x];
		c_shared[threadIdx.y][threadIdx.x] += a_shared[threadIdx.y][12] * b_shared[12][threadIdx.x];
		c_shared[threadIdx.y][threadIdx.x] += a_shared[threadIdx.y][13] * b_shared[13][threadIdx.x];
		c_shared[threadIdx.y][threadIdx.x] += a_shared[threadIdx.y][14] * b_shared[14][threadIdx.x];
		c_shared[threadIdx.y][threadIdx.x] += a_shared[threadIdx.y][15] * b_shared[15][threadIdx.x];

		c_shared[threadIdx.y][threadIdx.x] += a_shared[threadIdx.y][16] * b_shared[16][threadIdx.x];
		c_shared[threadIdx.y][threadIdx.x] += a_shared[threadIdx.y][17] * b_shared[17][threadIdx.x];
		c_shared[threadIdx.y][threadIdx.x] += a_shared[threadIdx.y][18] * b_shared[18][threadIdx.x];
		c_shared[threadIdx.y][threadIdx.x] += a_shared[threadIdx.y][19] * b_shared[19][threadIdx.x];
		c_shared[threadIdx.y][threadIdx.x] += a_shared[threadIdx.y][20] * b_shared[20][threadIdx.x];
		c_shared[threadIdx.y][threadIdx.x] += a_shared[threadIdx.y][21] * b_shared[21][threadIdx.x];
		c_shared[threadIdx.y][threadIdx.x] += a_shared[threadIdx.y][22] * b_shared[22][threadIdx.x];
		c_shared[threadIdx.y][threadIdx.x] += a_shared[threadIdx.y][23] * b_shared[23][threadIdx.x];
		c_shared[threadIdx.y][threadIdx.x] += a_shared[threadIdx.y][24] * b_shared[24][threadIdx.x];
		c_shared[threadIdx.y][threadIdx.x] += a_shared[threadIdx.y][25] * b_shared[25][threadIdx.x];
		c_shared[threadIdx.y][threadIdx.x] += a_shared[threadIdx.y][26] * b_shared[26][threadIdx.x];
		c_shared[threadIdx.y][threadIdx.x] += a_shared[threadIdx.y][27] * b_shared[27][threadIdx.x];
		c_shared[threadIdx.y][threadIdx.x] += a_shared[threadIdx.y][28] * b_shared[28][threadIdx.x];
		c_shared[threadIdx.y][threadIdx.x] += a_shared[threadIdx.y][29] * b_shared[29][threadIdx.x];
		c_shared[threadIdx.y][threadIdx.x] += a_shared[threadIdx.y][30] * b_shared[30][threadIdx.x];
		c_shared[threadIdx.y][threadIdx.x] += a_shared[threadIdx.y][31] * b_shared[31][threadIdx.x];
		__syncthreads();
	}
	// save to global
	Ce[c_idx] = c_shared[threadIdx.y][threadIdx.x];
}

////This is a sample implementation without using any memory hierarchy
////The function calculates the matrix multiplication, with C=A^T*B*A, A^T is the transpose of A, dimA=[Am,An], dimB=[Am,Am], and dimC=[An,An]
__global__ void Matrix_Multiplication_ATBA_Kernel_Poorman(const float* Ae,const float* Be,float* Ce,const int Am,const int An)
{
	int i=blockIdx.x*blockDim.x+threadIdx.x;
	int j=blockIdx.y*blockDim.y+threadIdx.y;
	
	float val=0.f;
	for(int l=0;l<Am;l++)
		for(int k=0;k<Am;k++)
			val+=Ae[l*An+i]*Be[l*Am+k]*Ae[k*An+j];
	Ce[i*An+j]=val;
}

//////////////////////////////////////////////////////////////////////////
////Task 2: calculate the matrix multiplication in the following kernel function. 
////The function parameters are the same as the sample function:
////The function calculates the matrix multiplication, with C=A^T*B*A, A^T is the transpose of A, dimA=[Am,An], dimB=[Am,Am], and dimC=[An,An]
//////////////////////////////////////////////////////////////////////////

__global__ void Matrix_Multiplication_ATBA_Kernel_Your_Version(const float* Ae,const float* Be,float* Ce,const int Am,const int An)
{
	// memory setup
	const int num_tiles = Am / 32;
	__shared__ float aTT_shared[32][32];
	__shared__ float b_shared[32][32];
	__shared__ float a_shared[32][32];
	__shared__ float accum_shared[32][32];
	__shared__ float c_shared[32][32];

	// coordinate setup
	int thr_per_block = blockDim.y * blockDim.x;
	int c_idx = blockIdx.y*gridDim.x*thr_per_block + threadIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x;
	int a_idx, b_idx, aTT_idx;

	// initialize memory
	c_shared[threadIdx.y][threadIdx.x] = 0;

	// following psuedocode coordinates are (y,x)
	for (int ay = 0; ay < num_tiles; ++ay) { //ay = bx
		// load a(ay,blockIdx.x)
		a_idx = ay*gridDim.x*thr_per_block + threadIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x;
		a_shared[threadIdx.y][threadIdx.x] = Ae[a_idx];
		// clear accumulator
		accum_shared[threadIdx.y][threadIdx.x] = 0;
		__syncthreads();
		for (int by = 0; by < num_tiles; ++by) { // by = aTx = aTTy
			// calculate indices
			b_idx = by*num_tiles*thr_per_block + threadIdx.y*num_tiles*blockDim.x + ay*blockDim.x + threadIdx.x;
			aTT_idx = by*gridDim.x*thr_per_block + threadIdx.y*gridDim.x*blockDim.x + blockIdx.y*blockDim.x + threadIdx.x;

			// load aTT(by, blockIdx.y) (since we load A but column access) and b(by,ay)
			b_shared[threadIdx.y][threadIdx.x] = Be[b_idx];
			aTT_shared[threadIdx.y][threadIdx.x] = Ae[aTT_idx];
			__syncthreads();

			// multiply aT x b, accumulate
			accum_shared[threadIdx.y][threadIdx.x] += aTT_shared[0][threadIdx.y] * b_shared[0][threadIdx.x];
			accum_shared[threadIdx.y][threadIdx.x] += aTT_shared[1][threadIdx.y] * b_shared[1][threadIdx.x];
			accum_shared[threadIdx.y][threadIdx.x] += aTT_shared[2][threadIdx.y] * b_shared[2][threadIdx.x];
			accum_shared[threadIdx.y][threadIdx.x] += aTT_shared[3][threadIdx.y] * b_shared[3][threadIdx.x];
			accum_shared[threadIdx.y][threadIdx.x] += aTT_shared[4][threadIdx.y] * b_shared[4][threadIdx.x];
			accum_shared[threadIdx.y][threadIdx.x] += aTT_shared[5][threadIdx.y] * b_shared[5][threadIdx.x];
			accum_shared[threadIdx.y][threadIdx.x] += aTT_shared[6][threadIdx.y] * b_shared[6][threadIdx.x];
			accum_shared[threadIdx.y][threadIdx.x] += aTT_shared[7][threadIdx.y] * b_shared[7][threadIdx.x];
			accum_shared[threadIdx.y][threadIdx.x] += aTT_shared[8][threadIdx.y] * b_shared[8][threadIdx.x];
			accum_shared[threadIdx.y][threadIdx.x] += aTT_shared[9][threadIdx.y] * b_shared[9][threadIdx.x];
			accum_shared[threadIdx.y][threadIdx.x] += aTT_shared[10][threadIdx.y] * b_shared[10][threadIdx.x];
			accum_shared[threadIdx.y][threadIdx.x] += aTT_shared[11][threadIdx.y] * b_shared[11][threadIdx.x];
			accum_shared[threadIdx.y][threadIdx.x] += aTT_shared[12][threadIdx.y] * b_shared[12][threadIdx.x];
			accum_shared[threadIdx.y][threadIdx.x] += aTT_shared[13][threadIdx.y] * b_shared[13][threadIdx.x];
			accum_shared[threadIdx.y][threadIdx.x] += aTT_shared[14][threadIdx.y] * b_shared[14][threadIdx.x];
			accum_shared[threadIdx.y][threadIdx.x] += aTT_shared[15][threadIdx.y] * b_shared[15][threadIdx.x];

			accum_shared[threadIdx.y][threadIdx.x] += aTT_shared[16][threadIdx.y] * b_shared[16][threadIdx.x];
			accum_shared[threadIdx.y][threadIdx.x] += aTT_shared[17][threadIdx.y] * b_shared[17][threadIdx.x];
			accum_shared[threadIdx.y][threadIdx.x] += aTT_shared[18][threadIdx.y] * b_shared[18][threadIdx.x];
			accum_shared[threadIdx.y][threadIdx.x] += aTT_shared[19][threadIdx.y] * b_shared[19][threadIdx.x];
			accum_shared[threadIdx.y][threadIdx.x] += aTT_shared[20][threadIdx.y] * b_shared[20][threadIdx.x];
			accum_shared[threadIdx.y][threadIdx.x] += aTT_shared[21][threadIdx.y] * b_shared[21][threadIdx.x];
			accum_shared[threadIdx.y][threadIdx.x] += aTT_shared[22][threadIdx.y] * b_shared[22][threadIdx.x];
			accum_shared[threadIdx.y][threadIdx.x] += aTT_shared[23][threadIdx.y] * b_shared[23][threadIdx.x];
			accum_shared[threadIdx.y][threadIdx.x] += aTT_shared[24][threadIdx.y] * b_shared[24][threadIdx.x];
			accum_shared[threadIdx.y][threadIdx.x] += aTT_shared[25][threadIdx.y] * b_shared[25][threadIdx.x];
			accum_shared[threadIdx.y][threadIdx.x] += aTT_shared[26][threadIdx.y] * b_shared[26][threadIdx.x];
			accum_shared[threadIdx.y][threadIdx.x] += aTT_shared[27][threadIdx.y] * b_shared[27][threadIdx.x];
			accum_shared[threadIdx.y][threadIdx.x] += aTT_shared[28][threadIdx.y] * b_shared[28][threadIdx.x];
			accum_shared[threadIdx.y][threadIdx.x] += aTT_shared[29][threadIdx.y] * b_shared[29][threadIdx.x];
			accum_shared[threadIdx.y][threadIdx.x] += aTT_shared[30][threadIdx.y] * b_shared[30][threadIdx.x];
			accum_shared[threadIdx.y][threadIdx.x] += aTT_shared[31][threadIdx.y] * b_shared[31][threadIdx.x];
			__syncthreads();
		}
		// multiply accum x a, add to c
		c_shared[threadIdx.y][threadIdx.x] += accum_shared[threadIdx.y][0] * a_shared[0][threadIdx.x];
		c_shared[threadIdx.y][threadIdx.x] += accum_shared[threadIdx.y][1] * a_shared[1][threadIdx.x];
		c_shared[threadIdx.y][threadIdx.x] += accum_shared[threadIdx.y][2] * a_shared[2][threadIdx.x];
		c_shared[threadIdx.y][threadIdx.x] += accum_shared[threadIdx.y][3] * a_shared[3][threadIdx.x];
		c_shared[threadIdx.y][threadIdx.x] += accum_shared[threadIdx.y][4] * a_shared[4][threadIdx.x];
		c_shared[threadIdx.y][threadIdx.x] += accum_shared[threadIdx.y][5] * a_shared[5][threadIdx.x];
		c_shared[threadIdx.y][threadIdx.x] += accum_shared[threadIdx.y][6] * a_shared[6][threadIdx.x];
		c_shared[threadIdx.y][threadIdx.x] += accum_shared[threadIdx.y][7] * a_shared[7][threadIdx.x];
		c_shared[threadIdx.y][threadIdx.x] += accum_shared[threadIdx.y][8] * a_shared[8][threadIdx.x];
		c_shared[threadIdx.y][threadIdx.x] += accum_shared[threadIdx.y][9] * a_shared[9][threadIdx.x];
		c_shared[threadIdx.y][threadIdx.x] += accum_shared[threadIdx.y][10] * a_shared[10][threadIdx.x];
		c_shared[threadIdx.y][threadIdx.x] += accum_shared[threadIdx.y][11] * a_shared[11][threadIdx.x];
		c_shared[threadIdx.y][threadIdx.x] += accum_shared[threadIdx.y][12] * a_shared[12][threadIdx.x];
		c_shared[threadIdx.y][threadIdx.x] += accum_shared[threadIdx.y][13] * a_shared[13][threadIdx.x];
		c_shared[threadIdx.y][threadIdx.x] += accum_shared[threadIdx.y][14] * a_shared[14][threadIdx.x];
		c_shared[threadIdx.y][threadIdx.x] += accum_shared[threadIdx.y][15] * a_shared[15][threadIdx.x];
		
		c_shared[threadIdx.y][threadIdx.x] += accum_shared[threadIdx.y][16] * a_shared[16][threadIdx.x];
		c_shared[threadIdx.y][threadIdx.x] += accum_shared[threadIdx.y][17] * a_shared[17][threadIdx.x];
		c_shared[threadIdx.y][threadIdx.x] += accum_shared[threadIdx.y][18] * a_shared[18][threadIdx.x];
		c_shared[threadIdx.y][threadIdx.x] += accum_shared[threadIdx.y][19] * a_shared[19][threadIdx.x];
		c_shared[threadIdx.y][threadIdx.x] += accum_shared[threadIdx.y][20] * a_shared[20][threadIdx.x];
		c_shared[threadIdx.y][threadIdx.x] += accum_shared[threadIdx.y][21] * a_shared[21][threadIdx.x];
		c_shared[threadIdx.y][threadIdx.x] += accum_shared[threadIdx.y][22] * a_shared[22][threadIdx.x];
		c_shared[threadIdx.y][threadIdx.x] += accum_shared[threadIdx.y][23] * a_shared[23][threadIdx.x];
		c_shared[threadIdx.y][threadIdx.x] += accum_shared[threadIdx.y][24] * a_shared[24][threadIdx.x];
		c_shared[threadIdx.y][threadIdx.x] += accum_shared[threadIdx.y][25] * a_shared[25][threadIdx.x];
		c_shared[threadIdx.y][threadIdx.x] += accum_shared[threadIdx.y][26] * a_shared[26][threadIdx.x];
		c_shared[threadIdx.y][threadIdx.x] += accum_shared[threadIdx.y][27] * a_shared[27][threadIdx.x];
		c_shared[threadIdx.y][threadIdx.x] += accum_shared[threadIdx.y][28] * a_shared[28][threadIdx.x];
		c_shared[threadIdx.y][threadIdx.x] += accum_shared[threadIdx.y][29] * a_shared[29][threadIdx.x];
		c_shared[threadIdx.y][threadIdx.x] += accum_shared[threadIdx.y][30] * a_shared[30][threadIdx.x];
		c_shared[threadIdx.y][threadIdx.x] += accum_shared[threadIdx.y][31] * a_shared[31][threadIdx.x];
		__syncthreads();
	}
	// save c to global
	Ce[c_idx] = c_shared[threadIdx.y][threadIdx.x];
}

//////////////////////////////////////////////////////////////////////////
////Task 3:  calculate the Frobenius norm of a matrix
////The definition of F-norm for a matrix is square root of (the sum of squares of all the matrix elements), i.e., F=sqrt(sum_(A_ij^2))
////See the definition: https://mathworld.wolfram.com/FrobeniusNorm.html
//////////////////////////////////////////////////////////////////////////

////Please write your own kernel function here, and call it in the function Test_F_Norm_On_GPU to test its correctness and performance
__global__ void F_Norm_On_GPU_Lazy(const float* Ae, float* sum)
{
	// lazy man's method for reference
	__shared__ float a_shared[16][16];
	int thr_per_block = blockDim.y * blockDim.x;
	int idx = blockIdx.y*gridDim.x*thr_per_block + threadIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x;
	
	float element = Ae[idx];
	a_shared[threadIdx.y][threadIdx.x] = element * element;
	atomicAdd(&sum[0], a_shared[threadIdx.y][threadIdx.x]);
}

__global__ void F_Norm_On_GPU(const float* Ae, float* Be, bool round1)
{
	extern __shared__ float data[];
	int idx = blockIdx.x*blockDim.x*2 + threadIdx.x;
	// use 2 registers
	float num1 = Ae[idx];
	float num2 = Ae[idx + blockDim.x]; // offset by stride is better for alignment

	// only square first time
	if (round1) {
		num1 *= num1;
		num2 *= num2;
	}
	// add two elements into one shared index
	data[threadIdx.x] = num1 + num2;
	__syncthreads();
	
	// from reduce4 in class notes
	for (unsigned int s = blockDim.x/2; s > 0; s >>= 1) {
		if(threadIdx.x < s){
			data[threadIdx.x]+=data[threadIdx.x+s];
		}
		__syncthreads();
	}
	if (threadIdx.x == 0) Be[blockIdx.x] = data[0];
}
////Congratulations, your tasks are all finished!
//////////////////////////////////////////////////////////////////////////


////Here are the test functions for your three kernel implementations

ofstream out;

__host__ void Test_Matrix_Multiplication_AB_On_GPU(const Matrix& A,const Matrix& B,Matrix& C)
{
	//// Load A and B to device memory
	Matrix A_on_dev(A.m,A.n,false);
	A_on_dev=A;
	Matrix B_on_dev(B.m,B.n,false);
	B_on_dev=B;

	//// Allocate C in device memory
	Matrix C_on_dev(A_on_dev.m,B_on_dev.n,false);

	cudaEvent_t start,end;
	cudaEventCreate(&start);
	cudaEventCreate(&end);
	float gpu_time=0.0f;
	cudaDeviceSynchronize();
	cudaEventRecord(start);

	//// Invoke kernel
	const int block_size=32;
	const int block_num_x=C.m/block_size;
	const int block_num_y=C.n/block_size;

	#ifdef POORMAN
	Matrix_Multiplication_AB_Kernel_Poorman<<<dim3(block_num_x,block_num_y),dim3(block_size,block_size)>>>(A_on_dev.elements_on_dev,B_on_dev.elements_on_dev,C_on_dev.elements_on_dev,A_on_dev.m,A_on_dev.n,B_on_dev.n);
	#endif

	#ifndef POORMAN
	Matrix_Multiplication_AB_Kernel_Your_Version<<<dim3(block_num_y,block_num_x),dim3(block_size,block_size)>>>(A_on_dev.elements_on_dev,B_on_dev.elements_on_dev,C_on_dev.elements_on_dev,A_on_dev.m,A_on_dev.n,B_on_dev.n);
	#endif

	cudaEventRecord(end);
	cudaEventSynchronize(end);
	cudaEventElapsedTime(&gpu_time,start,end);
	printf("\nGPU runtime for matrix multiplication AB: %.4f ms\n",gpu_time);
	cudaEventDestroy(start);
	cudaEventDestroy(end);

	//// Transfer data back to CPU
	C=C_on_dev;

	out<<"T1: "<<gpu_time<<endl;
}

__host__ void Test_Matrix_Multiplication_ATBA_On_GPU(const Matrix& A,const Matrix& B,Matrix& C)
{
	//// Load A and B to device memory
	Matrix A_on_dev(A.m,A.n,false);
	A_on_dev=A;
	Matrix B_on_dev(B.m,B.n,false);
	B_on_dev=B;
	
	//// Allocate C in device memory
	Matrix C_on_dev(A_on_dev.n,A_on_dev.n,false);

	cudaEvent_t start,end;
	cudaEventCreate(&start);
	cudaEventCreate(&end);
	float gpu_time=0.0f;
	cudaDeviceSynchronize();
	cudaEventRecord(start);

	//// Invoke kernel
	const int block_size=32;
	const int block_num_x=C.m/block_size;
	const int block_num_y=C.n/block_size;


	#ifdef POORMAN
		Matrix_Multiplication_ATBA_Kernel_Poorman<<<dim3(block_num_x,block_num_y),dim3(block_size,block_size)>>>(A_on_dev.elements_on_dev,B_on_dev.elements_on_dev,C_on_dev.elements_on_dev,A_on_dev.m,A_on_dev.n);
	#endif

	#ifndef POORMAN
	////NOTICE: You do not have to use the block_size I specified here. You may customize the size of your grid and blocks for better performance.
	Matrix_Multiplication_ATBA_Kernel_Your_Version<<<dim3(block_num_y,block_num_x),dim3(block_size,block_size)>>>(A_on_dev.elements_on_dev,B_on_dev.elements_on_dev,C_on_dev.elements_on_dev,A_on_dev.m,A_on_dev.n);
	#endif
	

	cudaEventRecord(end);
	cudaEventSynchronize(end);
	cudaEventElapsedTime(&gpu_time,start,end);
	printf("\nGPU runtime for matrix multiplication ATBA: %.4f ms\n",gpu_time);
	cudaEventDestroy(start);
	cudaEventDestroy(end);

	//// Transfer data back to CPU
	C=C_on_dev;

	out<<"T2: "<<gpu_time<<endl;
}

__host__ void Test_Matrix_F_Norm_On_GPU(const Matrix& A, float& norm)
{
	//// Load A and B to device memory
	Matrix A_on_dev(A.m,A.n,false);
	A_on_dev=A;

	cudaEvent_t start,end;
	cudaEventCreate(&start);
	cudaEventCreate(&end);
	float gpu_time=0.0f;
	cudaDeviceSynchronize();
	cudaEventRecord(start);


	#ifdef POORMAN // atomic add

	//// Invoke kernel
	const int block_size=16;
	const int block_num_x=A.n/block_size;
	const int block_num_y=A.m/block_size;

	float *sum_dev = nullptr;
	cudaMalloc((void**)&sum_dev, sizeof(float));
	F_Norm_On_GPU_Lazy<<<dim3(block_num_x,block_num_y), dim3(block_size,block_size)>>>(A_on_dev.elements_on_dev, sum_dev);
	float *sum_host = (float *)malloc(4);
	cudaMemcpy(sum_host, sum_dev, sizeof(float), cudaMemcpyDeviceToHost);
	cudaFree(sum_dev);

	norm = sqrt(*sum_host);
	free(sum_host);

	#endif // ifdef

	#ifndef POORMAN // parallel reduction

	const int r1_blocks = A.m;
	const int r1_threads = A.n / 2;
	const int r2_threads = A.m / 2;
	float *B_dev = nullptr;
	cudaMalloc((void**)&B_dev, A.m * sizeof(float));
	F_Norm_On_GPU<<<r1_blocks, r1_threads, r1_threads*sizeof(float)>>>(A_on_dev.elements_on_dev, B_dev, true);
	F_Norm_On_GPU<<<1, r2_threads, r2_threads*sizeof(float)>>>(B_dev, B_dev, false);

	float result = 0;
	cudaMemcpy(&result,B_dev,sizeof(float),cudaMemcpyDeviceToHost);
	norm = sqrt(result);
	cudaFree(B_dev);
	#endif // ifndef
	
	cudaEventRecord(end);
	cudaEventSynchronize(end);
	cudaEventElapsedTime(&gpu_time,start,end);
	printf("\nGPU runtime for F norm: %.4f ms\n",gpu_time);
	cudaEventDestroy(start);
	cudaEventDestroy(end);

	out<<"T3: "<<gpu_time<<endl;
}

int main()
{
	if(name::team=="Team_X"){
		printf("\nPlease specify your team name and team member names in name::team and name::author to start.\n");
		return 0;
	}

	std::string file_name=name::team+"_competition_1_matrix.dat";
	out.open(file_name.c_str());

	if(out.fail()){
		printf("\ncannot open file %s to record results\n",file_name.c_str());
		return 0;
	}

	//////////////////////////////////////////////////////////////////////////
	////NOTICE: We may use a different set of parameters to evaluate your code.
	////So please test your functions with different size and initial values.
	//////////////////////////////////////////////////////////////////////////

	const int m=512;
	const int n=2048;
	const int p=1024;

	Matrix h_A(m,n);
	for(int i=0;i<m;i++){
		for(int j=0;j<n;j++){
			h_A(i,j) = 1;
		}
	}

	Matrix h_B(n,p);
	for(int i=0;i<n;i++){
		for(int j=0;j<p;j++){
			h_B(i,j) = 1;
		}
	}

	Matrix h_C(m,p);

	Matrix h_B2(m,m);
	for(int i=0;i<m;i++){
		for(int j=0;j<m;j++){
			h_B2(i,j) = 1;
		}
	}

	Matrix h_C2(n,n);

	Test_Matrix_Multiplication_AB_On_GPU(h_A,h_B,h_C);
	cout<<"AB result: "<<h_C(h_C.m/2,h_C.n/2)<<endl;
	out<<"R1: "<<h_C(h_C.m/2,h_C.n/2)<<endl;

	Test_Matrix_Multiplication_ATBA_On_GPU(h_A,h_B2,h_C2);
	cout<<"ATBA result: "<<h_C2(h_C2.m/3,h_C2.n/3)<<endl;
	out<<"R2: "<<h_C2(h_C2.m/3,h_C2.n/3)<<endl;

	float f_norm=0.f;
	Test_Matrix_F_Norm_On_GPU(h_A,f_norm);
	cout<<"F-norm result: "<<f_norm<<endl;
	out<<"R3: "<<f_norm<<endl;

	return 0;
}
