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
	std::string team="Team_X";
	std::string author_1="Name_1";
	std::string author_2="Name_2";
	std::string author_3="Name_3";	////optional
};

//////////////////////////////////////////////////////////////////////////
////TODO: Read the following three CPU implementations for Jacobi, Gauss-Seidel, and Red-Black Gauss-Seidel carefully
////and understand the steps for these numerical algorithms
//////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////
////These are the global variables that define the domain of the problem to solver
////You will need to use these parameters or macros in your GPU implementations
//////////////////////////////////////////////////////////////////////////

const int n=8;							////grid size, we will change this value to up to 256 to test your code
const int g=1;							////padding size
const int s=(n+2*g)*(n+2*g);			////array size
#define I(i,j) (i+g)*(n+2*g)+(j+g)		////2D coordinate -> array index
#define B(i,j) i<0||i>=n||j<0||j>=n		////check boundary
const bool verbose=true;				////set false to turn off print for x and residual
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
	int max_num=1000;		////max iteration number
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
	}while(residual>tolerance||iter_num==max_num);	

	x=xr;

	cout<<"Jacobi solver converges in "<<iter_num<<" iterations, with residual "<<residual<<endl;

	delete [] buf;
}

void Gauss_Seidel_Solver(double* x,const double* b)
{
	int iter_num=0;			////iteration number
	int max_num=1000;		////max iteration number
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
	}while(residual>tolerance||iter_num==max_num);	

	cout<<"Gauss-Seidel solver converges in "<<iter_num<<" iterations, with residual "<<residual<<endl;
}

void Red_Black_Gauss_Seidel_Solver(double* x,const double* b)
{
	int iter_num=0;			////iteration number
	int max_num=1000;		////max iteration number
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
	}while(residual>tolerance||iter_num==max_num);	

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

	//////////////////////////////////////////////////////////////////////////

	delete [] x;
	delete [] b;
}

//////////////////////////////////////////////////////////////////////////
////TODO 1: your GPU variables and functions start here

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
			b[I(i,j)]=4.0;		////set the values for the right-hand side
		}
	}
	for(int i=-1;i<=n;i++){
		for(int j=-1;j<=n;j++){
			if(B(i,j))
				x[I(i,j)]=(double)(i*i+j*j);	////set boundary condition for x
		}
	}

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

	cudaEventRecord(end);
	cudaEventSynchronize(end);
	cudaEventElapsedTime(&gpu_time,start,end);
	printf("\nGPU runtime: %.4f ms\n",gpu_time);
	cudaEventDestroy(start);
	cudaEventDestroy(end);
	//////////////////////////////////////////////////////////////////////////

	////output x
	if(verbose){
		cout<<"\n\nx for your GPU solver:\n";
		for(int i=0;i<n;i++){
			for(int j=0;j<n;j++){
				cout<<x[I(i,j)]<<", ";
			}
			cout<<std::endl;
		}	
	}

	////calculate residual
	double residual=0.0;
	for(int i=0;i<n;i++){
		for(int j=0;j<n;j++){
			residual+=pow(4.0*x[I(i,j)]-x[I(i-1,j)]-x[I(i+1,j)]-x[I(i,j-1)]-x[I(i,j+1)]-b[I(i,j)],2);
		}
	}
	cout<<"\n\nresidual for your GPU solver: "<<residual<<endl;

	out<<"R0: "<<residual<<endl;
	out<<"T1: "<<gpu_time<<endl;

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

	Test_CPU_Solvers();	////You may comment out this line to run your GPU solver only
	Test_GPU_Solver();	////Test function for your own GPU implementation

	return 0;
}

