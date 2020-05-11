//////////////////////////////////////////////////////////////////////////
////This is the code implementation for GPU Premier League Round 2: n-body simulation
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

//////////////////////////////////////////////////////////////////////////
////Here is a sample function implemented on CPU for n-body simulation.

__host__ void N_Body_Simulation_CPU_Poorman(double* pos_x,double* pos_y,double* pos_z,		////position array
											double* vel_x,double* vel_y,double* vel_z,		////velocity array
											double* acl_x,double* acl_y,double* acl_z,		////acceleration array
											const double* mass,								////mass array
											const int n,									////number of particles
											const double dt,								////timestep
											const double epsilon_squared)					////epsilon to avoid 0-denominator
{		
	////Step 1: set particle accelerations to be zero
	memset(acl_x,0x00,sizeof(double)*n);
	memset(acl_y,0x00,sizeof(double)*n);
	memset(acl_z,0x00,sizeof(double)*n);

	////Step 2: traverse all particle pairs and accumulate gravitational forces for each particle from pairwise interactions
	for(int i=0;i<n;i++){
		for(int j=0;j<n;j++){
			////skip calculating force for itself
			if(i==j) continue;

			////r_ij=x_j-x_i
			double rx=pos_x[j]-pos_x[i];
			double ry=pos_y[j]-pos_y[i];
			double rz=pos_z[j]-pos_z[i];

			////a_ij=m_j*r_ij/(r+epsilon)^3, 
			////noticing that we ignore the gravitational coefficient (assuming G=1)
			double dis_squared=rx*rx+ry*ry+rz*rz;
			double one_over_dis_cube=1.0/pow(sqrt(dis_squared+epsilon_squared),3);
			double ax=mass[j]*rx*one_over_dis_cube;
			double ay=mass[j]*ry*one_over_dis_cube;
			double az=mass[j]*rz*one_over_dis_cube;

			////accumulate the force to the particle
			acl_x[i]+=ax;
			acl_y[i]+=ay;
			acl_z[i]+=az;
		}
	}

	////Step 3: explicit time integration to update the velocity and position of each particle
	for(int i=0;i<n;i++){
		////v_{t+1}=v_{t}+a_{t}*dt
		vel_x[i]+=acl_x[i]*dt;
		vel_y[i]+=acl_y[i]*dt;
		vel_z[i]+=acl_z[i]*dt;

		////x_{t+1}=x_{t}+v_{t}*dt
		pos_x[i]+=vel_x[i]*dt;
		pos_y[i]+=vel_y[i]*dt;
		pos_z[i]+=vel_z[i]*dt;
	}
}


//////////////////////////////////////////////////////////////////////////


__global__ void N_Body_Simulation_GPU(const double* pos_x, const double* pos_y, const double* pos_z,		////position array
									const double* vel_x, const double* vel_y, const double* vel_z,		////velocity array
									double* pos_x_dst, double* pos_y_dst, double* pos_z_dst,		////destination
                                    double* vel_x_dst, double* vel_y_dst, double* vel_z_dst,	    ////destination
                                    // acceleraion array unnecessary since I'm calculating in regisers
                                    const double* mass,								////mass array
                                    const int n,									////number of particles
                                    const double dt,								////timestep
                                    const double epsilon_squared)					////epsilon to avoid 0-denominator
{	
    // prepare dynamic shared memory
    extern __shared__ double data[];
    double *sharedX = &data[0];
    double *sharedY = &data[blockDim.x];
    double *sharedZ = &data[2*blockDim.x];
    double *sharedM = &data[3*blockDim.x];
	
    // prepare registers
	int arr_idx = blockIdx.x*blockDim.x + threadIdx.x;

    int shift_arr_idx;
    int shift_thread_idx;

    double dx; // save calculation time by saving these in registers
    double dy;
    double dz;

    double dis_squared; // registers for calculations
    double otherM;
    double one_over_dis_cube;

    double ax = 0; // acceleration
    double ay = 0;
    double az = 0;

    double vx = vel_x[arr_idx]; // velocity
    double vy = vel_y[arr_idx];
    double vz = vel_z[arr_idx];
    
    double px = pos_x[arr_idx]; // position
    double py = pos_y[arr_idx];
    double pz = pos_z[arr_idx];

    // here we go, begin cycling through tiles
    for (int blockShift = 0; blockShift < gridDim.x; ++blockShift) {
        // re-index with respect to blockShift, use modulo to loop back around
        shift_arr_idx = ((blockIdx.x + blockShift) & (gridDim.x-1)) * blockDim.x  + threadIdx.x;
		// load memory into shared
		// if (blockShift == 0) {
		// 	sharedX[threadIdx.x] = px;
		// 	sharedY[threadIdx.x] = py;
		// 	sharedZ[threadIdx.x] = pz;
		// 	sharedM[threadIdx.x] = mass[shift_arr_idx];
		// } else {
			sharedX[threadIdx.x] = pos_x[shift_arr_idx];
			sharedY[threadIdx.x] = pos_y[shift_arr_idx];
			sharedZ[threadIdx.x] = pos_z[shift_arr_idx];
			sharedM[threadIdx.x] = mass[shift_arr_idx];
		// }

		__syncthreads();
	
		#pragma unroll 64
        for (int threadShift = 0; threadShift < blockDim.x; ++threadShift) {
            // don't calculate your own gravity
			if (blockShift == 0 && threadShift == 0) continue;
			// if (blockShift == 0 && threadShift == threadIdx.x) continue;
            // shifted shared memory index that wraps around
            shift_thread_idx = (threadIdx.x + threadShift) & (blockDim.x-1);
            
            // // distance
            dx = sharedX[shift_thread_idx] - px;
            dy = sharedY[shift_thread_idx] - py;
			dz = sharedZ[shift_thread_idx] - pz;
			// dx = sharedX[threadShift] - px;
            // dy = sharedY[threadShift] - py;
			// dz = sharedZ[threadShift] - pz;
			// dx = 1;
			// dy = 1;
			// dz = 1;
            
            // calculations
            dis_squared = dx*dx + dy*dy + dz*dz;
			one_over_dis_cube = 1.0 / pow(sqrt(dis_squared + epsilon_squared), 3);
			// dis_squared = 2;
			// one_over_dis_cube = 2;
            // increment acceleration for one round
			otherM = sharedM[shift_thread_idx];
			// otherM = 100;
			// otherM = sharedM[threadShift];
			ax += otherM*dx*one_over_dis_cube;
			ay += otherM*dy*one_over_dis_cube;
            az += otherM*dz*one_over_dis_cube;
		}
		__syncthreads();
    }
    // all net forces and net accelerations done, now integrate velocity
    vx += ax * dt; 
    vy += ay * dt;
    vz += az * dt;
    // integrate position
    px += vx * dt; 
    py += vy * dt;
    pz += vz * dt;

    // write velocity to global memory
    vel_x_dst[arr_idx] = vx;
    vel_y_dst[arr_idx] = vy;
    vel_z_dst[arr_idx] = vz;
    // pos
    pos_x_dst[arr_idx] = px;
    pos_y_dst[arr_idx] = py;
    pos_z_dst[arr_idx] = pz;
}




__global__ void N_Body_Simulation_GPU_Unrolled(const double* pos_x, const double* pos_y, const double* pos_z,		////position array
	const double* vel_x, const double* vel_y, const double* vel_z,		////velocity array
	double* pos_x_dst, double* pos_y_dst, double* pos_z_dst,		////destination
	double* vel_x_dst, double* vel_y_dst, double* vel_z_dst,	    ////destination
	// acceleraion array unnecessary since I'm calculating in regisers
	const double* mass,								////mass array
	const int n,									////number of particles
	const double dt,								////timestep
	const double epsilon_squared)					////epsilon to avoid 0-denominator
{	
	// prepare dynamic shared memory
	extern __shared__ double data[];
	double *sharedX = &data[0];
	double *sharedY = &data[blockDim.x*2];
	double *sharedZ = &data[2*blockDim.x*2];
	double *sharedM = &data[3*blockDim.x*2];

	// prepare registers
	int arr_idx = blockIdx.x*2*blockDim.x + 2*threadIdx.x;

	int shift_arr_idx;
	int shift_thread_idx;

	double dx0; // save calculation time by saving these in registers
	double dy0;
	double dz0;

	double dis_squared0; // registers for calculations
	double otherM0;
	double one_over_dis_cube0;

	double ax0 = 0; // acceleration
	double ay0 = 0;
	double az0 = 0;

	double vx0 = vel_x[arr_idx]; // velocity
	double vy0 = vel_y[arr_idx];
	double vz0 = vel_z[arr_idx];

	double px0 = pos_x[arr_idx]; // position
	double py0 = pos_y[arr_idx];
	double pz0 = pos_z[arr_idx];

	double dx1; // save calculation time by saving these in registers
	double dy1;
	double dz1;

	double dis_squared1; // registers for calculations
	double otherM1;
	double one_over_dis_cube1;

	double ax1 = 0; // acceleration
	double ay1 = 0;
	double az1 = 0;

	double vx1 = vel_x[arr_idx + 1]; // velocity
	double vy1 = vel_y[arr_idx + 1];
	double vz1 = vel_z[arr_idx + 1];

	double px1 = pos_x[arr_idx + 1]; // position
	double py1 = pos_y[arr_idx + 1];
	double pz1 = pos_z[arr_idx + 1];


	// here we go, begin cycling through tiles
	for (int blockShift = 0; blockShift < gridDim.x; ++blockShift) {
		// re-index with respect to blockShift, use modulo to loop back around
		shift_arr_idx = ((blockIdx.x + blockShift)&(gridDim.x-1)) *2*blockDim.x + 2*threadIdx.x;
		// load memory into shared
		sharedX[2*threadIdx.x] = pos_x[shift_arr_idx];
		sharedY[2*threadIdx.x] = pos_y[shift_arr_idx];
		sharedZ[2*threadIdx.x] = pos_z[shift_arr_idx];
		sharedM[2*threadIdx.x] = mass[shift_arr_idx];

		sharedX[2*threadIdx.x + 1] = pos_x[shift_arr_idx + 1];
		sharedY[2*threadIdx.x + 1] = pos_y[shift_arr_idx + 1];
		sharedZ[2*threadIdx.x + 1] = pos_z[shift_arr_idx + 1];
		sharedM[2*threadIdx.x + 1] = mass[shift_arr_idx + 1];

		__syncthreads();

		#pragma unroll 32
        for (int threadShift = 0; threadShift < blockDim.x*2; threadShift += 2) {
            // don't calculate your own gravity
            if (blockShift == 0 && threadShift == 0) continue;
            // shifted shared memory index that wraps around
            shift_thread_idx = (2*threadIdx.x + threadShift) & (2*blockDim.x-1);
    
            // distance
            dx0 = sharedX[shift_thread_idx] - px0;
            dy0 = sharedY[shift_thread_idx] - py0;
			dz0 = sharedZ[shift_thread_idx] - pz0;
			
			dx1 = sharedX[shift_thread_idx + 1] - px1;
            dy1 = sharedY[shift_thread_idx + 1] - py1;
            dz1 = sharedZ[shift_thread_idx + 1] - pz1;
            
            // calculations
            dis_squared0 = dx0*dx0 + dy0*dy0 + dz0*dz0;
            one_over_dis_cube0 = 1.0 / pow(sqrt(dis_squared0 + epsilon_squared), 3);
            // increment acceleration for one round
            otherM0 = sharedM[shift_thread_idx];
			ax0 += otherM0*dx0*one_over_dis_cube0;
			ay0 += otherM0*dy0*one_over_dis_cube0;
			az0 += otherM0*dz0*one_over_dis_cube0;

			// calculations
            dis_squared1 = dx1*dx1 + dy1*dy1 + dz1*dz1;
            one_over_dis_cube1 = 1.0 / pow(sqrt(dis_squared1 + epsilon_squared), 3);
            // increment acceleration for one round
            otherM1 = sharedM[shift_thread_idx + 1];
			ax1 += otherM1*dx1*one_over_dis_cube1;
			ay1 += otherM1*dy1*one_over_dis_cube1;
			az1 += otherM1*dz1*one_over_dis_cube1;
		}
		__syncthreads();
	}
	// all net forces and net accelerations done, now integrate velocity
	vx0 += ax0 * dt; 
	vy0 += ay0 * dt;
	vz0 += az0 * dt;
	// integrate position
	px0 += vx0 * dt; 
	py0 += vy0 * dt;
	pz0 += vz0 * dt;

	// all net forces and net accelerations done, now integrate velocity
	vx1 += ax1 * dt; 
	vy1 += ay1 * dt;
	vz1 += az1 * dt;
	// integrate position
	px1 += vx1 * dt; 
	py1 += vy1 * dt;
	pz1 += vz1 * dt;

	// write velocity to global memory
	vel_x_dst[arr_idx] = vx0;
	vel_y_dst[arr_idx] = vy0;
	vel_z_dst[arr_idx] = vz0;
	// pos
	pos_x_dst[arr_idx] = px0;
	pos_y_dst[arr_idx] = py0;
	pos_z_dst[arr_idx] = pz0;
	// write velocity to global memory
	vel_x_dst[arr_idx + 1] = vx1;
	vel_y_dst[arr_idx + 1] = vy1;
	vel_z_dst[arr_idx + 1] = vz1;
	// pos
	pos_x_dst[arr_idx + 1] = px1;
	pos_y_dst[arr_idx + 1] = py1;
	pos_z_dst[arr_idx + 1] = pz1;
}


////Your implementations end here
//////////////////////////////////////////////////////////////////////////


//////////////////////////////////////////////////////////////////////////
////Test function for n-body simulator
ofstream out;

//////////////////////////////////////////////////////////////////////////
////Please do not change the values below
const double dt=0.001;							////time step
const int time_step_num=10;						////number of time steps
const double epsilon=1e-2;						////epsilon added in the denominator to avoid 0-division when calculating the gravitational force
const double epsilon_squared=epsilon*epsilon;	////epsilon squared

////We use grid_size=4 to help you debug your code, change it to a bigger number (e.g., 16, 32, etc.) to test the performance of your GPU code
const unsigned int grid_size=16;					////assuming particles are initialized on a background grid
const unsigned int particle_n=pow(grid_size,3);	////assuming each grid cell has one particle at the beginning

__host__ void Test_N_Body_Simulation()
{
	////initialize position, velocity, acceleration, and mass
	
	double* pos_x=new double[particle_n];
	double* pos_y=new double[particle_n];
	double* pos_z=new double[particle_n];
	////initialize particle positions as the cell centers on a background grid
	double dx=1.0/(double)grid_size;
	for(unsigned int k=0;k<grid_size;k++){
		for(unsigned int j=0;j<grid_size;j++){
			for(unsigned int i=0;i<grid_size;i++){
				unsigned int index=k*grid_size*grid_size+j*grid_size+i;
				pos_x[index]=dx*(double)i;
				pos_y[index]=dx*(double)j;
                pos_z[index]=dx*(double)k;
			}
		}
	}

	double* vel_x=new double[particle_n];
	memset(vel_x,0x00,particle_n*sizeof(double));
	double* vel_y=new double[particle_n];
	memset(vel_y,0x00,particle_n*sizeof(double));
	double* vel_z=new double[particle_n];
	memset(vel_z,0x00,particle_n*sizeof(double));

	double* acl_x=new double[particle_n];
	memset(acl_x,0x00,particle_n*sizeof(double));
	double* acl_y=new double[particle_n];
	memset(acl_y,0x00,particle_n*sizeof(double));
	double* acl_z=new double[particle_n];
	memset(acl_z,0x00,particle_n*sizeof(double));

	double* mass=new double[particle_n];
	for(int i=0;i<particle_n;i++){
		mass[i]=100.0;
	}
	cout.precision(6);
	#ifdef CPU
	//////////////////////////////////////////////////////////////////////////
	// Default implementation: n-body simulation on CPU
	// Comment the CPU implementation out when you test large-scale examples
	auto cpu_start=chrono::system_clock::now();
	cout<<"Total number of particles: "<<particle_n<<endl;
    cout<<"Tracking the motion of particle "<<particle_n/2<<endl;
	for(int i=0;i<time_step_num;i++){
		N_Body_Simulation_CPU_Poorman(pos_x,pos_y,pos_z,vel_x,vel_y,vel_z,acl_x,acl_y,acl_z,mass,particle_n,dt,epsilon_squared);
		cout<<"pos on timestep "<<fixed<<i<<": "<<pos_x[particle_n/2]<<", "<<pos_y[particle_n/2]<<", "<<pos_z[particle_n/2]<<endl;
        // cout<<"vel on timestep "<<i<<": "<<vel_x[particle_n/2]<<", "<<vel_y[particle_n/2]<<", "<<vel_z[particle_n/2]<<endl;
        // cout<<"acc on timestep "<<i<<": "<<acl_x[particle_n/2]<<", "<<acl_y[particle_n/2]<<", "<<acl_z[particle_n/2]<<endl;
    }
    // cout<<"pos on timestep "<<fixed<<time_step_num-1<<": "<<pos_x[particle_n/2]<<", "<<pos_y[particle_n/2]<<", "<<pos_z[particle_n/2]<<endl;

	auto cpu_end=chrono::system_clock::now();
	chrono::duration<double> cpu_time=cpu_end-cpu_start;
	cout<<"CPU runtime: "<<cpu_time.count()*1000.<<" ms."<<endl;
	#endif // CPU
	//////////////////////////////////////////////////////////////////////////


	// resetting starting positions before the timer begins
	pos_x=new double[particle_n];
	pos_y=new double[particle_n];
	pos_z=new double[particle_n];
	// cudaMallocHost((void **)&pos_x, particle_n*sizeof(double));
	// cudaMallocHost((void **)&pos_y, particle_n*sizeof(double));
	// cudaMallocHost((void **)&pos_z, particle_n*sizeof(double));
    for(unsigned int k=0;k<grid_size;k++){
		for(unsigned int j=0;j<grid_size;j++){
			for(unsigned int i=0;i<grid_size;i++){
				unsigned int index=k*grid_size*grid_size+j*grid_size+i;
				pos_x[index]=dx*(double)i;
				pos_y[index]=dx*(double)j;
				pos_z[index]=dx*(double)k;
				// cout << index << " " << pos_x[index] << endl;
			}
		}
    }

    // cout << pos_x[1] << " " << pos_y[1] << " " << pos_z[1] << endl;

	// cudaMallocHost((void **)&vel_x, particle_n*sizeof(double));
	// cudaMallocHost((void **)&vel_y, particle_n*sizeof(double));
	// cudaMallocHost((void **)&vel_z, particle_n*sizeof(double));
	vel_x=new double[particle_n];
	memset(vel_x,0x00,particle_n*sizeof(double));
	vel_y=new double[particle_n];
	memset(vel_y,0x00,particle_n*sizeof(double));
	vel_z=new double[particle_n];
	memset(vel_z,0x00,particle_n*sizeof(double));

	// // acl_x=new double[particle_n];
	// memset(acl_x,0x00,particle_n*sizeof(double));
	// // acl_y=new double[particle_n];
	// memset(acl_y,0x00,particle_n*sizeof(double));
	// // acl_z=new double[particle_n];
	// memset(acl_z,0x00,particle_n*sizeof(double));

	mass=new double[particle_n];
	// cudaMallocHost((void **)&mass, particle_n*sizeof(double));
	for(int i=0;i<particle_n;i++){
		mass[i]=100.0;
	}

	//////////////////////////////////////////////////////////////////////////
	////Your implementation: n-body simulator on GPU
	cudaEvent_t start,end;
	cudaEventCreate(&start);
	cudaEventCreate(&end);
	float gpu_time=0.0f;
	cudaDeviceSynchronize();
	cudaEventRecord(start);

	//////////////////////////////////////////////////////////////////////////
	////TODO 2: Your GPU functions are called here
	////Requirement: You need to copy data from the CPU arrays, conduct computations on the GPU, and copy the values back from GPU to CPU
	////The final positions should be stored in the same place as the CPU n-body function, i.e., pos_x, pos_y, pos_z
	////The correctness of your simulation will be evaluated by comparing the results (positions) with the results calculated by the default CPU implementations

	// double-buffering
    double *px_dev[2];
	cudaMalloc((void **)&px_dev[0], particle_n*sizeof(double));
	cudaMalloc((void **)&px_dev[1], particle_n*sizeof(double));
	cudaMemcpyAsync(px_dev[0], pos_x, particle_n*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpyAsync(px_dev[1], pos_x, particle_n*sizeof(double), cudaMemcpyHostToDevice);
    double *py_dev[2];
	cudaMalloc((void **)&py_dev[0], particle_n*sizeof(double));
	cudaMalloc((void **)&py_dev[1], particle_n*sizeof(double));
	cudaMemcpyAsync(py_dev[0], pos_y, particle_n*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpyAsync(py_dev[1], pos_y, particle_n*sizeof(double), cudaMemcpyHostToDevice);
    double *pz_dev[2];
    cudaMalloc((void **)&pz_dev[0], particle_n*sizeof(double));
	cudaMalloc((void **)&pz_dev[1], particle_n*sizeof(double));
	cudaMemcpyAsync(pz_dev[0], pos_z, particle_n*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpyAsync(pz_dev[1], pos_z, particle_n*sizeof(double), cudaMemcpyHostToDevice);

	double *vx_dev[2];
	cudaMalloc((void **)&vx_dev[0], particle_n*sizeof(double));
	cudaMalloc((void **)&vx_dev[1], particle_n*sizeof(double));
	cudaMemcpyAsync(vx_dev[0], vel_x, particle_n*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpyAsync(vx_dev[1], vel_x, particle_n*sizeof(double), cudaMemcpyHostToDevice);
    double *vy_dev[2];
	cudaMalloc((void **)&vy_dev[0], particle_n*sizeof(double));
	cudaMalloc((void **)&vy_dev[1], particle_n*sizeof(double));
	cudaMemcpyAsync(vy_dev[0], vel_y, particle_n*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpyAsync(vy_dev[1], vel_y, particle_n*sizeof(double), cudaMemcpyHostToDevice);
    double *vz_dev[2];
    cudaMalloc((void **)&vz_dev[0], particle_n*sizeof(double));
	cudaMalloc((void **)&vz_dev[1], particle_n*sizeof(double));
	cudaMemcpyAsync(vz_dev[0], vel_z, particle_n*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpyAsync(vz_dev[1], vel_z, particle_n*sizeof(double), cudaMemcpyHostToDevice);

    double *mass_dev;
    cudaMalloc((void **)&mass_dev, particle_n*sizeof(double));
	cudaMemcpyAsync(mass_dev, mass, particle_n*sizeof(double), cudaMemcpyHostToDevice);
	// cudaDeviceSynchronize();
    
	#ifdef GPU
	int blocksize = min(max(64, particle_n/32), 512);
	int gridsize = particle_n / blocksize;
	// blocksize /= 2;
	
	int src, dst;

	
	for (int i = 0;i < time_step_num; i++) {
		src = i & 1;
		dst = (src + 1) & 1;
		N_Body_Simulation_GPU<<<gridsize, blocksize, blocksize*4*sizeof(double)>>>(
			px_dev[src],py_dev[src],pz_dev[src],vx_dev[src],vy_dev[src],vz_dev[src],
			px_dev[dst],py_dev[dst],pz_dev[dst],vx_dev[dst],vy_dev[dst],vz_dev[dst],
			mass_dev,particle_n,dt,epsilon_squared);
		// N_Body_Simulation_GPU_Unrolled<<<gridsize, blocksize, blocksize*8*sizeof(double)>>>(
		// 	px_dev[src],py_dev[src],pz_dev[src],vx_dev[src],vy_dev[src],vz_dev[src],
		// 	px_dev[dst],py_dev[dst],pz_dev[dst],vx_dev[dst],vy_dev[dst],vz_dev[dst],
		// // 	mass_dev,particle_n,dt,epsilon_squared);
		// cudaMemcpy(pos_x, px_dev[dst], particle_n*sizeof(double), cudaMemcpyDeviceToHost);
    	// cudaMemcpy(pos_y, py_dev[dst], particle_n*sizeof(double), cudaMemcpyDeviceToHost);
    	// cudaMemcpy(pos_z, pz_dev[dst], particle_n*sizeof(double), cudaMemcpyDeviceToHost);
        // cout<<"pos on timestep "<<fixed<<i<<": "<<pos_x[particle_n/2]<<", "<<pos_y[particle_n/2]<<", "<<pos_z[particle_n/2]<<endl;
    }
	#endif // GPU

	cudaMemcpyAsync(pos_x, px_dev[dst], particle_n*sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpyAsync(pos_y, py_dev[dst], particle_n*sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpyAsync(pos_z, pz_dev[dst], particle_n*sizeof(double), cudaMemcpyDeviceToHost);
	// cudaDeviceSynchronize();
	//////////////////////////////////////////////////////////////////////////

	cudaEventRecord(end);
	cudaEventSynchronize(end);
	cudaEventElapsedTime(&gpu_time,start,end);
	printf("\nGPU runtime: %.4f ms\n",gpu_time);
	cudaEventDestroy(start);
	cudaEventDestroy(end);

	//////////////////////////////////////////////////////////////////////////

	cout<<"R0: "<<pos_x[particle_n/2]<<" " <<pos_y[particle_n/2]<<" " <<pos_z[particle_n/2]<<endl;
	out<<"T1: "<<gpu_time<<endl;
}

int main()
{
	if(name::team=="Team_X"){
		printf("\nPlease specify your team name and team member names in name::team and name::author to start.\n");
		return 0;
	}

	std::string file_name=name::team+"_competition_2_nbody.dat";
	out.open(file_name.c_str());
	
	if(out.fail()){
		printf("\ncannot open file %s to record results\n",file_name.c_str());
		return 0;
	}

	Test_N_Body_Simulation();

	return 0;
}
