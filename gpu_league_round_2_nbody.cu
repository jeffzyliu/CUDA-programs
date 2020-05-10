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

			// if (i == 0) {
			// 	printf("idx: %d\n", j);
			// 	printf("dx: %f, dy: %f, dz: %f\n", rx, ry, rz);
            //     printf("px: %f, py: %f, pz: %f\n", pos_x[i], pos_y[i], pos_z[i]);
            //     printf("sx: %f, sy: %f, sz: %f\n", pos_x[j], pos_y[j], pos_z[j]);
			// 	printf("mass: %f\n", mass[j]);
			// }

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

__global__ void printer(double* pos_x, double* pos_y, double* pos_z, int length)
{
	for (int i = 0; i < length; i++) {
		printf("%d: %f, %f, %f\n", i, pos_x[i], pos_y[i], pos_z[i]);
	}
}



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
	// printf("%d\n", arr_idx);
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
        shift_arr_idx = ((blockIdx.x + blockShift)%gridDim.x) * blockDim.x  + threadIdx.x;
		// load memory into shared
        sharedX[threadIdx.x] = pos_x[shift_arr_idx];
        sharedY[threadIdx.x] = pos_y[shift_arr_idx];
        sharedZ[threadIdx.x] = pos_z[shift_arr_idx];
		sharedM[threadIdx.x] = mass[shift_arr_idx];

		__syncthreads();
		// if (blockIdx.x * blockDim.x + threadIdx.x == 0) {
		// 	printf("index %d blockshift %d threadidx %d: %f, %f, %f\n", blockIdx.x * blockDim.x + threadIdx.x, blockShift, threadIdx.x, pos_x[shift_arr_idx], pos_y[shift_arr_idx], pos_z[shift_arr_idx]);
		// }
		
		#pragma unroll 16
        for (int threadShift = 0; threadShift < blockDim.x; ++threadShift) {
            // don't calculate your own gravity
            if (blockShift == 0 && threadShift == 0) continue;
            // shifted shared memory index that wraps around
            shift_thread_idx = (threadIdx.x + threadShift) % blockDim.x;
            
            // distance
            dx = sharedX[shift_thread_idx] - px;
            dy = sharedY[shift_thread_idx] - py;
            dz = sharedZ[shift_thread_idx] - pz;
            // if (blockIdx.x * blockDim.x + threadIdx.x == 0) {
			// 	printf("idx: %d\n", shift_arr_idx + shift_thread_idx);
            //     printf("dx: %f, dy: %f, dz: %f\n", dx, dy, dz);
            //     printf("px: %f, py: %f, pz: %f\n", px, py, pz);
			// 	printf("sx: %f, sy: %f, sz: %f\n", sharedX[shift_thread_idx], sharedY[shift_thread_idx], sharedZ[shift_thread_idx]);
			// 	printf("mass: %f\n", sharedM[shift_thread_idx]);
			// 	// printf("shift_arr_idx: %d, shift_thread_idx: %d\n", shift_arr_idx, shift_thread_idx);
				
            // }
            // __syncthreads();
            // calculations
            dis_squared = dx*dx + dy*dy + dz*dz;
            one_over_dis_cube = 1.0 / pow(sqrt(dis_squared + epsilon_squared), 3);
            // increment acceleration for one round
            otherM = sharedM[shift_thread_idx];
			ax += otherM*dx*one_over_dis_cube;
			ay += otherM*dy*one_over_dis_cube;
            az += otherM*dz*one_over_dis_cube;
            __syncthreads();
        }
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

    // if (blockIdx.x * blockDim.x + threadIdx.x == n/2) {
    //     printf("vx: %f, vy: %f, vz: %f, px: %f, py: %f, pz: %f\n", vx, vy, vz, px, py, pz);
    //     printf("ax: %f, ay: %f, az: %f\n", ax, ay, az);
	// }
}


__global__ void N_Body_Simulation_GPU_Loop(volatile double* pos_x,volatile double* pos_y,volatile double* pos_z,		////position array
	volatile double* vel_x,volatile double* vel_y,volatile double* vel_z,		////velocity array
	// acceleraion array unnecessary since I'm calculating in regisers
	const volatile double* mass,								////mass array
	const int n,									////number of particles
	const double dt,								////timestep
	const double epsilon_squared,				////epsilon to avoid 0-denominator
	const int timesteps)					
{
	// prepare dynamic shared memory
	extern __shared__ double data[];
	volatile double *sharedX = &data[0];
	volatile double *sharedY = &data[blockDim.x];		
	volatile double *sharedZ = &data[2*blockDim.x];
	volatile double *sharedM = &data[3*blockDim.x];

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

	double ax; // acceleration
	double ay;
	double az;

	double vx = vel_x[arr_idx]; // velocity
	double vy = vel_y[arr_idx];
	double vz = vel_z[arr_idx];

	double px = pos_x[arr_idx]; // position
	double py = pos_y[arr_idx];
	double pz = pos_z[arr_idx];

	for (int i = 0; i < timesteps; ++i) { // each timestep
		ax = 0;
		ay = 0;
		az = 0;
		__syncthreads();

		// here we go, begin cycling through tiles
		for (int blockShift = 0; blockShift < gridDim.x; ++blockShift) {
			// re-index with respect to blockShift, use modulo to loop back around
			shift_arr_idx = ((blockIdx.x + blockShift)%gridDim.x) * blockDim.x  + threadIdx.x;
			// load memory into shared
			sharedX[threadIdx.x] = pos_x[shift_arr_idx];
			sharedY[threadIdx.x] = pos_y[shift_arr_idx];
			sharedZ[threadIdx.x] = pos_z[shift_arr_idx];
			sharedM[threadIdx.x] = mass[shift_arr_idx];
			__syncthreads();

			#pragma unroll
			for (int threadShift = 0; threadShift < blockDim.x; ++threadShift) {
				// don't calculate your own gravity
				if (blockShift == 0 && threadShift == 0) continue;
				// shifted shared memory index that wraps around
				shift_thread_idx = (threadIdx.x + threadShift) % blockDim.x;

				// distance
				dx = sharedX[shift_thread_idx] - px;
				dy = sharedY[shift_thread_idx] - py;
				dz = sharedZ[shift_thread_idx] - pz;
				// if (blockIdx.x * blockDim.x + threadIdx.x == 0 && blockShift == 21 && threadShift == 67) {
				//     // printf("dx: %f, dy: %f, dz: %f\n", dx, dy, dz);
				//     // printf("px: %f, py: %f, pz: %f\n", px, py, pz);
				//     // printf("sx: %f, sy: %f, sz: %f\n", sharedX[shift_thread_idx], sharedY[shift_thread_idx], sharedZ[shift_thread_idx]);
				// 	printf("shift_arr_idx: %d, shift_thread_idx: %d\n", shift_arr_idx, shift_thread_idx);
				// }
				// __syncthreads();
				// calculations
				dis_squared = dx*dx + dy*dy + dz*dz;
				one_over_dis_cube = 1.0 / pow(sqrt(dis_squared + epsilon_squared), 3);
				// increment acceleration for one round
				otherM = sharedM[shift_thread_idx];
				ax += otherM*dx*one_over_dis_cube;
				ay += otherM*dy*one_over_dis_cube;
				az += otherM*dz*one_over_dis_cube;
				__syncthreads();
			}
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
		vel_x[arr_idx] = vx;
		vel_y[arr_idx] = vy;
		vel_z[arr_idx] = vz;
		// pos
		pos_x[arr_idx] = px;
		pos_y[arr_idx] = py;
		pos_z[arr_idx] = pz;
		// if (blockIdx.x * blockDim.x + threadIdx.x == n/2) {
		//     printf("vx: %f, vy: %f, vz: %f, px: %f, py: %f, pz: %f\n", vx, vy, vz, px, py, pz);
		//     printf("ax: %f, ay: %f, az: %f\n", ax, ay, az);
		// }
		__threadfence_system();
	}
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

	vel_x=new double[particle_n];
	memset(vel_x,0x00,particle_n*sizeof(double));
	vel_y=new double[particle_n];
	memset(vel_y,0x00,particle_n*sizeof(double));
	vel_z=new double[particle_n];
	memset(vel_z,0x00,particle_n*sizeof(double));

	acl_x=new double[particle_n];
	memset(acl_x,0x00,particle_n*sizeof(double));
	acl_y=new double[particle_n];
	memset(acl_y,0x00,particle_n*sizeof(double));
	acl_z=new double[particle_n];
	memset(acl_z,0x00,particle_n*sizeof(double));

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
	cudaMemcpy(px_dev[0], pos_x, particle_n*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(px_dev[1], pos_x, particle_n*sizeof(double), cudaMemcpyHostToDevice);
    double *py_dev[2];
	cudaMalloc((void **)&py_dev[0], particle_n*sizeof(double));
	cudaMalloc((void **)&py_dev[1], particle_n*sizeof(double));
	cudaMemcpy(py_dev[0], pos_y, particle_n*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(py_dev[1], pos_y, particle_n*sizeof(double), cudaMemcpyHostToDevice);
    double *pz_dev[2];
    cudaMalloc((void **)&pz_dev[0], particle_n*sizeof(double));
	cudaMalloc((void **)&pz_dev[1], particle_n*sizeof(double));
	cudaMemcpy(pz_dev[0], pos_z, particle_n*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(pz_dev[1], pos_z, particle_n*sizeof(double), cudaMemcpyHostToDevice);

	double *vx_dev[2];
	cudaMalloc((void **)&vx_dev[0], particle_n*sizeof(double));
	cudaMalloc((void **)&vx_dev[1], particle_n*sizeof(double));
	cudaMemcpy(vx_dev[0], vel_x, particle_n*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(vx_dev[1], vel_x, particle_n*sizeof(double), cudaMemcpyHostToDevice);
    double *vy_dev[2];
	cudaMalloc((void **)&vy_dev[0], particle_n*sizeof(double));
	cudaMalloc((void **)&vy_dev[1], particle_n*sizeof(double));
	cudaMemcpy(vy_dev[0], vel_y, particle_n*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(vy_dev[1], vel_y, particle_n*sizeof(double), cudaMemcpyHostToDevice);
    double *vz_dev[2];
    cudaMalloc((void **)&vz_dev[0], particle_n*sizeof(double));
	cudaMalloc((void **)&vz_dev[1], particle_n*sizeof(double));
	cudaMemcpy(vz_dev[0], vel_z, particle_n*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(vz_dev[1], vel_z, particle_n*sizeof(double), cudaMemcpyHostToDevice);

    double *mass_dev;
    cudaMalloc((void **)&mass_dev, particle_n*sizeof(double));
    cudaMemcpy(mass_dev, mass, particle_n*sizeof(double), cudaMemcpyHostToDevice);
    
	#ifdef GPU
	int blocksize = max(64, particle_n/32);
    int gridsize = particle_n / blocksize;
	
	int src, dst;

	for(int i=0;i<time_step_num;i++){
		src = i % 2;
		dst = (src + 1) % 2;
		N_Body_Simulation_GPU<<<gridsize, blocksize, blocksize*4*sizeof(double)>>>(
			px_dev[src],py_dev[src],pz_dev[src],vx_dev[src],vy_dev[src],vz_dev[src],
			px_dev[dst],py_dev[dst],pz_dev[dst],vx_dev[dst],vy_dev[dst],vz_dev[dst],
			mass_dev,particle_n,dt,epsilon_squared);
		// cudaDeviceSynchronize();
		// cudaMemcpy(pos_x, px_dev[dst], particle_n*sizeof(double), cudaMemcpyDeviceToHost);
    	// cudaMemcpy(pos_y, py_dev[dst], particle_n*sizeof(double), cudaMemcpyDeviceToHost);
    	// cudaMemcpy(pos_z, pz_dev[dst], particle_n*sizeof(double), cudaMemcpyDeviceToHost);
        // cout<<"pos on timestep "<<fixed<<i<<": "<<pos_x[particle_n/2]<<", "<<pos_y[particle_n/2]<<", "<<pos_z[particle_n/2]<<endl;
    }
	#endif // GPU
	// N_Body_Simulation_GPU_Loop<<<gridsize, blocksize, blocksize*4*sizeof(double)>>>(px_dev,py_dev,pz_dev,vx_dev,vy_dev,vz_dev,mass_dev,particle_n,dt,epsilon_squared,time_step_num);

	cudaMemcpy(pos_x, px_dev[dst], particle_n*sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(pos_y, py_dev[dst], particle_n*sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(pos_z, pz_dev[dst], particle_n*sizeof(double), cudaMemcpyDeviceToHost);



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
