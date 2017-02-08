//
// Assignment 1: ParallelSine
// CSCI 415: Networking and Parallel Computation
// Spring 2017
// Name(s): Scott St. Amant, Micah Schmit
//
// Sine implementation derived from slides here: http://15418.courses.cs.cmu.edu/spring2016/lecture/basicarch


// standard imports
#include <stdio.h>
#include <math.h>
#include <iomanip>
#include <iostream>
#include <string>
#include <sys/time.h>

// problem size (vector length) N
static const int N = 12345678;

// Number of terms to use when approximating sine
static const int TERMS = 6;

//Max number of threads per block
const int threads_per_block = 256;

// kernel function (CPU - Do not modify)
void sine_serial(float *input, float *output)
{
  int i;

  for (i=0; i<N; i++) {
      float value = input[i]; 
      float numer = input[i] * input[i] * input[i]; 
      int denom = 6; // 3! 
      int sign = -1; 
      for (int j=1; j<=TERMS;j++) 
      { 
         value += sign * numer / denom; 
         numer *= input[i] * input[i]; 
         denom *= (2*j+2) * (2*j+3); 
         sign *= -1; 
      } 
      output[i] = value; 
    }
}

// kernel function (CUDA device)
// TODO: Implement your graphics kernel here. See assignment instructions for method information
__global__ void sine_parallel(float *input, float *output) {
  int block_id = blockIdx.x + gridDim.x * blockIdx.y;
	int thread_id = blockDim.x * block_id + threadIdx.x;
  printf("%d\n",thread_id);
   if(thread_id < N){
       float value = input[thread_id]; 
       float numer = input[thread_id] * input[thread_id] * input[thread_id]; 
       int denom = 6; // 3! 
       int sign = -1; 
       for (int j=1; j<=TERMS;j++) 
         { 
          value += sign * numer / denom; 
          numer *= input[thread_id] * input[thread_id]; 
          denom *= (2*j+2) * (2*j+3); 
          sign *= -1; 
         } 
       output[thread_id] = value;
       printf("Output: %d\n",output[thread_id]); 
  }



}
// BEGIN: timing and error checking routines (do not modify)

// Returns the current time in microseconds
long long start_timer() {
	struct timeval tv;
	gettimeofday(&tv, NULL);
	return tv.tv_sec * 1000000 + tv.tv_usec;
}


// Prints the time elapsed since the specified time
long long stop_timer(long long start_time, std::string name) {
	struct timeval tv;
	gettimeofday(&tv, NULL);
	long long end_time = tv.tv_sec * 1000000 + tv.tv_usec;
        std::cout << std::setprecision(5);	
	std::cout << name << ": " << ((float) (end_time - start_time)) / (1000 * 1000) << " sec\n";
	return end_time - start_time;
}

void checkErrors(const char label[])
{
  // we need to synchronise first to catch errors due to
  // asynchroneous operations that would otherwise
  // potentially go unnoticed

  cudaError_t err;

  err = cudaThreadSynchronize();
  if (err != cudaSuccess)
  {
    char *e = (char*) cudaGetErrorString(err);
    fprintf(stderr, "CUDA Error: %s (at %s)", e, label);
  }

  err = cudaGetLastError();
  if (err != cudaSuccess)
  {
    char *e = (char*) cudaGetErrorString(err);
    fprintf(stderr, "CUDA Error: %s (at %s)", e, label);
  }
}

// END: timing and error checking routines (do not modify)



int main (int argc, char **argv)
{
  //BEGIN: CPU implementation (do not modify)
  float *h_cpu_result = (float*)malloc(N*sizeof(float));
  float *h_input = (float*)malloc(N*sizeof(float));
  //Initialize data on CPU
  int i;
  for (i=0; i<N; i++)
  {
    h_input[i] = 0.1f * i;
  }

  //Execute and time the CPU version
  long long CPU_start_time = start_timer();
  sine_serial(h_input, h_cpu_result);
  long long CPU_time = stop_timer(CPU_start_time, "\nCPU Run Time");
  //END: CPU implementation (do not modify)


  //TODO: Prepare and run your kernel, make sure to copy your results back into h_gpu_result and display your timing results
  float *h_gpu_result = (float*)malloc(N*sizeof(float));

  int vector_size = N * sizeof(float);
	float *d_input, *d_output;
  cudaMalloc((void **) &d_input, vector_size);
	cudaMalloc((void **) &d_output, vector_size);
	// if (cudaMalloc((void **) &d_input, vector_size) != cudaSuccess) die("Error allocating GPU memory");
	// if (cudaMalloc((void **) &d_output, vector_size) != cudaSuccess) die("Error allocating GPU memory");
	
	// Transfer the input vectors to GPU memory
	cudaMemcpy(d_input, h_input, vector_size, cudaMemcpyHostToDevice);
  int num_blocks = (int) ((float) (N + threads_per_block - 1) / (float) threads_per_block);
	int max_blocks_per_dimension = 65535;
	int num_blocks_y = (int) ((float) (num_blocks + max_blocks_per_dimension - 1) / (float) max_blocks_per_dimension);
	int num_blocks_x = (int) ((float) (num_blocks + num_blocks_y - 1) / (float) num_blocks_y);
	dim3 grid_size(num_blocks_x, num_blocks_y, 1);
  //long long kernal_start_timer = start_timer();
  sine_parallel <<< grid_size, threads_per_block >>> (h_input, h_gpu_result);
  //stop_timer(kernal_start_timer, "\t Kernel execution");

  checkErrors((char*)'k');
  cudaMemcpy(h_gpu_result, d_output, vector_size, cudaMemcpyDeviceToHost);
  cudaFree(d_output);
  cudaFree(d_input);

  // Checking to make sure the CPU and GPU results match - Do not modify
  int errorCount = 0;
  for (i=0; i<N; i++)
  {
    if (abs(h_cpu_result[i]-h_gpu_result[i]) > 1e-6)
      errorCount = errorCount + 1;
  }
  if (errorCount > 0)
    printf("Result comparison failed.\n");
  else
    printf("Result comparison passed.\n");

  // Cleaning up memory
  free(h_input);
  free(h_cpu_result);
  free(h_gpu_result);
  return 0;
}






