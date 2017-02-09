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
	int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
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
  long long gpu_total_start = start_timer();  
  float *h_gpu_result = (float*)malloc(N*sizeof(float));

  int vector_size = N * sizeof(float);
	float *d_input;
  float *d_output;

  long long gpu_mem_start = start_timer();  
  cudaMalloc((void **) &d_input, vector_size);
	cudaMalloc((void **) &d_output, vector_size);
	stop_timer(gpu_mem_start, "\nGPU memory allocation");
	// Transfer the input vectors to GPU memory
	cudaMemcpy(d_input, h_input, vector_size, cudaMemcpyHostToDevice);
  long long kernal_start_timer = start_timer();
  sine_parallel<<<48226,threads_per_block>>>(d_input, d_output);
  stop_timer(kernal_start_timer, "\nKernel execution");

  long long gpu_mem_copy = start_timer();
  cudaMemcpy(h_gpu_result, d_output, vector_size, cudaMemcpyDeviceToHost);
  stop_timer(gpu_mem_copy, "\nCopying results from GPU");
  cudaThreadSynchronize();
  cudaFree(d_output);
  cudaFree(d_input);
  stop_timer(gpu_total_start,"\nTotal GPU Runtime");

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






