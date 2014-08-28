/*
 * This program is an extended version of "Jens Lang"'s work.  
 * Lang, J., & Rünger, G. (2013). High-Resolution power profiling of GPU
 * functions using low-resolution measurement. In Euro-Par 2013 Parallel
 * Processing (pp. 801-812). Springer Berlin Heidelberg.
 */

#include <papi.h>
#include <nvml.h>
#include <stdio.h>
#include <unistd.h>
#include <pthread.h>

#include "high_resolution_power.h"

#define CEIL(x,y) 1 + (((x) - 1) / (y))

int sleep_time_after_kernel_call = 0; //in seconds
int reps = 1; //getenv
unsigned int device_id = 0;
long long time_kernel_start_ex = 0;

long long kernel_execution_time = 0; //nano seconds

nvmlReturn_t nvml_result;
nvmlDevice_t nvml_device;


void set_reps(int nvml_reps) {
  reps = nvml_reps;
} 

void handle_papi_error(int retval)
{
	PAPI_perror((char *) "Fehler");
	printf((char *) "PAPI error %d: %s\n", retval, PAPI_strerror(retval));
}

int nvml_finalize(nvmlReturn_t result) {
  result = nvmlShutdown();
  if (NVML_SUCCESS != result) {
    printf("Failed to shutdown NVML: %s\n", nvmlErrorString(result));
    return -1;
  }
  return 1;
}

unsigned int synchronizeTime(long long & cpu_time, nvmlDevice_t nvml_device, unsigned int &temperature, int &pstate)
{
	unsigned int gpu_value = 0;
	unsigned int last_gpu_value = 0;


  int device_id = 0;
  long long starttt = PAPI_get_real_nsec();
	nvmlReturn_t nvml_result;
	nvml_result = nvmlDeviceGetPowerUsage(nvml_device, &gpu_value);
	if (nvml_result != NVML_SUCCESS) printf("NVML error: %s.\n", nvmlErrorString(nvml_result));
	last_gpu_value = gpu_value;

	while (gpu_value == last_gpu_value)
	{
		last_gpu_value = gpu_value;
		nvml_result = nvmlDeviceGetPowerUsage(nvml_device, &gpu_value);
		if (nvml_result != NVML_SUCCESS) printf("NVML error: %d.\n", nvmlErrorString(nvml_result));
	}

	cpu_time = PAPI_get_real_nsec();
  
///////////////////////////////////////
  nvml_result = nvmlDeviceGetTemperature(nvml_device, NVML_TEMPERATURE_GPU, &temperature);
  if (NVML_SUCCESS != nvml_result)
  {
    printf("Failed to get temperature of the device %i: %s\n", device_id, nvmlErrorString(nvml_result));
  }
  nvmlPstates_t p_state;
  nvml_result = nvmlDeviceGetPerformanceState(nvml_device, &p_state);
  if (NVML_SUCCESS != nvml_result)
  {
    printf("Failed to get perf state of the device %i: %s\n", device_id, nvmlErrorString(nvml_result));
  }

  pstate = p_state;
//////////////////////////////////////

	return gpu_value;
}


int nvml_init() { 
  char *nvml_reps = getenv("NVML_REPS");
  if (nvml_reps != NULL) {
    reps = atoi(nvml_reps);
  }

  char *nvml_sleep_time = getenv("NVML_SLEEP_TIME");
  if (nvml_sleep_time != NULL) {
    sleep_time_after_kernel_call = atoi(nvml_sleep_time);
  }

	// Initialize the PAPI library
	int retval = PAPI_NULL;
  retval = PAPI_library_init(PAPI_VER_CURRENT);
	if (retval != PAPI_VER_CURRENT) {
		printf("PAPI library init error: %d!\n", retval);
    handle_papi_error(retval);
		return -1;
	} 

  //initializing nvml
  nvml_result = nvmlInit();
  if (NVML_SUCCESS != nvml_result)
  {
    printf("Failed to initialize NVML: %s\n", nvmlErrorString(nvml_result));
    return -1;
  }

	nvml_result = nvmlDeviceGetHandleByIndex(device_id, &nvml_device);
  if (NVML_SUCCESS != nvml_result)
  {
    printf("Failed to get handle for device %i: %s\n", device_id, nvmlErrorString(nvml_result));
    nvml_result = nvmlShutdown();
    nvml_finalize(nvml_result);
    return -1;
  }
  
  printf("#defined #NUM_REPS: %d COOLDOWN_TIME: %ds\n", reps, sleep_time_after_kernel_call);

  return 0;
}


int power_profile(void *(*call_cuda_kernel)(void*), void *(*reset_kenel_data)(void*)) {
  struct timespec sleep_time;
  sleep_time.tv_sec = sleep_time_after_kernel_call;
  sleep_time.tv_nsec = 0;

  pthread_t pthread;

	reps = CEIL(7E9, kernel_execution_time);
	
	printf("#RREPS: %d\n", reps);

  // perform actual energy measurement
	for (int n_ex = 0; n_ex < reps; n_ex++)
	{
    reset_kenel_data(NULL);
		// wait a random time // this functionality is suspended since we just
		// interested in power after 5seconds
	/*	sleep_time.tv_nsec = rand()%(DELTA_T) + DELTA_T;
		nanosleep(&sleep_time, NULL);
*/
		// call GPU kernel
		bool gpu_power_before_printed = false; // indicates wether the first power measurement (before the start of the GPU kernel) has already been printed out
		long long time_start_kernel = PAPI_get_real_nsec();
		long long time_current = time_start_kernel;
		if (n_ex == 0) {
			time_kernel_start_ex = time_start_kernel;
		}

    int rc = pthread_create(&pthread, NULL, call_cuda_kernel, NULL);
    if (rc){
     printf("ERROR; return code from pthread_create() is %d\n", rc);
     exit(-1);
    }

		unsigned int gpu_power_before;
		nvml_result = nvmlDeviceGetPowerUsage(nvml_device, &gpu_power_before);
		if (nvml_result != NVML_SUCCESS) {
      printf("NVML error: %s.\n", nvmlErrorString(nvml_result));
    }

    
		// continually retrieve power values of the GPU
		while (time_current < time_start_kernel + kernel_execution_time)
		{
			long long time_running_update;
      unsigned int temperature;
      int pstate;
			unsigned int gpu_power = synchronizeTime(time_running_update, nvml_device, temperature, pstate);

			if (!gpu_power_before_printed)
			{
				printf("%.5f ms\t%.5f W\t%u C\t%d\t%.5f ms\n", (time_running_update - time_start_kernel - DELTA_T) / 1e6, gpu_power_before / 1e3, temperature, pstate, (time_current - time_kernel_start_ex - DELTA_T) / 1e6);
				gpu_power_before_printed = true;
			}

			printf("%.5f ms\t%.5f W\t%u C\t%d\t%.5f ms\n", (time_running_update - time_start_kernel) / 1e6, gpu_power / 1e3, temperature, pstate, (time_current - time_kernel_start_ex) / 1e6);
			time_current = time_running_update;
		}

     pthread_join( pthread, NULL);
    
	}

	long long time_simulation_end;
  unsigned int t;
  int p;
	synchronizeTime(time_simulation_end, nvml_device, t, p);
	printf("# end time: %.5f\n", (time_simulation_end - time_kernel_start_ex) / 1e6);

  return 0;
}


int high_resolution_power_profile(void *(*call_cuda_kernel)(void*), void *(*reset_kenel_data)(void*)) {
  int error = nvml_init();
  if (error != 0) {
    return error;
  }
//---------------------------------------------------------------------------------------------------------------------------------
  //set kernel execution time
  reset_kenel_data(NULL);
  time_kernel_start_ex = PAPI_get_real_nsec();
  call_cuda_kernel(NULL);
  cudaDeviceSynchronize();
  long long time_kernel_finish_ex = PAPI_get_real_nsec();
  kernel_execution_time = time_kernel_finish_ex - time_kernel_start_ex; 

  printf("#Kernel execution time: %.5fms\n", kernel_execution_time/1e6);
//---------------------------------------------------------------------------------------------------------------------------------
  reset_kenel_data(NULL);
  return power_profile(call_cuda_kernel, reset_kenel_data);  
}

// in nanoseconds
long long get_exec_time_in_nanoseconds(void *(*call_cuda_kernel)(void*), void *(*reset_kenel_data)(void*)) {
   //get kernel execution time
  reset_kenel_data(NULL);
  time_kernel_start_ex = PAPI_get_real_nsec();
  call_cuda_kernel(NULL);
  cudaDeviceSynchronize();
  long long time_kernel_finish_ex = PAPI_get_real_nsec();
  kernel_execution_time = time_kernel_finish_ex - time_kernel_start_ex;

  return kernel_execution_time;

}
