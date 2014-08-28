#ifndef __HIGH_RESOLUTION_POWER_H__
#define __HIGH_RESOLUTION_POWER_H__

const long long DELTA_T = 15046128; // time between 2 power updates (in nanoseconds) for k20

/*
 * Profiles the power for the specified cuda kernel "call_cuda_kernel". The 
 * sequence of kernel calls will be repeated multiple times (can be set by \
 * setting "NVML_REPS" environment variable) and between two consecutive kernel
 * calls there will be a time gap. This gap can be specified by setting 
 * "NVML_SLEEP_TIME" environment variable.
 *
 * call_cuda_kernel - Function pointer containing the sequence of kernel calls. 
 * reset_kenel_data - Set the preconditions for "call_cuda_kernel" like initializing 
 *                    arrays. This function is executed before run the kernels 
 *                    (call_cuda_kernel function pointer). Use this function pointer 
 *                    if you need to reset some data before run the set of kernel calls 
 *                    again. So that, the result after a sequence of kernel calls is 
 *                    correct
 * Return value - Ignore it
 * */
int high_resolution_power_profile(void *(*call_cuda_kernel)(void*), void *(*reset_kenel_data)(void*));
/*
 * Returns the execution time for the sequence of kernels calls in "call_cuda_kernel"
 * call_cuda_kernel - Function pointer containing the sequence of kernel calls. 
 * reset_kenel_data - Set the preconditions for "call_cuda_kernel" like initializing 
 *                    arrays. This function is executed before run the kernels 
 *                    (call_cuda_kernel function pointer). Use this function pointer 
 *                    if you need to reset some data before run the set of kernel calls 
 *                    again. So that, the result after a sequence of kernel calls is 
 *                    correct
 * */
long long get_exec_time_in_nanoseconds(void *(*call_cuda_kernel)(void*), void *(*reset_kenel_data)(void*));

/*
 * Deprecated function. Not recommend to use. Set the environment variable 
 * "NVML_REPS" instead.
 */
void set_reps(int nvml_reps);

#endif //__HIGH_RESOLUTION_POWER_H__
