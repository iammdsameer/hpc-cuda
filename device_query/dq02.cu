#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

/*****************************************************************************
 * The most complicated thing this program does is work out how many CUDA 
 * cores are available. This is based on the particular architecture found.
 * Documentation for this is difficult to find. The 
 * convert_compute_capability_to_cores function used here was copied from the
 * examples that come with CUDA (but other versions exist on the Internet with
 * different copyright notices!). This code really over complicates things so 
 * just take it for granted that it works. It will need to be updated as new 
 * GPU architectures become available and is known to be out of date at the 
 * time this source code set was put together, i.e. the Volta architecture
 * has superceded Pascal (found in GTX1000 series consumer GPUs, such as the
 * GTX1070s in wlv labs) 
 * 
 * Compile with:
 *   nvcc -o dq02 dq02.cu
 * 
 * Dr Kevan Buckley, University of Wolverhampton, 2018 
 ****************************************************************************/

// This function which is heavily  based on
// NVIDIA_CUDA-8.0_Samples/common/inc/helper_cuda.h _ConvertSMVer2Cores()
// takes a compute capability (in the form of 2 parameters and returns the 
// number of cores per multiprocessor. The algorithm is complicated by
// not all major, minor pairs being valid it might be more readable using a 
// sparse array, but to aid the update of this code when a new architecture 
// comes out as little as possible has been changed.

int convert_compute_capability_to_cores(int major, int minor) {
    // Defines for GPU Architecture types (using the SM version to determine 
    // the # of cores per SM
    typedef struct {
        int SM; // 0xMm (hexidecimal notation)
                // M = SM Major version
                // m = SM minor version
        int Cores;
    } sSMtoCores;

    sSMtoCores nGpuArchCoresPerSM[] = {
      { 0x20, 32 }, // Fermi Generation (SM 2.0) GF100 class
      { 0x21, 48 }, // Fermi Generation (SM 2.1) GF10x class
      { 0x30, 192}, // Kepler Generation (SM 3.0) GK10x class
      { 0x32, 192}, // Kepler Generation (SM 3.2) GK10x class
      { 0x35, 192}, // Kepler Generation (SM 3.5) GK11x class
      { 0x37, 192}, // Kepler Generation (SM 3.7) GK21x class
      { 0x50, 128}, // Maxwell Generation (SM 5.0) GM10x class
      { 0x52, 128}, // Maxwell Generation (SM 5.2) GM20x class
      { 0x53, 128}, // Maxwell Generation (SM 5.3) GM20x class
      { 0x60, 64 }, // Pascal Generation (SM 6.0) GP100 class
      { 0x61, 128}, // Pascal Generation (SM 6.1) GP10x class
      { 0x62, 128}, // Pascal Generation (SM 6.2) GP10x class
      {   -1, -1 }
    };

    int index = 0;

    while (nGpuArchCoresPerSM[index].SM != -1) {
      if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor)) {
          return nGpuArchCoresPerSM[index].Cores;
      }

      index++;
    }

    // If we don't find the values, we default use the previous one to run properly
    printf("MapSMtoCores for SM %d.%d is undefined. Default to use %d Cores/SM\n", 
           major, minor, nGpuArchCoresPerSM[index-1].Cores);
    return nGpuArchCoresPerSM[index-1].Cores;
}


int main() {
  
  cudaDeviceProp device_properties;
  cudaGetDeviceProperties(&device_properties, 0);
  
  printf("Device 0 is a \"%s\"\n", device_properties.name);

  printf("  Clock rate = %d MHz\n", device_properties.clockRate/1000);
  printf("  Memory = %lu bytes\n", device_properties.totalGlobalMem);
  
  int cores_per_multiprocessor = 
    convert_compute_capability_to_cores(device_properties.major, 
                                        device_properties.minor);
  printf("  Number of multi processors = %d\n",
         device_properties.multiProcessorCount);

  printf("  Cores per multiprocessors = %d\n", 
         cores_per_multiprocessor);

  printf("  Total CUDA cores = %d\n",
         cores_per_multiprocessor * device_properties.multiProcessorCount);

  return 0;
}
