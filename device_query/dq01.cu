#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

/*****************************************************************************
 * This program introduces using cudaGetDeviceProperties to find about the 
 * available GPUs. Information is returned in a cudaDeviceProp data structure
 * which can be found, with a limited amount of documentation in 
 * cuda_runtim_api.h. A lot of the fields are self explanatory so the struct
 * declaration is copied below for convenience. The next few programs in this
 * set demonstrate using this information. When you have tried them you could
 * look at investigating some of the other available fields and by reading 
 * about what they mean gain a greater understanding of CUDA and GPU 
 * architecture.
 * 
    struct cudaDeviceProp {
      char name[256];
      size_t totalGlobalMem;
      size_t sharedMemPerBlock;
      int regsPerBlock;
      int warpSize;
      size_t memPitch;
      int maxThreadsPerBlock;
      int maxThreadsDim[3];
      int maxGridSize[3];
      int clockRate;
      size_t totalConstMem;
      int major;
      int minor;
      size_t textureAlignment;
      size_t texturePitchAlignment;
      int deviceOverlap;
      int multiProcessorCount;
      int kernelExecTimeoutEnabled;
      int integrated;
      int canMapHostMemory;
      int computeMode;
      int maxTexture1D;
      int maxTexture1DMipmap;
      int maxTexture1DLinear;
      int maxTexture2D[2];
      int maxTexture2DMipmap[2];
      int maxTexture2DLinear[3];
      int maxTexture2DGather[2];
      int maxTexture3D[3];
      int maxTexture3DAlt[3];
      int maxTextureCubemap;
      int maxTexture1DLayered[2];
      int maxTexture2DLayered[3];
      int maxTextureCubemapLayered[2];
      int maxSurface1D;
      int maxSurface2D[2];
      int maxSurface3D[3];
      int maxSurface1DLayered[2];
      int maxSurface2DLayered[3];
      int maxSurfaceCubemap;
      int maxSurfaceCubemapLayered[2];
      size_t surfaceAlignment;
      int concurrentKernels;
      int ECCEnabled;
      int pciBusID;
      int pciDeviceID;
      int pciDomainID;
      int tccDriver;
      int asyncEngineCount;
      int unifiedAddressing;
      int memoryClockRate;
      int memoryBusWidth;
      int l2CacheSize;
      int maxThreadsPerMultiProcessor;
      int streamPrioritiesSupported;
      int globalL1CacheSupported;
      int localL1CacheSupported;
      size_t sharedMemPerMultiprocessor;
      int regsPerMultiprocessor;
      int managedMemSupported;
      int isMultiGpuBoard;
      int multiGpuBoardGroupID;
      int singleToDoublePrecisionPerfRatio;
      int pageableMemoryAccess;
      int concurrentManagedAccess;
    }
 * 
 * Compile with:
 *   nvcc -o dq01 dq01.cu
 * 
 * Dr Kevan Buckley, University of Wolverhampton, 2018 
 ****************************************************************************/


int main() {
  
  // cudaSetDevice(0); // only needed if there are multiple GPUs
  
  cudaDeviceProp device_properties;
  cudaGetDeviceProperties(&device_properties, 0);

  int driver_version, runtime_version;
  cudaDriverGetVersion(&driver_version);
  cudaRuntimeGetVersion(&runtime_version);
  
  printf("Device 0 is a \"%s\"\n", device_properties.name);
  printf("  Compute capability %d.%d\n", device_properties.major, 
                                         device_properties.minor);
                                         
  printf("  CUDA Driver version %d.%d\n", 
         driver_version/1000, (driver_version%100)/10);
 
  printf("  CUDA runtime version %d.%d\n",                
         runtime_version/1000, (runtime_version%100)/10);
         
         
  return 0;
}
