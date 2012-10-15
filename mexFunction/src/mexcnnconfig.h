#define __CUDACNNMEX_FLOAT

//TODO: change CMake script to define this automatically
#ifdef HAVE_CUDA
#ifndef __CUDACNNMEX_CPU
#define __CUDACNNMEX_CUDA
#endif //__CUDACNNMEX_CPU
#else
#define __CUDACNNMEX_CPU
#endif //HAVE_CUDA
// now defined in command line depending on configuration
//#define __CUDACNNMEX_CUDA
//#define __CUDACNNMEX_CPU