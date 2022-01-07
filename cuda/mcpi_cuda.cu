#include <iostream>
#include <chrono>
#include <cstdlib>
#include <random>
#include <iomanip>

#include <cuda.h>
#include <curand_kernel.h>

using namespace std;

typedef chrono::high_resolution_clock timer;

// check if there are any errors launching the kernel
#define cuda_error_check() { cuda_assert(__FILE__, __LINE__); }
inline void cuda_assert(const char *file, int line, bool abort = true)
{
    auto error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        cerr << "CUDA error: "
             << cudaGetErrorString(error)
             << " (" << error << ") -- "
             << file << " -- line: "
             << line << endl;
        if (abort) exit(error);
    }
}

// get input from the command line for total number of tosses
size_t process_cmdline(int argc, char* argv[])
{
    if (argc > 4)
    {
        cout << "Usage: "
             << argv[0]
             << " [number of tosses] [number of threads]"
             << endl;
        return 0;
    }
    else if (1 == argc)
        return 10'000'000;
    else
        return atoll(argv[1]);
}

// device kernel to perform Monte Carlo version of tossing darts at a board
__global__ void cuda_toss(size_t n, size_t* in)
{
    size_t rank = threadIdx.x;
    size_t size = blockDim.x;

    // Initialize RNG
    curandState_t rng;
    curand_init(clock64(), threadIdx.x + blockIdx.x * blockDim.x, 0, &rng);

    in[rank] = 0;                           // local number of points in circle
    for (size_t i = 0; i < n / size; ++i)
    {
        float x = curand_uniform(&rng);     // Random x position in [0,1]
        float y = curand_uniform(&rng);     // Random y position in [0,1]
        // if (x * x + y * y <= 1)          // is point in circle?
        //     ++in[rank];                  // increase thread-local counter
        in[rank] += 1 - int(x * x + y * y); // no conditional version (faster)
    }
}

int main(int argc, char* argv[])
{
    // querying device properties
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    cuda_error_check();

    // set the number of threads
    size_t n_threads = prop.maxThreadsPerBlock;
    if (3 == argc)
    {
        int n = atoi(argv[2]);
        if (0 == n)
        {
            cout << "Usage: "
                 << argv[0]
                 << " [number of tosses] [number of threads]"
                 << endl;
            return -1;
        }
        if (n < n_threads)
            n_threads = n;
    }

    // read total number of tosses from the command line
    size_t n_tosses = process_cmdline(argc, argv);
    if (0 == n_tosses)
        return -1;

    cout << "Monte-Carlo Pi Estimator\n"
         << "Method: CUDA (GPU) -- "
         << n_threads << " thread(s)\n"
         << "Device name: " << prop.name
         << "\nNumber of tosses: " << n_tosses << endl;

    // run the simulation and time it...
    //------> start timer
    timer::time_point start = timer::now();

    // memory for thread local results
    size_t* in_device;
    cudaMalloc(&in_device, n_threads * sizeof(size_t));
    cuda_error_check();
    // start parallel Monte Carlo
    cuda_toss<<<1, n_threads>>>(n_tosses, in_device);
    cuda_error_check();

    // reducing...
    vector<size_t> in(n_threads);
    cudaMemcpy( 
        in.data()
    ,   in_device
    ,   n_threads * sizeof(size_t)
    ,   cudaMemcpyDeviceToHost);
    cuda_error_check();
    cudaFree(in_device);
    size_t n_in_circle{0};
    for (size_t i{0}; i < n_threads; ++i)
        n_in_circle += in[i];

    timer::duration elapsed = timer::now() - start;
    //------> end timer

    // ouput the results
    const long double pi = 3.141592653589793238462643L; // 25-digit Pi
    long double pi_estimate = 4.0L * n_in_circle / n_tosses;
    cout << "Estimated Pi: " << fixed << setw(17) << setprecision(15)
         << pi_estimate << endl
         << "Percent error: " << setprecision(3)
         << abs(pi_estimate - pi) / pi * 100.0 << '%' << endl
         << "Elapsed time: "
         << chrono::duration_cast<chrono::milliseconds>(elapsed).count()
         << " ms" << endl;
}
