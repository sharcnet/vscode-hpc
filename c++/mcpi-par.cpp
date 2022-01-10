#include <iostream>
#include <chrono>
#include <cstdlib>
#include <random>
#include <iomanip>

#include <omp.h>

using namespace std;

typedef chrono::high_resolution_clock timer;

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

// perform Monte Carlo version of tossing darts at a board
size_t toss(size_t n)
{
    size_t in{};
    std::random_device rx, ry;
    std::uniform_real_distribution<float> u(0, 1);

    // distribute workload over all processes and make a global reduction
    #pragma omp parallel for reduction(+ : in)
    for (auto i = 0; i < n; ++i)
    {
        const float x{u(rx)}, y{u(ry)};  // choose random x- and y-coords
        if (x * x + y * y <= 1.0)        // is point in circle?
            ++in;                        // increase counter
    }

    return in;
}

int main(int argc, char* argv[])
{
    // set the number of threads
    auto n_threads = omp_get_max_threads();
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
         << "Method: Parallel (OpenMP) -- "
         << n_threads << " thread(s)\n"
         << "Number of tosses: " << n_tosses << endl;

    // run the simulation and time it...
    omp_set_num_threads(n_threads);
    timer::time_point start = timer::now();
    size_t n_in_circle = toss(n_tosses);
    timer::duration elapsed = timer::now() - start;

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
