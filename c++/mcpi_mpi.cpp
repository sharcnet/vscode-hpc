#include <iostream>
#include <chrono>
#include <cstdlib>
#include <random>
#include <iomanip>

#include <mpi.h>

using namespace std;

typedef chrono::high_resolution_clock timer;

// get input from the command line for total number of tosses
size_t process_cmdline(int argc, char* argv[])
{
    if (argc > 2)
    {
        cout << "Usage: "
             << argv[0]
             << " [number of tosses]"
             << endl;
        return 0;
    }
    else if (1 == argc)
        return 10'000'000;
    else
        return atoll(argv[1]);
}

// perform Monte Carlo version of tossing darts at a board
size_t toss(size_t n, int size, int rank)
{
    size_t in{};
    std::random_device rd;
    std::uniform_real_distribution<float> u(0, 1);


    for (size_t i{size_t(rank)}; i < n; i += size)
    {
        float x{u(rd)}, y{u(rd)};   // choose random x- and y-coordinates
        if (x * x + y * y <= 1.0)   // is point in circle?
            ++in;                   // increase counter
    }
    return in;
}

int main(int argc, char* argv[])
{
    // initialize MPI environment
    int size, rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);  // get total number of processes
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);  // get rank of current process

    // read total number of tosses from the command line
    size_t n_tosses = process_cmdline(argc, argv);
    if (0 == n_tosses)
        return -1;

    if (0 == rank)
    {
        cout << "Method: MPI Monte-Carlo -- "
             << size << " process(es)\n";
        cout << "Number of tosses: " << n_tosses << endl;
    }

    // run the simulation and time it...

    timer::time_point start = timer::now();
    size_t n_partial = toss(n_tosses, size, rank);
    // calculate sum of all local variables 'in' and storre result in 'in_all' on process 0
    size_t n_in_circle;
    MPI_Reduce(&n_partial, &n_in_circle, 1, MPI_UNSIGNED_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    // ouput the results
    if (0 == rank)
    {
        timer::duration elapsed = timer::now() - start;
        const long double pi = 3.141592653589793238462643L; // 25-digit Pi
        long double pi_estimate = 4.0L * n_in_circle / n_tosses;
        cout << "Estimated Pi: " << fixed << setw(17) << setprecision(15)
            << pi_estimate << endl;
        cout << "Error is: " << abs(pi_estimate - pi) << endl;
        cout << "Elapsed time: "
            << chrono::duration_cast<chrono::seconds>(elapsed).count()
            << " seconds" << endl;
    }

    MPI_Finalize();  // quit MPI
}
