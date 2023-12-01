/**
 * This file includes all utility methods
 */
#include "common.h"

using namespace std;
using namespace std::chrono;
MPI_Datatype hipgraph::distviz::common::SPTUPLE;
MPI_Datatype hipgraph::distviz::common::DENSETUPLE;

vector<string> hipgraph::distviz::common::perf_counter_keys = {
    "IO Time", "KNNG Communication Time", "KNNG Total Time", "Embedding Communication Time","Embedding Total Time"};

map<string, int> hipgraph::distviz::common::call_count;
map<string, double> hipgraph::distviz::common::total_time;

int hipgraph::distviz::common::divide_and_round_up(uint64_t num, int denom) {
  if (num % denom > 0) {
    return num / denom + 1;
  } else {
    return num / denom;
  }
}

void hipgraph::distviz::common::prefix_sum(vector<int> &values, vector<int> &offsets) {
  int sum = 0;
  for (int i = 0; i < values.size(); i++) {
    offsets.push_back(sum);
    sum += values[i];
  }
}

vector<uint64_t> hipgraph::distviz::common::generate_random_numbers(int lower_bound,
                                                         int upper_bound,
                                                         int seed, int ns) {
  vector<uint64_t> vec(ns);
  std::minstd_rand generator(seed);

  // Define the range of the uniform distribution
  std::uniform_int_distribution<int> distribution(lower_bound, upper_bound-1);

  // Generate and print random numbers
  //#pragma omp parallel
  for (int i = 0; i < ns; ++i) {
    int random_number = distribution(generator);
    vec[i] = static_cast<uint64_t>(random_number);
  }
  return vec;
}

size_t hipgraph::distviz::common::get_memory_usage() {
  std::ifstream statm("/proc/self/statm");
  if (statm.is_open()) {
    unsigned long size, resident, shared, text, lib, data, dt;
    statm >> size >> resident >> shared >> text >> lib >> data >> dt;

    // Memory values are in pages, typically 4 KB each on Linux
    size_t pageSize = sysconf(_SC_PAGESIZE);
    size_t virtualMemUsed = size * pageSize;
    size_t residentSetSize = resident * pageSize;

    size_t mem_usage = residentSetSize / (1024 * 1024);
    return mem_usage;
  }
  return 0;
}

void hipgraph::distviz::common::reset_performance_timers() {
  for (auto it = perf_counter_keys.begin(); it != perf_counter_keys.end();
       it++) {
    call_count[*it] = 0;
    total_time[*it] = 0.0;
  }
}

void hipgraph::distviz::common::stop_clock_and_add(my_timer_t &start,
                                        string counter_name) {
  int rank;
  int world_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  if (find(perf_counter_keys.begin(), perf_counter_keys.end(), counter_name) !=
      perf_counter_keys.end()) {
    call_count[counter_name]++;
    total_time[counter_name] += stop_clock_get_elapsed(start);
  } else {
    cout << "Error, performance counter " << counter_name << " not registered."
         << endl;
    exit(1);
  }
}

void hipgraph::distviz::common::add_memory(size_t mem, string counter_name) {
  if (find(perf_counter_keys.begin(), perf_counter_keys.end(), counter_name) !=
      perf_counter_keys.end()) {
    call_count[counter_name]++;
    total_time[counter_name] += mem;
  } else {
    cout << "Error, performance counter " << counter_name << " not registered."
         << endl;
    exit(1);
  }
}

void hipgraph::distviz::common::add_datatransfers(uint64_t count, string counter_name) {
  int rank;
  int world_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  if (find(perf_counter_keys.begin(), perf_counter_keys.end(), counter_name) !=
      perf_counter_keys.end()) {
    call_count[counter_name]++;
    total_time[counter_name] += count;
  } else {
    cout << "Error, performance counter " << counter_name << " not registered."
         << endl;
    exit(1);
  }
}

void hipgraph::distviz::common::print_performance_statistics() {
  // This is going to assume that all timing starts and ends with a barrier,
  // so that all processors enter and leave the call at the same time. Also,
  // I'm taking an average over several calls by all processors; might want to
  // compute the variance as well.

  int rank;
  int world_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  if (rank == 0) {
    cout << endl;
    cout << "================================" << endl;
    cout << "==== Performance Statistics ====" << endl;
    cout << "================================" << endl;
    //      print_algorithm_info();
  }

  cout << json_perf_statistics().dump(4);

  if (rank == 0) {
    cout << "=================================" << endl;
  }
}

json hipgraph::distviz::common::json_perf_statistics() {
  json j_obj;
  int rank;
  int world_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  for (auto it = perf_counter_keys.begin(); it != perf_counter_keys.end();
       it++) {
    double val = total_time[*it];

    MPI_Allreduce(MPI_IN_PLACE, &val, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    // We also have the call count for each statistic timed
    val /= world_size;

    if (rank == 0) {
      j_obj[*it] = val;
    }
  }
  return j_obj;
}

my_timer_t hipgraph::distviz::common::start_clock() {
  return std::chrono::steady_clock::now();
}

double hipgraph::distviz::common::stop_clock_get_elapsed(my_timer_t &start) {
  auto end = std::chrono::steady_clock::now();
  std::chrono::duration<double> diff = end - start;
  return diff.count();
}

std::unordered_set<uint64_t>
hipgraph::distviz::common::random_select(const std::unordered_set<uint64_t> &originalSet, int count) {
  std::unordered_set<uint64_t> result;

  // Check if the count is greater than the size of the original set
  if (count >= originalSet.size()) {
    return originalSet; // Return the original set as-is
  }

  std::random_device rd;  // Random device for seed
  std::mt19937 gen(rd()); // Mersenne Twister PRNG
  std::uniform_int_distribution<int> dis(0, originalSet.size() - 1);

  while (result.size() < count) {
    auto it = originalSet.begin();
    std::advance(it, dis(gen)); // Advance the iterator to a random position
    result.insert(*it); // Insert the selected element into the result set
  }

  return result;
}

int hipgraph::distviz::common::get_proc_length(double beta, int world_size) {
  return std::max(static_cast<int>((beta * world_size)), 1);
}

int hipgraph::distviz::common::get_end_proc(int starting_index, double beta, int world_size) {
  int proc_length = get_proc_length(beta,world_size);
  return std::min((starting_index + proc_length), world_size);
}
