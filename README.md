# DistViz

This framework includes the implementation of the following algorithms.
  - [Distributed Random Projection Trees (DRPT)](https://github.com/HipGraph/DRPT) algorithm constructs a K-Nearest Neighbor Graph (KNNG) for Approximate Nearest Neighbor Search (ANNS) from a given dataset.
  - [Force2Vec](https://github.com/HipGraph/Force2Vec) graph emdbedding algorithm which can be used to visualize the resulting K-Nearest Neighbor Graph (KNNG) produced by DRPT.

Theese algorithms are implemented in C++ and leverages MPI for distributed processing, making it suitable for large-scale datasets across multiple nodes in HPC environments.


## System Requirments
Ensure the following tools and libraries are installed before building the project. The source code was compiled and run successfully in NERSC Perlmutter.
```
GCC version >= 13.2
OpenMP version >= 4.5
CMake version >= 3.30.2
intel version >= 2024.1.0
cray-mpich version >= 8.1.30
CombBLAS
Eigen version >= 3.3.9
PCG version >= 0.98
```

Some helpful links for installation can be found at [GCC](https://gcc.gnu.org/install/), [OpenMP](https://clang-omp.github.io) and [Environment Setup](http://heather.cs.ucdavis.edu/~matloff/158/ToolsInstructions.html#compile_openmp).

Please refer to the following guides to install CombBLAS, Eigen, and PCG respectively:
  1. CombBLAS - [Installation Guide](https://github.com/PASSIONLab/CombBLAS/blob/master/README.md)
  2. Eigen - [Installation Guide](https://libeigen.gitlab.io/docs/GettingStarted.html)
  3. PCG - [Download & Setup](https://www.pcg-random.org/download.html)

## Installation
1. Load/ install the modules mentioned in System Requirments. 
2. Clone the repository using the following command.
```bash
git clone https://github.com/HipGraph/DistViz.git
```
3.  Set the following environment variables
```bash
export COMBLAS_ROOT=<path/to/CombBLAS/root>
export EIGEN_ROOT=<path/to/Eigen/root>
export PCG_ROOT=<path/to/PCG/root>
```
4. Run the following commands in order to compile and build the framework using CMake.
```bash
cd DistViz
mkdir build
cd build
cmake -DCMAKE_CXX_FLAGS="-fopenmp"  ..
make all
```

## Run
Let’s run an example to demonstrate how to execute the DRPT algorithm on a dataset and visualize the resulting K-Nearest Neighbor Graph (KNNG) using the Force2Vec embedding algorithm within this framework.

For this example, we’ll be using the [Cora dataset](https://paperswithcode.com/dataset/cora).

This test run is designed for High-Performance Computing (HPC) environments with Slurm as the workload manager. Ensure you have access to an HPC cluster running Slurm before proceeding.

### Build the KNNG and Generate Embeddings
1. Navigate into the [test](./test) folder.
```bash
cd test
```
1. Before running the framework, create a directory to store the outputs. For example:
```bash
mkdir output
```
2. Run the framework by submitting the Slurm script [run_distviz.sh](./test/run_distviz.sh), as a batch job.
```bash
sbatch run_distviz.sh
```
  - This script will run the DRPT algorithm on the Cora dataset located in [test/data/cora](./test/data/cora) and generates the KNNG and its embeddings.
  - Here, the K value is set to 10, and the workload is distributed among 4 processors.
3. Once the job completes:
    - Check the output log file (`%j.log`) for progress and any errors.
    - Verify that the KNNG output file (`knng.txt`) appears in your specified output directory.
    - The `knng.txt` file should contain an edge list representing the KNNG graph.
    - Additionally, confirm that an `embedding.txt` file is generated in the output directory. This file contains the generated embeddings corresponding to the nodes in the KNNG, which can be used for visualization and analysis.
4. In order to generate the embeddings, convert the `knng.txt` file into Matrix Market (`.mtx`) format by adding the following two lines to the top of the file and renaming the extension to `mtx` instead of `txt`.
```
%%MatrixMarket matrix coordinate pattern general
2708 2708 27080
```
  - First two 2708s represent the number of nodes (rows and columns of the adjacency matrix).
  - The last 27080 is the number of edges (non-zero entries).

### Visualize the KNNG using the Embeddings
1. Make sure you have python (version >= 3.6) is installed. You can check your version with:
```bash
python3 --version
```
2. Next create and activate a virtual environment to keep dependencies organized:
```
pip install venv
python3 -m venv venv
source venv/bin/activate
```
3. Install the following python dependnies using `pip install`.
```
scikit-learn version >= 0.24.2
matplotlib version >= 3.3.4
networkx version >= 2.5.1
community version >= 1.0.0
```
4. Run the python script [run_visualization.py](./test/visualization/run_visualization.py) (in the visualization directory) to generate the visualization of the KNNG:
```bash
cd visualization
python3 -u ./run_visualization.py ../output/knng.mtx 1 ../output/embedding.txt 2 ../data/cora/cora.nodes.labels cora_viz
```
> Note: [cora.nodes.labels](./test/data/cora/cora.nodes.labels) represents the actual node lables of the dataset.
6. After running the script, check if a PDF file with the visualization has been generated.
    - Open the PDF to ensure the embeddings are properly visualized. You should see a clear structure where similar data points (data points belonging to the same label) cluster together, reflecting the generated KNNG.
