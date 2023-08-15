# Directionality-aware Mixture Model(Parallel Implementation)

This module consists of the parallel implementation of Directionality-aware Mixture Model(DAMM) that has been optimized for real-time learning performance. Given a set of demonstration trajectories, this module performs unsupervised learning of Gaussian mixture models that best describe the structure of provided data. In addition to the general clustering purposes, this module serves as an intermediate step in the pipeline of learning a Dynamical system-based motion policies from data, and the learned model will proceed to be optimized in the linear parameter varying(LPV) learning of a dynamical system.

--- 

### Update

8/15 Rebuttal Submission

---

### Dependencies
- **[Required]** [Eigen](https://eigen.tuxfamily.org/index.php?title=Main_Page): Eigen library provides fast and efficient implementations of matrix operations and other linear algebra tools. The latest stable release, Eigen 3.4 is required.
- **[Required]** [Boost](https://www.boost.org/): Boost provides portable C++ source libraries that works efficiently with standard C++ libraries. The latest release Boost 1.81 is recommended.
- **[Required]** [OpenMP](https://www.openmp.org/): OpenMP allows the parallel computation in C++ platform.

---

### Installation

Create a python virtual environment and install the dependencies.

```
virtualenv -p /path/to/python3.8 env3.8
source env3.8/bin/activate
pip install -r requirements.txt
```

Make sure to replace `/path/to/python3.8` with the correct path to the Python 3.8 executable on your system. You can use the which command on Unix-like systems or the where command on Windows to find the path to the Python interpreter. For example:

On Unix-like systems (Linux, macOS):

```
which python3.8
```

Compile the source code:

```
mkdir build
cd build
cmake ../src
make
cd ..
```

### Instruction


Input:

1. Franka Emika Demonstration

1. LASA-Handwriting Dataset
2. PC-GMM Benchmark Dataset
<!-- 3. Franka Emika Demonstration -->





```python main.py  [-d DATA] [-t ITERATION] [-a ALPHA] [--init INIT]```