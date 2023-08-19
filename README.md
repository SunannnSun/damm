# Directionality-aware Mixture Model(Parallel Implementation)

This module consists of the parallel implementation of Directionality-aware Mixture Model(DAMM) that has been optimized for real-time learning performance. Given a set of demonstration trajectories, this module performs unsupervised learning of Gaussian mixture models that best describe the structure of provided data. In addition to the general clustering purposes, this module serves as an intermediate step in the pipeline of learning a Dynamical system-based motion policies from data, and the learned model will proceed to be optimized in the linear parameter varying(LPV) learning of a dynamical system.

--- 

### Update
8/18
- finshed split/merge
- review and organize a note on circular dependency(line 167 in dpmmDir.cpp) 
- spectral.cpp containing cv::Kmeans should be converted to a header file like karcher.hpp


8/17
- check computation of the empirical scatter
- line 68 in niwDir.cpp: NOTE ON Posterior SIGMA DIRRECTION
- line 127 in niwDir.cpp, ficed covDir or drawn from posterior


8/16
- ~~return c++ outputs(assignment, etc) as memory binary file to improve efficiency~~
- parse parameters for option 0, ~~1, 2~~
- need to check the effects of kappa on clustering results
- verify split/merge proposal
- makse sure the plot description fit the selected option
- collapsed sample? (already implemented)

8/15 Rebuttal Submission
- ~~Started a new branch named module intended to design the damm as a module-only package and can only be imported and used in damm-lpv-ds environment where loading tools are located~~
- ~~if i can circumvent using a csv writer and directly pass the input DATA to c++~~
- ~~need to include a full covariance passed to c++ to allow for clustering in full dimension~~


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



### Current Split/Merge Scheme
- Split proposal of every component every 30 iteration before t=150
- Merge proposal between two selected components every 3 iteration after t=30 until t=175
- IW Gibbs filling in between