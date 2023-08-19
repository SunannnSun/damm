# Directionality-aware Mixture Model Parallel Sampling for Efficient Dynamical System Learning

This module is the implementation of Directionality-aware Mixture Model(DAMM) that has been optimized for real-time learning performance. Given a set of demonstration trajectories, DAMM performs unsupervised learning and fits an augmented Gaussian Mixture Models (GMM) that encodes the structure of given motion while identifying linear components along the trajectory. DAMM serves as the statistical model in the pipeline of Linear Parameter Varying Dynamical System (LPV-DS).

## Note
This module is part of [DAMM-based LPV-DS framework](https://github.com/SunannnSun/damm_lpvds), and cannot be used by its own. Please refer to https://github.com/SunannnSun/damm_lpvds for the usage.

--- 

### Update
8/19 (Last day in a 5 days streak to maintain the damm repository)
- optimize by replacing large vector with pointer
- 
- wrap up

8/18
- ~~finshed split/merge~~
- ~~review and organize a note on circular dependency(line 167 in dpmmDir.cpp)~~ (circular dependency has been removed)
- spectral.cpp containing cv::Kmeans should be converted to a header file like karcher.hpp


8/17
- check computation of the empirical scatter
- line 68 in niwDir.cpp: NOTE ON Posterior SIGMA DIRRECTION
- line 127 in niwDir.cpp, ficed covDir or drawn from posterior


8/16
- ~~return c++ outputs(assignment, etc) as memory binary file to improve efficiency~~
- ~~parse parameters for option 0, 1, 2~~
- need to check the effects of kappa on clustering results
- ~~verify split/merge proposal~~
- makse sure the plot description fit the selected option
- ~~collapsed sample? (already implemented)~~ (inactive method has been hidden)

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
Compile the source code:

```
mkdir build
cd build
cmake ../src
make
```
---

### Current Split/Merge Scheme
- Split proposal of every component every 30 iteration before t=150
- Merge proposal between two selected components every 3 iteration after t=30 until t=175
- IW Gibbs filling in between