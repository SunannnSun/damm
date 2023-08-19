# Directionality-aware Mixture Model Parallel Sampling for Efficient Dynamical System Learning

This module is the implementation of Directionality-aware Mixture Model(DAMM) that has been optimized for near real-time learning performance. Given a set of demonstration trajectories, DAMM performs unsupervised learning and fits an augmented Gaussian Mixture Models (GMM) that encodes the structure of given motion while identifying linear components along the trajectory. DAMM serves as the statistical model in the pipeline of Linear Parameter Varying Dynamical System (LPV-DS), exhibiting state-of-the-art performance.

## Note
This module is part of [DAMM-based LPV-DS framework](https://github.com/SunannnSun/damm_lpvds), and cannot be used by its own. Please refer between this repo and damm-lpvds https://github.com/SunannnSun/damm_lpvds for the usage.

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


### Update
The code base has been significantly cleaned up during the five day period (8/15 - 8/19). Only minor maintainance and optimization is required in the future. Two matters need to be highlighted:
1. Hyperparameters in ``damm.py`` plays a signicant role in clustering results. Though possible to tune for the best results for different trajectories, it could be more advantageous to explore an adaptive prior that can continuously update. This is especially useful in online and incremental learning secenarios where the previous results should help reshape the prior belief for the incoming new observations.
2. The arrangement of the mixed sampler in ``main.cpp`` remains a heuristic. There exists huge room for improvement by further exploiting the more efficient arrangement. Through obeservations, most clustering should be able to reach convergence within 50~75 iterations if propoerly managing the sampling cycle. The current implementation follows the following scheme:
    - Split proposal of every component every 30 iteration before t=150
    - Merge proposal between two selected components every 3 iteration after t=30 until t=175
    - IW Gibbs filling in between and ends at t=200
    
___


8/19 ~~(Last day in a 5 days streak to wrap up the damm repository)~~
- ~~optimize by replacing large vector with pointer (done by passing arguments by reference)~~
- ~~need to quantify the effects of kappa on clustering results (see what's in Summary)~~
- ~~line 78 in niwDir.cpp: NOTE ON Posterior SIGMA DIRRECTION (posterior inverse chi-squared)~~
- ~~line 130 in niwDir.cpp, ficed covDir or drawn from posterior (implemented drawing from scaled inverse chi-squared) ~~
- modify and verify the CMakeLists.txt on mac and linux


8/18
- ~~finshed split/merge~~
- ~~review and organize a note on circular dependency(line 167 in dpmmDir.cpp)~~ (circular dependency has been removed)
- ~~spectral.cpp containing cv::Kmeans should be converted to a header file like karcher.hpp~~


8/17
- ~~check computation of the empirical scatter~~


8/16
- ~~return c++ outputs(assignment, etc) as memory binary file to improve efficiency~~
- ~~parse parameters for option 0, 1, 2~~
- ~~verify split/merge proposal~~
- ~~makse sure the plot description fit the selected option~~ (PC-GMM not integrated yet, required manual input)
- ~~collapsed sample? (already implemented)~~ (inactive method has been hidden)

8/15 (Rebuttal Submission)
- ~~Started a new branch named module intended to design the damm as a module-only package and can only be imported and used in damm-lpv-ds environment where loading tools are located~~
- ~~if i can circumvent using a csv writer and directly pass the input DATA to c++~~
- ~~need to include a full covariance passed to c++ to allow for clustering in full dimension~~

