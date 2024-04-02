# Directionality-aware Mixture Model Parallel Sampling for Efficient Dynamical System Learning

Barebone implementation of Directionality-aware Mixture Model(DAMM) that has been optimized for near real-time learning performance. Given a set of demonstration trajectories, DAMM performs unsupervised learning and fits an augmented Gaussian Mixture Models (GMM) that encodes the structure of given motion while identifying linear components along the trajectory. DAMM serves as the statistical model in the pipeline of Linear Parameter Varying Dynamical System (LPV-DS), exhibiting state-of-the-art performance.

![damm](https://github.com/SunannnSun/damm/assets/97807687/c14b3afe-a50d-43bc-b437-2dfbe864bbf0)



## Note
This module is part of [LPV-DS framework](https://github.com/SunannnSun/damm_lpvds), and cannot be used by its own. Please refer between this repo and damm-lpvds https://github.com/SunannnSun/damm_lpvds for the usage. The module has been tested in both Mac OS 12.6 (M1) and Ubuntu 24.04.

--- 

### Dependencies
- **[Required]** [Eigen](https://eigen.tuxfamily.org/index.php?title=Main_Page): Eigen 3.4 is required.
- **[Required]** [Boost](https://www.boost.org/): Boost 1.74 is recommended.
- **[Required]** [OpenMP](https://www.openmp.org/): OpenMP 5.0 is recommended.
- **[Required]** [OpenCV](https://opencv.org/) : OpenCV 4.8 is recommended.

---




### Caution:

Mac users please download LLVM compiler separately. The built-in clang compiler in Xcode does not support OpenMP. Please refer to ``CMakeLists.txt`` in ``src`` folder to ensure correct compiler.

### Compilation

```
mkdir build
cd build
cmake ../src
make
```