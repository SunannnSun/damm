#include <iostream>
#include <Eigen/Dense>
#include "global.hpp"

using namespace Eigen;
using namespace std;



template <typename T>
T unsigned_angle(const Matrix<T,Dynamic, 1>&u, const Matrix<T,Dynamic, 1>&v)
{
    T theta;
    theta = acos(u.dot(v));
    return theta; 
}


template <typename T>
Matrix<T,Dynamic, 1> rie_log(const Matrix<T,Dynamic, 1>&pp, const Matrix<T,Dynamic, 1>&xx)
{
    //Return the coordinate of x_tp starting from the tip of p
    Matrix<T,Dynamic, 1> x_tp;
    T theta = unsigned_angle(pp, xx);
    if (theta < 0.001)
    {   
        x_tp.setZero(pp.rows());
        return x_tp; //p and x are same
    }
    return (xx - pp * cos(theta)) * theta / sin(theta);
}

template <typename T>
Matrix<T,Dynamic, 1> rie_exp(Matrix<T,Dynamic, 1>&pp, const Matrix<T,Dynamic, 1>&xx_tp)
{
    // Given the coordinate of x_tp starting from the tip of p,
    // return the coordinate of x starting from the origin
    // T theta;
    T theta = xx_tp.norm();
    if (theta < 0.001)
        return pp;   // p and x are same
    pp =  pp * cos(theta) + xx_tp / theta * sin(theta);
    return pp;
}










// def unsigned_angle(u, v):
//     if np.dot(u, v) > 1:
//         print(np.dot(u, v))
//         return 0
//     elif np.dot(u,v) < -1:
//         print(np.dot(u, v))
//         return np.pi
//     return np.arccos(np.dot(u, v))


// def rie_log(p, x):
//     """Return the coordinate of x_tp starting from the tip of p"""
//     theta = unsigned_angle(p, x)
//     if theta < 0.001:
//         return np.zeros(p.shape)  # p and x are same
//     return (x - p * np.cos(theta)) * theta / np.sin(theta)


// def rie_exp(p, x_tp):
//     """Given the coordinate of x_tp starting from the tip of p,
//     return the coordinate of x starting from the origin"""
//     theta = np.linalg.norm(x_tp)
//     if theta < 0.001:
//         return p   # p and x are same
//     return p * np.cos(theta) + x_tp / theta * np.sin(theta)


// def karcher_mean(data):
//     if data.shape[0] == 1:
//         return data
//     else:
//         num, dim = data.shape
//     p = np.array([1, 0])
//     while True:
//         angle_sum = 0
//         for index in range(num):
//             angle_sum += rie_log(p, data[index, :])
//         x_tilde = 1 / num * angle_sum
//         if np.linalg.norm(x_tilde) <= 0.01:
//             # print(np.linalg.norm(p))
//             return np.array([p[0], p[1]])
//         else:
//             p = rie_exp(p, x_tilde)