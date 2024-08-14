#pragma once

#include <cstring>
#include <stdexcept>
#include <vector>
#include <cuComplex.h>

#include <cuda_runtime.h>
#include <curand.h>

// CUDA API error checking
#define CUDA_CHECK(err)                                                        \
  do {                                                                         \
    cudaError_t err_ = (err);                                                  \
    if (err_ != cudaSuccess) {                                                 \
      std::printf("CUDA error %d at %s:%d\n", err_, __FILE__, __LINE__);       \
      throw std::runtime_error("CUDA error");                                  \
    }                                                                          \
  } while (0)

// curand API error checking
#define CURAND_CHECK(err)                                                      \
  do {                                                                         \
    curandStatus_t err_ = (err);                                               \
    if (err_ != CURAND_STATUS_SUCCESS) {                                       \
      std::printf("curand error %d at %s:%d\n", err_, __FILE__, __LINE__);     \
      throw std::runtime_error("curand error");                                \
    }                                                                          \
  } while (0)

// cublas API error checking
#define CUBLAS_CHECK(err)                                                                          \
    do {                                                                                           \
        cublasStatus_t err_ = (err);                                                               \
        if (err_ != CUBLAS_STATUS_SUCCESS) {                                                       \
            std::printf("cublas error %d at %s:%d\n", err_, __FILE__, __LINE__);                   \
            throw std::runtime_error("cublas error");                                              \
        }                                                                                          \
    } while (0)

template <typename T>
void printVector(T *vec, int n, std::string name = "")
{
    std::vector<T> v(n);
    cudaMemcpy(v.data(), vec, n * sizeof(T), cudaMemcpyDeviceToHost);
    if (name != "")
    {
        printf("%s: \n", name.c_str());
    }
    for (int i = 0; i < n; i++)
    {
        printf("%f ", static_cast<double>(v[i]));
    }
    printf("\n");
}

// cuComplex 和 cuDoubleComplex 重载
template <>
void printVector<cuComplex>(cuComplex *vec, int n, std::string name)
{
    std::vector<cuComplex> v(n);
    cudaMemcpy(v.data(), vec, n * sizeof(cuComplex), cudaMemcpyDeviceToHost);
    if (name != "")
    {
        printf("%s: \n", name.c_str());
    }
    for (int i = 0; i < n; i++)
    {
        printf("(%f, %f) ", v[i].x, v[i].y);
    }
    printf("\n");
}

template <>
void printVector<cuDoubleComplex>(cuDoubleComplex *vec, int n, std::string name)
{
    std::vector<cuDoubleComplex> v(n);
    cudaMemcpy(v.data(), vec, n * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);
    if (name != "")
    {
        printf("%s: \n", name.c_str());
    }
    for (int i = 0; i < n; i++)
    {
        printf("(%f, %f) ", v[i].x, v[i].y);
    }
    printf("\n");
}



// column-major matrix
template <typename T>
void printMatrix(T *Mat, int row, int col, std::string name = "") 
{
    std::vector<T> vec(row * col);
    cudaMemcpy(vec.data(), Mat, row * col * sizeof(T), cudaMemcpyDeviceToHost);
    if (name != "")
    {
        printf("%s: \n", name.c_str());
    }

    for (int i = 0; i < row; i++)
    {
        for (int j = 0; j < col; j++)
        {
            printf("%f ", vec[j * row + i]);
            // print a , after each element except the last one
            if (j != col - 1)
            {
                printf(", ");
            }
        }
        printf(";");
        printf("\n");
    }
    printf("\n");
}

// cuComplex 和 cuDoubleComplex 重载
template <>
void printMatrix<cuComplex>(cuComplex *Mat, int row, int col, std::string name)
{
    std::vector<cuComplex> vec(row * col);
    cudaMemcpy(vec.data(), Mat, row * col * sizeof(cuComplex), cudaMemcpyDeviceToHost);
    if (name != "")
    {
        printf("%s: \n", name.c_str());
    }
    for (int i = 0; i < row; i++)
    {
        for (int j = 0; j < col; j++)
        {
            printf("(%f, %f) ", vec[j * row + i].x, vec[j * row + i].y);
        }
        printf("\n");
    }
    printf("\n");
}

template <>
void printMatrix<cuDoubleComplex>(cuDoubleComplex *Mat, int row, int col, std::string name)
{
    std::vector<cuDoubleComplex> vec(row * col);
    cudaMemcpy(vec.data(), Mat, row * col * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);
    if (name != "")
    {
        printf("%s: \n", name.c_str());
    }
    for (int i = 0; i < row; i++)
    {
        for (int j = 0; j < col; j++)
        {
            printf("(%f, %f) ", vec[j * row + i].x, vec[j * row + i].y);
        }
        printf("\n");
    }
    printf("\n");
}