
#include <stdio.h>
#include "cuda_runtime.h"
#include "BsP.h"
    #include <chrono>

void printMatrix(float *matrix, int rows, int cols)
{
    float *host = new float[rows * cols];
    cudaMemcpy(host, matrix, rows * cols * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            printf("%f ", host[i + j * rows]);
        }
        printf("\n");
    }

    delete[] host;
    fflush(stdout);
}

void printMatrix(int *matrix, int rows, int cols)
{
    int *host = new int[rows * cols];
    cudaMemcpy(host, matrix, rows * cols * sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            printf("%d ", host[i + j * rows]);
        }
        printf("\n");
    }

    delete[] host;
    fflush(stdout);
}

void printVector(float *vector, int size)
{
    float *host = new float[size];
    cudaMemcpy(host, vector, size * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < size; i++)
    {
        printf("%f ,", host[i]);
    }
    printf("\n");

    delete[] host;
    fflush(stdout);
}

void printVector(int *vector, int size)
{
    int *host = new int[size];
    cudaMemcpy(host, vector, size * sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < size; i++)
    {
        printf("%d ", host[i]);
    }
    printf("\n");

    delete[] host;
    fflush(stdout);
}

void printCube(float *cube, int rows, int cols, int pages)
{
    //  print for each page
    for (int i = 0; i < pages; i++)
    {
        printf("Page %d\n", i);
        printMatrix(cube + i * rows * cols, rows, cols);
    }
}

void printCube(int *cube, int rows, int cols, int pages)
{
    //  print for each page
    for (int i = 0; i < pages; i++)
    {
        printf("Page %d\n", i);
        printMatrix(cube + i * rows * cols, rows, cols);
    }
}

int main()
{
    constexpr int Tx = 128;
    constexpr int Rx = 128;

    constexpr int dm = 4;

    auto det = Detection<Tx, Rx, QAM<256, RD>>();

    auto bsp = BsP<Tx, Rx, QAM<256, RD>, dm>();

    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < 100; i++)
    {
        det.generate();

        cudaDeviceSynchronize();

        bsp.execute(det);

        cudaDeviceSynchronize();
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    printf("Execution time: %ld milliseconds\n", duration);
}
