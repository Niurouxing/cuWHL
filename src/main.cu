
#include <stdio.h>
#include "cuda_runtime.h"
#include "BsP.h"

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
}

void printVector(float *vector, int size)
{
    float *host = new float[size];
    cudaMemcpy(host, vector, size * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < size; i++)
    {
        printf("%f ", host[i]);
    }
    printf("\n");

    delete[] host;
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
    constexpr int Tx = 3;
    constexpr int Rx = 4;

    constexpr int dm = 3;

    auto det = Detection<Tx, Rx>();

    det.generate();

    auto bsp = BsP<Tx, Rx, QAM<16,RD>,dm>();

    bsp.execute(det);

    printf("H\n");
    printMatrix(bsp.HtH, 2 * Tx, 2 * Tx);

    printf("alpha\n");
    // [ConSize][2 * TxAntNum][2 * RxAntNum]
    printCube(bsp.alpha, det.ConSize, 2 * Tx, 2 * Rx);

    printf("gamma\n");
    printMatrix(bsp.gamma, det.ConSize, 2 * Tx);

    printf("Tx\n");
    printVector(det.TxSymbols, 2 * Tx);
    printVector(det.TxIndices, 2 * Tx);

    printf("Est\n");
    printVector(bsp.HtY, 2 * Tx);

    printf("sIndex\n");
    printCube(bsp.sIndex,dm, 2 * Tx, 2 * Rx);
}