#include "cuda_colide.cuh"


// 定义核函数（在GPU上运行的函数）
__global__ void vectorAddition(const float* a, const float* b, float* c, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // 检查索引是否越界
    if (idx < size) {
        c[idx] = a[idx] + b[idx];
    }
}

int here() {
    const int size = 1024; // 向量大小
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

    // 分配主机内存
    float* h_a = new float[size];
    float* h_b = new float[size];
    float* h_c = new float[size];

    // 初始化向量数据
    for (int i = 0; i < size; ++i) {
        h_a[i] = i;
        h_b[i] = i * 2;
    }

    // 分配设备内存
    float* d_a, * d_b, * d_c;
    cudaMalloc((void**)&d_a, size * sizeof(float));
    cudaMalloc((void**)&d_b, size * sizeof(float));
    cudaMalloc((void**)&d_c, size * sizeof(float));

    // 将数据从主机内存复制到设备内存
    cudaMemcpy(d_a, h_a, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size * sizeof(float), cudaMemcpyHostToDevice);

    // 启动核函数
    vectorAddition << < blocksPerGrid, threadsPerBlock >> > (d_a, d_b, d_c, size);

    // 将结果从设备内存复制回主机内存
    cudaMemcpy(h_c, d_c, size * sizeof(float), cudaMemcpyDeviceToHost);

    // 打印结果
    for (int i = 0; i < 10; ++i) {
        std::cout << h_c[i] << " ";
    }

    // 释放内存
    delete[] h_a;
    delete[] h_b;
    delete[] h_c;
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}