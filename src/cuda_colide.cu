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

__device__ bool GPUTriangleContact(vec3f& P1, vec3f& P2, vec3f& P3, vec3f& Q1, vec3f& Q2, vec3f& Q3)
{
    vec3f p1;
    vec3f p2 = P2 - P1;
    vec3f p3 = P3 - P1;
    vec3f q1 = Q1 - P1;
    vec3f q2 = Q2 - P1;
    vec3f q3 = Q3 - P1;

    vec3f e1 = p2 - p1;
    vec3f e2 = p3 - p2;
    vec3f e3 = p1 - p3;

    vec3f f1 = q2 - q1;
    vec3f f2 = q3 - q2;
    vec3f f3 = q1 - q3;

    vec3f n1 = e1.cross(e2);
    vec3f m1 = f1.cross(f2);

    vec3f g1 = e1.cross(n1);
    vec3f g2 = e2.cross(n1);
    vec3f g3 = e3.cross(n1);

    vec3f  h1 = f1.cross(m1);
    vec3f h2 = f2.cross(m1);
    vec3f h3 = f3.cross(m1);

    vec3f ef11 = e1.cross(f1);
    vec3f ef12 = e1.cross(f2);
    vec3f ef13 = e1.cross(f3);
    vec3f ef21 = e2.cross(f1);
    vec3f ef22 = e2.cross(f2);
    vec3f ef23 = e2.cross(f3);
    vec3f ef31 = e3.cross(f1);
    vec3f ef32 = e3.cross(f2);
    vec3f ef33 = e3.cross(f3);

    // now begin the series of tests
    if (!project3(n1, q1, q2, q3)) return false;
    if (!project3(m1, -q1, p2 - q1, p3 - q1)) return false;

    if (!project6(ef11, p1, p2, p3, q1, q2, q3)) return false;
    if (!project6(ef12, p1, p2, p3, q1, q2, q3)) return false;
    if (!project6(ef13, p1, p2, p3, q1, q2, q3)) return false;
    if (!project6(ef21, p1, p2, p3, q1, q2, q3)) return false;
    if (!project6(ef22, p1, p2, p3, q1, q2, q3)) return false;
    if (!project6(ef23, p1, p2, p3, q1, q2, q3)) return false;
    if (!project6(ef31, p1, p2, p3, q1, q2, q3)) return false;
    if (!project6(ef32, p1, p2, p3, q1, q2, q3)) return false;
    if (!project6(ef33, p1, p2, p3, q1, q2, q3)) return false;
    if (!project6(g1, p1, p2, p3, q1, q2, q3)) return false;
    if (!project6(g2, p1, p2, p3, q1, q2, q3)) return false;
    if (!project6(g3, p1, p2, p3, q1, q2, q3)) return false;
    if (!project6(h1, p1, p2, p3, q1, q2, q3)) return false;
    if (!project6(h2, p1, p2, p3, q1, q2, q3)) return false;
    if (!project6(h3, p1, p2, p3, q1, q2, q3)) return false;

    return true;
}

__global__ void MeshIntersectCUDA(
    vec3f* mesh0_vertex_array, tri3f* mesh0_triangle_array, transf* transform0,
    vec3f* mesh1_vertex_array, tri3f* mesh1_triangle_array, transf* transform1,
    int* triangle_num, bool* triangle0_result, bool* triangle1_result)
{
    int triangle_index0 = blockIdx.x * blockDim.x + threadIdx.x;
    int triangle_index1 = blockIdx.y * blockDim.y + threadIdx.y;

    // make sure there is no overflow
    if (triangle_index0 >= *triangle_num || triangle_index1 >= *triangle_num)
    {
        return;
    }
    // reduce repeated computation
    if (triangle_index0 > triangle_index1)
    {
        return;
    }

    printf("compare triangle %d in mesh0 with triangle %d in mesh1\n", triangle_index0, triangle_index1);

    // get vertex coords of triangle0, with transformation
    tri3f triangle0 = mesh0_triangle_array[triangle_index0];
    vec3f triangle0_vertex_coords[3];
    for (int i = 0; i < 3; i++)
    {
        triangle0_vertex_coords[i] = mesh0_vertex_array[triangle0.id(i)];
        triangle0_vertex_coords[i] = transform0->getVertex(triangle0_vertex_coords[i]);
    }

    // get vertex coords of triangle1, with transformation
    tri3f triangle1 = mesh1_triangle_array[triangle_index1];
    vec3f triangle1_vertex_coords[3];
    for (int i = 0; i < 3; i++)
    {
        triangle1_vertex_coords[i] = mesh1_vertex_array[triangle1.id(i)];
        triangle1_vertex_coords[i] = transform1->getVertex(triangle1_vertex_coords[i]);
    }

    // get triangle contact check result
    bool result = GPUTriangleContact(
        triangle0_vertex_coords[0], triangle0_vertex_coords[1], triangle0_vertex_coords[2],
        triangle1_vertex_coords[0], triangle1_vertex_coords[1], triangle1_vertex_coords[2]
    );

    triangle0_result[triangle_index0] = result;
    triangle1_result[triangle_index1] = result;
}