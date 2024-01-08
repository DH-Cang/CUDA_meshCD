#include "cuda_colide.cuh"

// 定义核函数（在GPU上运行的函数）
__global__ void vectorAddition(const float* a, const float* b, float* c, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // 检查索引是否越界
    if (idx < size) {
        c[idx] = a[idx] + b[idx];
    }
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

    //// now begin the series of tests
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
    int* mesh0_tri_ids, int* mesh1_tri_ids,
    vec3f* mesh0_vertex_array, tri3f* mesh0_triangle_array,
    vec3f* mesh1_vertex_array, tri3f* mesh1_triangle_array,
    int triangle0_num, int triangle1_num, 
    bool* triangle0_result, bool* triangle1_result)
{
    int thread_id0 = blockIdx.x * blockDim.x + threadIdx.x;
    int thread_id1 = blockIdx.y * blockDim.y + threadIdx.y;

    // make sure there is no overflow
    if (thread_id0 >= triangle0_num || thread_id1 >= triangle1_num)
    {
        return;
    }

    int triangle_index0 = mesh0_tri_ids[thread_id0];
    int triangle_index1 = mesh1_tri_ids[thread_id1];

    // get vertex coords of triangle0, with transformation
    tri3f triangle0 = mesh0_triangle_array[triangle_index0];
    vec3f triangle0_vertex_coords[3];
    for (int i = 0; i < 3; i++)
    {
        triangle0_vertex_coords[i] = mesh0_vertex_array[triangle0.id(i)];
    }

    // get vertex coords of triangle1, with transformation
    tri3f triangle1 = mesh1_triangle_array[triangle_index1];
    vec3f triangle1_vertex_coords[3];
    for (int i = 0; i < 3; i++)
    {
        triangle1_vertex_coords[i] = mesh1_vertex_array[triangle1.id(i)];
    }

    // get triangle contact check result
    bool result = GPUTriangleContact(
        triangle0_vertex_coords[0], triangle0_vertex_coords[1], triangle0_vertex_coords[2],
        triangle1_vertex_coords[0], triangle1_vertex_coords[1], triangle1_vertex_coords[2]
    );
    
    if (result)
    {  
        triangle0_result[triangle_index0] = true;
        triangle1_result[triangle_index1] = true;
    }    
}


__global__ void MeshPreprocessCUDA(
    vec3f* vertex_array,
    bool* vertex_result,
    BoundingSphere* other_b_sphere,
    transf* transform,
    int vertex_num)
{
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;

    // no overflow
    if (thread_id >= vertex_num)
    {
        return;
    }

    // make transformation
    vec3f vtx_coord = transform->getVertex(vertex_array[thread_id]);
    vertex_array[thread_id] = vtx_coord;

    // check if inside bounding sphere
    bool in_sphere_flag =
        (dist_square(vtx_coord, other_b_sphere->center) <= other_b_sphere->radius * other_b_sphere->radius);
    vertex_result[thread_id] = in_sphere_flag;
}


__global__ void TriCullingCUDA(
    tri3f* triangle_array,
    bool* tri_culling_result,
    bool* vertex_culling_result,
    int tri_num
)
{
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;

    // no overflow
    if (thread_id >= tri_num)
    {
        return;
    }

    unsigned int vtx_id0 = triangle_array[thread_id].id(0);
    unsigned int vtx_id1 = triangle_array[thread_id].id(1);
    unsigned int vtx_id2 = triangle_array[thread_id].id(2);

    bool tri_in_sphere_flag =
        vertex_culling_result[vtx_id0] ||
        vertex_culling_result[vtx_id1] ||
        vertex_culling_result[vtx_id2];

    tri_culling_result[thread_id] = tri_in_sphere_flag;
}

