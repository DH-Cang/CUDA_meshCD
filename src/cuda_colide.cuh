#pragma once
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "vec3f.h"
#include "mat3f.h"
#include "box.h"
#include "crigid.cuh"

inline __host__ __device__ double fmax(double a, double b, double c)
{
	double t = a;
	if (b > t) t = b;
	if (c > t) t = c;
	return t;
}

inline __host__ __device__ double fmin(double a, double b, double c)
{
	double t = a;
	if (b < t) t = b;
	if (c < t) t = c;
	return t;
}

inline __host__ __device__ int project3(const vec3f& ax,
	const vec3f& p1, const vec3f& p2, const vec3f& p3)
{
	double P1 = ax.dot(p1);
	double P2 = ax.dot(p2);
	double P3 = ax.dot(p3);

	double mx1 = fmax(P1, P2, P3);
	double mn1 = fmin(P1, P2, P3);

	if (mn1 > 0) return 0;
	if (0 > mx1) return 0;
	return 1;
}

inline __host__ __device__ int project6(vec3f& ax,
	vec3f& p1, vec3f& p2, vec3f& p3,
	vec3f& q1, vec3f& q2, vec3f& q3)
{
	double P1 = ax.dot(p1);
	double P2 = ax.dot(p2);
	double P3 = ax.dot(p3);
	double Q1 = ax.dot(q1);
	double Q2 = ax.dot(q2);
	double Q3 = ax.dot(q3);

	double mx1 = fmax(P1, P2, P3);
	double mn1 = fmin(P1, P2, P3);
	double mx2 = fmax(Q1, Q2, Q3);
	double mn2 = fmin(Q1, Q2, Q3);

	if (mn1 > mx2) return 0;
	if (mn2 > mx1) return 0;
	return 1;
}

__device__ bool GPUTriangleContact(vec3f& P1, vec3f& P2, vec3f& P3, vec3f& Q1, vec3f& Q2, vec3f& Q3);

// input:
//	kmesh0's all vertices with its transformation
//	kmesh1's all vertices with its transformation
// return:
//	a list of pair: intersected triangles
__global__ void MeshIntersectCUDA(
	int* mesh0_tri_ids, int* mesh1_tri_ids,
	vec3f* mesh0_vertex_array, tri3f* mesh0_triangle_array,
	vec3f* mesh1_vertex_array, tri3f* mesh1_triangle_array,
	int triangle0_num, int triangle1_num, 
	bool* triangle0_result, bool* triangle1_result);

// vertex array: each vertex will be transformed
// vertex result: vertex_result[i] == true, if this vertex i is in the bounding sphere of another mesh
__global__ void MeshPreprocessCUDA(
	vec3f* vertex_array,
	bool* vertex_result,
	BoundingSphere* other_b_sphere,
	transf* transform,
	int vertex_num);


__global__ void TriCullingCUDA(
	tri3f* triangle_array,
	bool* tri_culling_result,
	bool* vertex_culling_result,
	int tri_num
);
