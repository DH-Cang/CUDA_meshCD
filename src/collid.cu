//**************************************************************************************
//  Copyright (C) 2022 - 2024, Min Tang (tang_m@zju.edu.cn)
//  All rights reserved.
//  Redistribution and use in source and binary forms, with or without
//  modification, are permitted provided that the following conditions
//  are met:
//     1. Redistributions of source code must retain the above copyright
//        notice, this list of conditions and the following disclaimer.
//     2. Redistributions in binary form must reproduce the above copyright
//        notice, this list of conditions and the following disclaimer in the
//        documentation and/or other materials provided with the distribution.
//     3. The names of its contributors may not be used to endorse or promote
//        products derived from this software without specific prior written
//        permission.
//  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
//  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
//  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
//  A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
//  CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
//  EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
//  PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
//  PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
//  LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
//	NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
//	SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//**************************************************************************************

#include <set>
#include <iostream>
#include <stdio.h>
#include <vector>

using namespace std;
#include "mat3f.h"
#include "box.h"
#include "crigid.cuh"
#include "cuda_colide.cuh"
#include "book.h"




// very robust triangle intersection test
// uses no divisions
// works on coplanar triangles

bool
triContact(vec3f& P1, vec3f& P2, vec3f& P3, vec3f& Q1, vec3f& Q2, vec3f& Q3)
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








// input: kmesh0(this) with transformation t0
//        kmesh1(other) with transformation t1
void
kmesh::collide(const kmesh* other, const transf& t0, const transf &t1, std::vector<id_pair>& rets)
{
	//for (int i = 0; i < _num_tri; i++) {
	//	printf("checking %d of %d...\n", i, _num_tri);

	//	for (int j = 0; j < other->_num_tri; j++) {
	//		vec3f v0, v1, v2;
	//		this->getTriangleVtxs(i, v0, v1, v2);
	//		vec3f p0 = t0.getVertex(v0);
	//		vec3f p1 = t0.getVertex(v1);
	//		vec3f p2 = t0.getVertex(v2);

	//		other->getTriangleVtxs(j, v0, v1, v2);
	//		vec3f q0 = t1.getVertex(v0);
	//		vec3f q1 = t1.getVertex(v1);
	//		vec3f q2 = t1.getVertex(v2);

	//		if (triContact(p0, p1, p2, q0, q1, q2))
	//			rets.push_back(id_pair(i, j, false));
	//	}
	//}

	vec3f* device_bunny_vertex_array;
	tri3f* device_bunny_triangle_array;
	int* device_triangle_num;
	transf* device_transform0;
	transf* device_transform1;
	bool* device_triangle0_result;
	bool* device_triangle1_result;

	// allocate memory
	HANDLE_ERROR(cudaMalloc((void**)&device_bunny_vertex_array, _num_vtx * sizeof(vec3f)));
	HANDLE_ERROR(cudaMalloc((void**)&device_bunny_triangle_array, _num_tri * sizeof(tri3f)));
	HANDLE_ERROR(cudaMalloc((void**)&device_triangle_num, sizeof(unsigned int)));
	HANDLE_ERROR(cudaMalloc((void**)&device_transform0, sizeof(transf)));
	HANDLE_ERROR(cudaMalloc((void**)&device_transform1, sizeof(transf)));
	HANDLE_ERROR(cudaMalloc((void**)&device_triangle0_result, _num_tri * sizeof(bool)));
	HANDLE_ERROR(cudaMalloc((void**)&device_triangle1_result, _num_tri * sizeof(bool)));

	// copy data from host to device
	HANDLE_ERROR(cudaMemcpy(device_bunny_vertex_array, _vtxs, _num_vtx * sizeof(vec3f), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(device_bunny_triangle_array, _tris, _num_tri * sizeof(tri3f), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(device_triangle_num, &_num_tri, sizeof(unsigned int), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(device_transform0, &t0, sizeof(transf), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(device_transform1, &t1, sizeof(transf), cudaMemcpyHostToDevice));

	// call kernel
	//dim3    grids(128, 128);
	//dim3    threads(1024, 1024);
	unsigned int block_size = 256;
	unsigned int num_blocks = (_num_tri + (block_size - 1)) / block_size;
	dim3    grids(num_blocks, num_blocks);
	dim3    threads(block_size, block_size);
	MeshIntersectCUDA << < grids, threads >> > (
		device_bunny_vertex_array, device_bunny_triangle_array, device_transform0,
		device_bunny_vertex_array, device_bunny_triangle_array, device_transform1,
		device_triangle_num, device_triangle0_result, device_triangle1_result);

	// copy result from device to host
	bool* triangle0_result = new bool[_num_tri];
	bool* triangle1_result = new bool[_num_tri];
	cudaMemcpy(triangle0_result, device_triangle0_result, _num_tri * sizeof(bool), cudaMemcpyDeviceToHost);
	cudaMemcpy(triangle1_result, device_triangle1_result, _num_tri * sizeof(bool), cudaMemcpyDeviceToHost);

	// free memory
	HANDLE_ERROR(cudaFree(device_bunny_vertex_array));
	HANDLE_ERROR(cudaFree(device_bunny_triangle_array));
	HANDLE_ERROR(cudaFree(device_triangle_num));
	HANDLE_ERROR(cudaFree(device_transform0));
	HANDLE_ERROR(cudaFree(device_transform1));
	HANDLE_ERROR(cudaFree(device_triangle0_result));
	HANDLE_ERROR(cudaFree(device_triangle1_result));

	int mesh0_first_tri = -1;
	for (int i = 0; i < _num_tri; i++)
	{
		if (triangle0_result[i]) {
			mesh0_first_tri = i;
			break;
		}
	}
	if (mesh0_first_tri == -1) return;

	int mesh1_first_tri = -1;
	for (int i = 0; i < _num_tri; i++)
	{
		if (triangle1_result[i]) {
			mesh1_first_tri = i;
			break;
		}
	}
	if (mesh1_first_tri == -1) return;

	for (int i = 0; i < _num_tri; i++)
	{
		if (triangle0_result[i]) 
		{
			rets.push_back(id_pair(i, mesh1_first_tri, false));
		}
		if (triangle1_result[i])
		{
			rets.push_back(id_pair(mesh0_first_tri, i, false));
		}
	}
	delete[] triangle0_result;
	delete[] triangle1_result;

}
