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
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <thrust/copy.h>
#include <thrust/iterator/counting_iterator.h>

using namespace std;
#include "mat3f.h"
#include "box.h"
#include "crigid.cuh"
#include "cuda_colide.cuh"
#include "book.h"

#define GPU_ACCELE


#ifndef GPU_ACCELE
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
#endif


// input: kmesh0(this) with transformation t0
//        kmesh1(other) with transformation t1
void
kmesh::collide(const kmesh* other, const transf& t0, const transf &t1, std::vector<id_pair>& rets)
{

#ifndef GPU_ACCELE

	for (int i = 0; i < _num_tri; i++) {
		//printf("checking %d of %d...\n", i, _num_tri);

		for (int j = 0; j < other->_num_tri; j++) {
			vec3f v0, v1, v2;
			this->getTriangleVtxs(i, v0, v1, v2);
			vec3f p0 = t0.getVertex(v0);
			vec3f p1 = t0.getVertex(v1);
			vec3f p2 = t0.getVertex(v2);

			other->getTriangleVtxs(j, v0, v1, v2);
			vec3f q0 = t1.getVertex(v0);
			vec3f q1 = t1.getVertex(v1);
			vec3f q2 = t1.getVertex(v2);

			if (triContact(p0, p1, p2, q0, q1, q2))
				rets.push_back(id_pair(i, j, false));
		}
}
#else
	thrust::device_vector<tri3f> d_mesh0_tris(_tris, _tris + _num_tri);
	thrust::device_vector<tri3f> d_mesh1_tris(other->_tris, other->_tris + other->_num_tri);
	thrust::device_vector<vec3f> d_mesh0_vtxs(_vtxs, _vtxs + _num_vtx);
	thrust::device_vector<vec3f> d_mesh1_vtxs(other->_vtxs, other->_vtxs + other->_num_vtx);
	thrust::device_vector<bool> d_triangle0_result(_num_tri);
	thrust::device_vector<bool> d_triangle1_result(other->_num_tri);

	transf* d_transform0;
	transf* d_transform1;
	// allocate memory
	HANDLE_ERROR(cudaMalloc((void**)&d_transform0, sizeof(transf)));
	HANDLE_ERROR(cudaMalloc((void**)&d_transform1, sizeof(transf)));
	// copy from host to device
	HANDLE_ERROR(cudaMemcpy(d_transform0, &t0, sizeof(transf), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(d_transform1, &t1, sizeof(transf), cudaMemcpyHostToDevice));

	const unsigned int block_size = 16;
	dim3 threads;
	dim3 grids;

	// ====================================== check bounding sphere ======================================

	// prepare result array
	thrust::device_vector<bool> d_vertex0_preprocess_result(_num_vtx);
	thrust::device_vector<bool> d_vertex1_preprocess_result(other->_num_vtx);

	// bounding sphere transformation
	BoundingSphere sphere0(this->m_bounding_sphere), sphere1(other->m_bounding_sphere);
	sphere0.center = t0.getVertex(sphere0.center);
	sphere1.center = t1.getVertex(sphere1.center);

	BoundingSphere* d_sphere0;
	BoundingSphere* d_sphere1;
	// allocate memory
	HANDLE_ERROR(cudaMalloc((void**)&d_sphere0, sizeof(BoundingSphere)));
	HANDLE_ERROR(cudaMalloc((void**)&d_sphere1, sizeof(BoundingSphere)));
	// copy from host to device
	HANDLE_ERROR(cudaMemcpy(d_sphere0, &sphere0, sizeof(BoundingSphere), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(d_sphere1, &sphere1, sizeof(BoundingSphere), cudaMemcpyHostToDevice));

	// =================================== check mesh0 with sphere1 =============================
	// check mesh0 intersect with sphere1, culling vertex
	threads = dim3(block_size);
	grids = dim3((d_vertex0_preprocess_result.size() + (block_size - 1)) / block_size);
	MeshPreprocessCUDA << < grids, threads >> > (
		thrust::raw_pointer_cast(d_mesh0_vtxs.data()),
		thrust::raw_pointer_cast(d_vertex0_preprocess_result.data()),
		d_sphere1,
		d_transform0,
		d_vertex0_preprocess_result.size()
		);
	// culling triangles
	thrust::device_vector<bool> d_tri0_culling_stencil(_num_tri);
	threads = dim3(block_size);
	grids = dim3((d_tri0_culling_stencil.size() + (block_size - 1)) / block_size);
	TriCullingCUDA << < grids, threads >> > (
		thrust::raw_pointer_cast(d_mesh0_tris.data()),
		thrust::raw_pointer_cast(d_tri0_culling_stencil.data()),
		thrust::raw_pointer_cast(d_vertex0_preprocess_result.data()),
		d_tri0_culling_stencil.size()
		);
	// reduce triangle id
	thrust::counting_iterator<int> counting(0);
	thrust::device_vector<int> d_culled_tri0_ids(_num_tri);
	auto end0 = thrust::copy_if(
		counting,
		counting + _num_tri,
		d_tri0_culling_stencil.begin(),
		d_culled_tri0_ids.begin(),
		thrust::identity<bool>()
	);
	d_culled_tri0_ids.resize(end0 - d_culled_tri0_ids.begin());

	// =================================== check mesh1 with sphere0 =============================
	// check mesh1 intersect with sphere0, culling vertex
	grids = dim3((d_vertex1_preprocess_result.size() + (block_size - 1)) / block_size);
	MeshPreprocessCUDA << < grids, threads >> > (
		thrust::raw_pointer_cast(d_mesh1_vtxs.data()),
		thrust::raw_pointer_cast(d_vertex1_preprocess_result.data()),
		d_sphere0,
		d_transform1,
		d_vertex1_preprocess_result.size()
		);
	// culling mesh1 triangles
	thrust::device_vector<bool> d_tri1_culling_stencil(other->_num_tri);
	threads = dim3(block_size);
	grids = dim3((d_tri1_culling_stencil.size() + (block_size - 1)) / block_size);
	TriCullingCUDA << < grids, threads >> > (
		thrust::raw_pointer_cast(d_mesh1_tris.data()),
		thrust::raw_pointer_cast(d_tri1_culling_stencil.data()),
		thrust::raw_pointer_cast(d_vertex1_preprocess_result.data()),
		d_tri1_culling_stencil.size()
		);
	// reduce triangle id
	thrust::device_vector<int> d_culled_tri1_ids(other->_num_tri);
	auto end1 = thrust::copy_if(
		counting,
		counting + other->_num_tri,
		d_tri1_culling_stencil.begin(),
		d_culled_tri1_ids.begin(),
		thrust::identity<bool>()
	);
	d_culled_tri1_ids.resize(end1 - d_culled_tri1_ids.begin());
	
	printf("mesh0 tri num: %d   mesh1 tri num: %d\n", d_culled_tri0_ids.size(), d_culled_tri1_ids.size());


	// ====================================== use cuda intersect ===========================================
	
	// call kernel
	threads = dim3(block_size, block_size);
	grids = dim3(
		(d_culled_tri0_ids.size() + (block_size - 1)) / block_size,
		(d_culled_tri1_ids.size() + (block_size - 1)) / block_size);
	//grids = dim3(500, 500);
	MeshIntersectCUDA << < grids, threads >> > (
		thrust::raw_pointer_cast(d_culled_tri0_ids.data()), thrust::raw_pointer_cast(d_culled_tri1_ids.data()),
		thrust::raw_pointer_cast(d_mesh0_vtxs.data()), thrust::raw_pointer_cast(d_mesh0_tris.data()), 
		thrust::raw_pointer_cast(d_mesh1_vtxs.data()), thrust::raw_pointer_cast(d_mesh1_tris.data()), 
		d_culled_tri0_ids.size(), d_culled_tri1_ids.size(),
		thrust::raw_pointer_cast(d_triangle0_result.data()), thrust::raw_pointer_cast(d_triangle1_result.data()));
	cudaDeviceSynchronize();

	// copy result from device to host
	thrust::host_vector<bool> h_triangle0_result = d_triangle0_result;
	thrust::host_vector<bool> h_triangle1_result = d_triangle1_result;

	

	int mesh0_collide_num = 0;
	int mesh1_collide_num = 0;
	for (int i = 0; i < _num_tri; i++)
	{
		if (h_triangle0_result[i]) mesh0_collide_num++;
		if (h_triangle1_result[i]) mesh1_collide_num++;
	}
	printf("mesh0: %d, mesh1: %d\n", mesh0_collide_num, mesh1_collide_num);

	int mesh0_first_tri = -1;
	for (int i = 0; i < _num_tri; i++)
	{
		if (h_triangle0_result[i]) {
			mesh0_first_tri = i;
			break;
		}
	}
	if (mesh0_first_tri == -1)
	{
		return;
	}

	int mesh1_first_tri = -1;
	for (int i = 0; i < _num_tri; i++)
	{
		if (h_triangle1_result[i]) {
			mesh1_first_tri = i;
			break;
		}
	}
	if (mesh1_first_tri == -1)
	{
		return;
	}

	for (int i = 0; i < _num_tri; i++)
	{
		if (h_triangle0_result[i])
		{
			rets.push_back(id_pair(i, mesh1_first_tri, false));
		}
		if (h_triangle1_result[i])
		{
			rets.push_back(id_pair(mesh0_first_tri, i, false));
		}
	}

	// free memory
	HANDLE_ERROR(cudaFree(d_transform0));
	HANDLE_ERROR(cudaFree(d_transform1));
	HANDLE_ERROR(cudaFree(d_sphere0));
	HANDLE_ERROR(cudaFree(d_sphere1));
#endif // !GPU_ACCELE
}
