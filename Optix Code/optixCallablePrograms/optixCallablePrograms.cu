//
// Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//

#include <optix.h>
#include <sutil/vec_math.h>
//#include <stdio.h>

/*
    NOTE: Cant use whitted_cuda header file as the definition for params is already defined within it, this is a problem as 
    we need to define custom params which mean we cant use the default whitted:launchParams, yet we need the name params
    since some of the camera code IM assuming used it behind the scenes.

*/
//#include <cuda/whitted_cuda.h>

#include "whitted.h"
#include "optixCallablePrograms.h"



namespace whitted {

    //------------------------------------------------------------------------------
    //
    // GGX/smith shading helpers
    // TODO: move into header so can be shared by path tracer and bespoke renderers
    //
    //------------------------------------------------------------------------------

    __device__ __forceinline__ float3 schlick(const float3 spec_color, const float V_dot_H)
    {
        return spec_color + (make_float3(1.0f) - spec_color) * powf(1.0f - V_dot_H, 5.0f);
    }

    __device__ __forceinline__ float vis(const float N_dot_L, const float N_dot_V, const float alpha)
    {
        const float alpha_sq = alpha * alpha;

        const float ggx0 = N_dot_L * sqrtf(N_dot_V * N_dot_V * (1.0f - alpha_sq) + alpha_sq);
        const float ggx1 = N_dot_V * sqrtf(N_dot_L * N_dot_L * (1.0f - alpha_sq) + alpha_sq);

        return 2.0f * N_dot_L * N_dot_V / (ggx0 + ggx1);
    }


    __device__ __forceinline__ float ggxNormal(const float N_dot_H, const float alpha)
    {
        const float alpha_sq = alpha * alpha;
        const float N_dot_H_sq = N_dot_H * N_dot_H;
        const float x = N_dot_H_sq * (alpha_sq - 1.0f) + 1.0f;
        return alpha_sq / (M_PIf * x * x);
    }


    __device__ __forceinline__ float3 linearize(float3 c)
    {
        return make_float3(
            powf(c.x, 2.2f),
            powf(c.y, 2.2f),
            powf(c.z, 2.2f)
        );
    }


    //------------------------------------------------------------------------------
    //
    //
    //
    //------------------------------------------------------------------------------


    static __forceinline__ __device__ void traceRadiance(
        OptixTraversableHandle      handle,
        float3                      ray_origin,
        float3                      ray_direction,
        float                       tmin,
        float                       tmax,
        whitted::PayloadRadiance* payload
    )
    {
        unsigned int u0 = 0, u1 = 0, u2 = 0, u3 = 0;
        optixTrace(
            handle,
            ray_origin, ray_direction,
            tmin,
            tmax,
            0.0f,                     // rayTime
            OptixVisibilityMask(1),
            OPTIX_RAY_FLAG_NONE,
            whitted::RAY_TYPE_RADIANCE,        // SBT offset
            whitted::RAY_TYPE_COUNT,           // SBT stride
            whitted::RAY_TYPE_RADIANCE,        // missSBTIndex
            u0, u1, u2, u3);

        payload->result.x = __int_as_float(u0);
        payload->result.y = __int_as_float(u1);
        payload->result.z = __int_as_float(u2);
        payload->depth = u3;
    }



    static __forceinline__ __device__ bool traceOcclusion(
        OptixTraversableHandle handle,
        float3                 ray_origin,
        float3                 ray_direction,
        float                  tmin,
        float                  tmax
    )
    {
        unsigned int occluded = 0u;
        optixTrace(
            handle,
            ray_origin,
            ray_direction,
            tmin,
            tmax,
            0.0f,                    // rayTime
            OptixVisibilityMask(1),
            OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT,
            whitted::RAY_TYPE_OCCLUSION,      // SBT offset
            whitted::RAY_TYPE_COUNT,          // SBT stride
            whitted::RAY_TYPE_OCCLUSION,      // missSBTIndex
            occluded);
        return occluded;
    }


    __forceinline__ __device__ void setPayloadResult(float3 p)
    {
        optixSetPayload_0(float_as_int(p.x));
        optixSetPayload_1(float_as_int(p.y));
        optixSetPayload_2(float_as_int(p.z));
    }


    __forceinline__ __device__ void setPayloadOcclusion(bool occluded)
    {
        optixSetPayload_0(static_cast<unsigned int>(occluded));
    }

} // namespace whitted



extern "C" {
    __constant__ lightFieldParams params;
}


extern "C" __global__ void __closesthit__ch1()
{
    // When built-in triangle intersection is used, a number of fundamental
    // attributes are provided by the OptiX API, indlucing barycentric coordinates.
    //const float2 barycentrics = optixGetTriangleBarycentrics();
   // printf( "woweee");
    

    if (optixGetPrimitiveIndex() == 0)
    {
        whitted::setPayloadResult(*params.testColor1);
    }
    else
    {
        whitted::setPayloadResult(*params.testColor2);
    }
}

// Miss
extern "C" __global__ void __miss__raydir_shade()
{
    //const float3 ray_dir = optixGetWorldRayDirection();

    //This was causing an invalid memory access, unsure why
    //Have now removed the function this call may have referenced but issue percisted before this was done
    //float3 result = //optixContinuationCall<float3, float3>( 0, ray_dir );


    whitted::setPayloadResult(make_float3(0.001f, 0.0f, 0.0f));
}
