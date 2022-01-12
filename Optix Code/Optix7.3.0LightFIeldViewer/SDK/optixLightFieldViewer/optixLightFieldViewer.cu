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
#include "math_constants.h"
#include <cuda/whitted_cuda.h>

#include "whitted.h"
#include "optixLightFieldViewer.h"


extern "C" {
    __constant__ whitted::LaunchParams params;
}


extern "C" __global__ void __closesthit__ch1()
{
    // When built-in triangle intersection is used, a number of fundamental
    // attributes are provided by the OptiX API, indlucing barycentric coordinates.
    //const float2 barycentrics = optixGetTriangleBarycentrics();
    const HitGroupData hitData = *(const HitGroupData*)optixGetSbtDataPointer();
    float2 uv = optixGetTriangleBarycentrics();    

    uv = (optixGetPrimitiveIndex() == 1) ? make_float2(1 - uv.x, uv.y) : make_float2(uv.x, 1 - uv.y);

    float maxAngleLength = tan((hitData.fov / 2.0) * CUDART_PI_F / 180.0);
    float3 dir = optixGetWorldRayDirection();
    float horizontalAngle = (dir.x / -dir.z) / maxAngleLength;
    float verticalAngle =   (dir.y / -dir.z) / maxAngleLength;
    
    horizontalAngle = (horizontalAngle/ 2) + 0.5;
    verticalAngle = (verticalAngle / 2) + 0.5;

    if (horizontalAngle >= 1 || verticalAngle >= 1 || horizontalAngle < 0 || verticalAngle < 0)
    {
        whitted::setPayloadResult(make_float3(0, 0, 0));
        return;
    }

    float HogelX = round(hitData.widthInHogel   * uv.x);
    float HogelY = round(hitData.heightInHogels * uv.y);

    //This is the number of pixels per hogel
    float pixelsPerHogelX = hitData.texWidth /  hitData.widthInHogel;
    float pixelsPerHogelY = hitData.texHeight / hitData.heightInHogels;

    //This is the position within a hogel an angle corresponds too.
    // We get the number of pixels within a single hogel then use the percentage generated from  inHogel Index to get recieve the actual location.
    float inHogelPosX = pixelsPerHogelX * horizontalAngle;
    float inHogelPosY = pixelsPerHogelY * verticalAngle;

    float texPosX = ((HogelX * pixelsPerHogelX) + inHogelPosX);
    float texPosY = ((HogelY * pixelsPerHogelY) + inHogelPosY);

   if (hitData.texHeight != hitData.heightInHogels) { texPosY = hitData.texHeight - texPosY; }
//    texPosY = (hitData.texHeight == hitData.heightInHogels) ? texPosY : hitData.texHeight - texPosY;
    float4 r = tex2D<float4>(hitData.tex, texPosX, texPosY);
    whitted::setPayloadResult(make_float3(r.x, r.y, r.z));
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
