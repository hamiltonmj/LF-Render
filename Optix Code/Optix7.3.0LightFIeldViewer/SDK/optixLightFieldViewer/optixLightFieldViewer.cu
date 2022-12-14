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
#include "RecordData.h"
//#include <stdio.h>

extern "C" {
    __constant__ whitted::LaunchParams params;
}

__device__ float4 uchar4Tofloat4(uchar4 a)
{
    return make_float4((float)a.x, (float)a.y, (float)a.z, (float)a.w);
}


__device__ float4 billinearInterp(float texPosX, float texPosY, const cudaTextureObject_t& tex)
{
    //Bottom location for bilinear interp
    float leftX = floor(texPosX);
    float bottomY = floor(texPosY);
    
    float rightX = ceil(texPosX);
    float topY = ceil(texPosY);

    float distLR = (rightX - leftX) == 0 ? 1 : (rightX - leftX);
    float distTB = (topY - bottomY) == 0 ? 1 : (topY - bottomY);

    float dx = (rightX - texPosX) / distLR;
    float dy = (topY - texPosY) / distTB;


    float4 bottomLeft = uchar4Tofloat4(tex2D<uchar4>(tex, leftX, topY));
    float4 bottomRight = uchar4Tofloat4(tex2D<uchar4>(tex, rightX, topY));
    float4 topLeft = uchar4Tofloat4(tex2D<uchar4>(tex, leftX,bottomY));
    float4 topRight = uchar4Tofloat4(tex2D<uchar4>(tex, rightX, bottomY));

    float4 r1 = (topLeft * dx) + (topRight * (1- dx));
    float4 r2 = (bottomLeft * dx) + (bottomRight * (1-dx));
    return (r1 * dy) + (r2 * (1-dy));
}

__device__ float4 nearestNeighbor(float texPosX, float texPosY, const cudaTextureObject_t& tex)
{
    return uchar4Tofloat4(tex2D<uchar4>(tex, roundf(texPosX), roundf(texPosY)));
}


extern "C" __global__ void __closesthit__ch1()
{
    // When built-in triangle intersection is used, a number of fundamental
    // attributes are provided by the OptiX API, indcluding barycentric coordinates.
    const HitGroupData hitData = *(const HitGroupData*)optixGetSbtDataPointer();
    float2 uv = optixGetTriangleBarycentrics();    

    if ((optixGetPrimitiveIndex() == 2)) 
    {
        whitted::setPayloadResult(make_float3(0.0f, 1.0f, 0.0f));
        return;
    
    }
    if ((optixGetPrimitiveIndex() == 3))
    {
        whitted::setPayloadResult(make_float3(1.0f, 1.0f, 0.0f));
        return;

    }
    if ((optixGetPrimitiveIndex() == 4))
    {
        whitted::setPayloadResult(make_float3(0.0f, 1.0f, 1.0f));
        return;

    }


    //uv values are between 0,1 |  0 being left/ top  and 1 being right/bottom
    //Since we have 2 triangles we need to geth them into the same space for referencing the texture
    uv = (optixGetPrimitiveIndex() == 1) ? make_float2(1 - uv.x, uv.y) : make_float2(uv.x, 1 - uv.y);


   // uv.y = 1- uv.y;
    float maxAngleLength = tan((hitData.m_fov / 2.0) * CUDART_PI_F / 180.0);
    float3 dir = optixGetWorldRayDirection();
    float horizontalAngle = (dir.x / -dir.z) / maxAngleLength;
    float verticalAngle =   (dir.y / -dir.z) / maxAngleLength;

    horizontalAngle = (horizontalAngle/ 2) + 0.5;
    verticalAngle = (verticalAngle / 2) + 0.5;

    //IF the viewing angle is outside the capture views of the lightfield black the pixel out
    if (horizontalAngle >= 1 || verticalAngle >= 1 || horizontalAngle < 0 || verticalAngle < 0)
    {
        whitted::setPayloadResult(make_float3(0, 10, 0));
        return;
    }
               
    //Gets the hogel pixel is in 
    float HogelX = floor(hitData.m_widthInHogel  * uv.x);
    float HogelY = floor(hitData.m_heightInHogel * uv.y);

    //This is the number of pixels per hogel
    float pixelsPerHogelX = hitData.m_texWidth /  hitData.m_widthInHogel;
    float pixelsPerHogelY = hitData.m_texHeight / hitData.m_heightInHogel;

    //This is the position within a hogel an angle corresponds too.
    // We get the number of pixels within a single hogel then use the percentage generated from  inHogel Index to recieve the actual location.
    float inHogelPosX = pixelsPerHogelX * horizontalAngle;
    float inHogelPosY = pixelsPerHogelY * verticalAngle;


    float texPosX = ((HogelX * pixelsPerHogelX) + inHogelPosX);
    float texPosY = ((HogelY * pixelsPerHogelY) + inHogelPosY);

    //printf(texPosY);
    if(hitData.m_texHeight != hitData.m_heightInHogel)
    {
        texPosY = hitData.m_texHeight - texPosY;
    }

    //float4 outColor = billinearInterp(texPosX,texPosY, hitData.m_tex);
    float4 outColor = nearestNeighbor(texPosX,texPosY, hitData.m_tex);

   //Converts image pixel values into floats then divides them by 255 to properly represent colors
   // Note: color is originally in bgr format this also converts into rgb 
   whitted::setPayloadResult(make_float3(outColor.z/255, outColor.y/255, outColor.x / 255));
}

// Miss
extern "C" __global__ void __miss__raydir_shade()
{
    //const float3 ray_dir = optixGetWorldRayDirection();

    //This was causing an invalid memory access, unsure why
    //Have now removed the function this call may have referenced but issue percisted before this was done
    //float3 result = //optixContinuationCall<float3, float3>( 0, ray_dir );


    whitted::setPayloadResult(make_float3(0.0f, 0.0f, 1.0f));
}
