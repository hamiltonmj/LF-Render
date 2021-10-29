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
#include "optixCallablePrograms.h"


extern "C" {
    __constant__ whitted::LaunchParams params;
}


extern "C" __global__ void __closesthit__ch1()
{
    // When built-in triangle intersection is used, a number of fundamental
    // attributes are provided by the OptiX API, indlucing barycentric coordinates.
    //const float2 barycentrics = optixGetTriangleBarycentrics();
    //printf( "woweee");
    const HitGroupData hitData = *(const HitGroupData*)optixGetSbtDataPointer();
   // unsigned int w = hitData.texWidth;
    //unsigned int h = hitData.texHeight;

    OptixTraversableHandle gas = optixGetGASTraversableHandle();
    unsigned int index = optixGetPrimitiveIndex();
    unsigned int sbtIndx = optixGetSbtGASIndex();
    float time = optixGetRayTime();

    float3 vertexData[3];
    //optixGetGASTraversableHandle();
    optixGetTriangleVertexData(gas, index, sbtIndx, time, vertexData);

    float2 uv = optixGetTriangleBarycentrics();
    float w;
    if (index == 1)
    {
//        whitted::setPayloadResult(make_float3(1, 1, 1));
  //          return;
        uv = make_float2(1 - uv.x,uv.y);
//        w = 1 + (uv.x + uv.y);
    }
    else
    {
        uv = make_float2(uv.x,1 - uv.y);
        w = 1 - (uv.x + uv.y);
    }


    float maxAngleLength = tan((hitData.fov / 2.0) * CUDART_PI_F / 180.0);
    float3 dir = optixGetWorldRayDirection();
    float horizontalAngle = (dir.x / -dir.z) / maxAngleLength;
    float verticalAngle =   (dir.y / -dir.z) / maxAngleLength;

    
    horizontalAngle = (horizontalAngle/ 2) + 0.5;
    verticalAngle = (verticalAngle / 2) + 0.5;

    if (horizontalAngle >= 1 || verticalAngle >= 1 || horizontalAngle < 0 || verticalAngle < 0)
    {
        whitted::setPayloadResult(make_float3(1, 1, 0));
        return;
    }



    float HogelX;
    float HogelY;
/*
    if (index == 0)
    {
        float2 inverseUV = make_float2(1 - uv.x, 1 - uv.y);
        HogelX = floor(hitData.widthInHogel * inverseUV.x);
        HogelY = floor(hitData.heightInHogels * inverseUV.y);
    }
    else
*/ {
        HogelX = round(hitData.widthInHogel   * uv.x);
        HogelY = round(hitData.heightInHogels * uv.y);
    }

    HogelY = hitData.heightInHogels - HogelY;
    //This is the number of pixels per hogel
    float pixelsPerHogelX = hitData.texWidth /  hitData.widthInHogel;
    float pixelsPerHogelY = hitData.texHeight / hitData.heightInHogels;

    //This is the position within a hogel an angle corresponds too.
    // We get the number of pixels within a single hogel then use the percentage generated from  inHogel Index to get recieve the actual location.
    float inHogelPosX = pixelsPerHogelX * horizontalAngle;
    float inHogelPosY = pixelsPerHogelY * verticalAngle;

    //printf("ABC: %2.5f \n", (HogelX * pixelsPerHogelX));


    float texPosX = ((HogelX * pixelsPerHogelX) + inHogelPosX);
    float texPosY = ((HogelY * pixelsPerHogelY) + inHogelPosY);

//    float texPosX = ((HogelX * pixelsPerHogelX) + 2);
//    float texPosY = ((HogelY * pixelsPerHogelY) + 6);


    if (texPosX >= hitData.texWidth)
    {
        whitted::setPayloadResult(make_float3(0, 0, 1));
        return;
    }
    if (texPosX >= hitData.texWidth)
    {
        whitted::setPayloadResult(make_float3(0, 1, 0));
        return;
    }
    float4 r = tex2D<float4>(hitData.tex, texPosX, texPosY);
    whitted::setPayloadResult(make_float3(r.x, r.y, r.z));
    return;
/*


    //Right now we imply that normal of 

    float x; 
    float y; 
    float z; 

    float3 ab; 
    float3 ac; 

        ab = vertexData[1] - vertexData[0];
        ac = vertexData[2] - vertexData[0];

    x = ((vertexData[1].x * uv.x) + (vertexData[2].x * uv.x) + (vertexData[0].x * uv.x));
    y = ((vertexData[1].y * uv.y) + (vertexData[2].y * uv.y) + (vertexData[0].y * uv.y));
    z = ((vertexData[1].z * w) + (vertexData[2].z * w) + (vertexData[0].z * w)) / 2;

    //printf("x: %4.5f , y: %4.5f, z: %4.5f    |||", vertexData[0].x , vertexData[0].y, vertexData[0].z);


    float3 norm = normalize(cross(ab, ac));
    
    //float3 dir = normalize(optixGetWorldRayDirection() - optixGetWorldRayOrigin());// make_float3(x, y, z));
     
    float3 uDir = (optixGetWorldRayDirection() + optixGetWorldRayOrigin());
    float3 dir =  (optixGetWorldRayDirection() + make_float3(x, y, z));
    
    // norm represents the first plane and we then use the plane projection formula to retrieve the sin(theta) of this 
    //horizontal angle surface 
    float dotNormDir = dot(ab, dir);
    dotNormDir = dotNormDir < 0 ? -dotNormDir : dotNormDir;
    float cosTheta = dotNormDir / length(ab) * length(dir);

    //We repeat this process to create the vertical angle now using the (norm X plane direction) to create our plane
    //vertical angle surface
    float dotVerticalDir = dot(ac, dir);
    dotVerticalDir = dotVerticalDir < 0 ? -dotVerticalDir : dotVerticalDir;
    float cosOmega = dotVerticalDir / length(ac) * length(dir);


    float halfcosFOV = 1 - cos(((hitData.fov/2.0) * CUDART_PI_F/180.0));
    if (halfcosFOV < 0) { halfcosFOV = -halfcosFOV; }

    //POTENTIAL PROBLEM HERE AS NO GRADIENT FOR ANGLE CHANGE
    //0-1 value of the position within a hogel the pixel will take;
    float inHogelIndexX = (cosTheta/ halfcosFOV);
    float inHogelIndexY = (cosOmega/ halfcosFOV);

    //printf("ABC: %2.5f \n", cosTheta);

    //These if statements convert the angles given from the form 1,0 then back to 0,1 based on the angle(0 -> 180 degrees) and instead now angles fall on 0,1 in total 
    if (inHogelIndexX >= 1 || inHogelIndexY >= 1 )
    {
        whitted::setPayloadResult(make_float3(1, 1, 1));
    //    return;
    }
    if (dir.x > 0)
    {
        inHogelIndexX = 0.5 - inHogelIndexX;
    }
    else
    {
        inHogelIndexX += 0.5; 
    }
    if (dir.y > 0)
    {
        inHogelIndexY += 0.5;
    }
    else
    {
        inHogelIndexY = 0.5 - inHogelIndexY / 2;
    }

    //inHogelIndexX = 1.0 - inHogelIndexX;
    //inHogelIndexY = 1.0 - inHogelIndexY;

   // printf("THETA: %2.5f \n", inHogelIndexX);


    if (inHogelIndexX >= 1 || inHogelIndexY >= 1)
    {
        whitted::setPayloadResult(make_float3(1, 1, 1));
        //    return;
    }

    float HogelX;
    float HogelY;

    if (index == 0)
    {
        float2 inverseUV = make_float2(1 - uv.x, 1 - uv.y);
        HogelX = floor(hitData.widthInHogel * inverseUV.x);
        HogelY = floor(hitData.heightInHogels * inverseUV.y);
    }
    else
    {
        HogelX = floor(hitData.widthInHogel * uv.x);
        HogelY = floor(hitData.heightInHogels * uv.y);
    }

    HogelY = hitData.heightInHogels - HogelY;
    //This is the number of pixels per hogel
    float pixelsPerHogelX = hitData.texWidth  / hitData.widthInHogel;
    float pixelsPerHogelY = hitData.texHeight / hitData.heightInHogels;

    //This is the position within a hogel an angle corresponds too.
    // We get the number of pixels within a single hogel then use the percentage generated from  inHogel Index to get recieve the actual location.
    float inHogelPosX = (pixelsPerHogelX * inHogelIndexX);
    float inHogelPosY = (pixelsPerHogelY * inHogelIndexY);
    //printf("X: %2.5f \n", optixGetWorldRayOrigin().x);
    //printf("Y: %2.5f \n", optixGetWorldRayOrigin().y);
    //printf("Z: %2.5f \n", optixGetWorldRayOrigin().z);


    float texPosX = ((HogelX * pixelsPerHogelX) +inHogelPosX);
    float texPosY = ((HogelY * pixelsPerHogelY) +inHogelPosY);



    if (cosOmega > -0.01 && cosOmega < 0.01)
    {
        whitted::setPayloadResult(make_float3(0.01, 0.01, 0.01));
        return;

    }
    if (cosTheta > -0.01 && cosTheta < 0.01)
    {
        whitted::setPayloadResult(make_float3(0.01, 0.01, 0.01));
        return;
    }

    //float4 r = tex2D<float4>(hitData.tex, texPosX, texPosY);

//    printf("ABC: %2.5f \n", uv.x);
    float4 r = tex2D<float4>(hitData.tex, uv.x* hitData.texWidth, uv.y * hitData.texHeight);

    whitted::setPayloadResult(make_float3(uv.x, uv.y, 0));

   // whitted::setPayloadResult(make_float3(r.x, r.y, r.z));
   // whitted::setPayloadResult(make_float3(0, inHogelIndexY, inHogelIndexY));
   // whitted::setPayloadResult(make_float3((cosTheta + cosOmega)/2.0, 0, 0));
    //whitted::setPayloadResult(make_float3((inHogelIndexX), inHogelIndexY, 0) );
    //whitted::setPayloadResult(make_float3(cosOmega, cosOmega, cosOmega) );
*/
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
