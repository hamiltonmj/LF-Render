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
    unsigned int w = hitData.texWidth;
    unsigned int h = hitData.texHeight;

    float2 uv = optixGetTriangleBarycentrics();

    OptixTraversableHandle gas = optixGetGASTraversableHandle();
    unsigned int index = optixGetPrimitiveIndex();
    unsigned int sbtIndx = optixGetSbtGASIndex();
    float time = optixGetRayTime();

    float3 vertexData[3];
    //optixGetGASTraversableHandle();
    optixGetTriangleVertexData(gas, index, sbtIndx, time, vertexData);
    

    float3 ab = vertexData[1] - vertexData[0];
    float3 ac = vertexData[2] - vertexData[0];
    //printf("x: %4.5f , y: %4.5f, z: %4.5f    |||", vertexData[0].x , vertexData[0].y, vertexData[0].z);

    float3 norm = normalize(cross(vertexData[1] - vertexData[0], vertexData[2] - vertexData[0]));
    float3 dir = normalize(optixGetWorldRayDirection() - optixGetWorldRayOrigin());


    // norm represents the first plane and we then use the plane projection formula to retrieve the sin(theta) of this 
    //horizontal angle surface 
    float dotNormDir = dot(normalize(vertexData[1] - vertexData[0]), dir);
    dotNormDir = dotNormDir < 0 ? -dotNormDir : dotNormDir;
    float cosTheta = dotNormDir / length(normalize(vertexData[1] - vertexData[0])) * length(dir);
    //We repeat this process to create the vertical angle now using the (norm X plane direction) to create our plane
    //vertical angle surface
    float dotVerticalDir = dot(normalize(vertexData[2] - vertexData[0]), dir);
    dotVerticalDir = dotVerticalDir < 0 ? -dotVerticalDir : dotVerticalDir;

    float cosOmega = dotVerticalDir / length(normalize(vertexData[2] - vertexData[0])) * length(dir);

    // printf("THETA: %2.5f \n", sinTheta);

    float halfcosFOV = cos(hitData.fov / 2);

    //0-1 value of the position within a hogel the pixel will take;
    float lightFieldIndexX = (cosTheta + halfcosFOV)/ halfcosFOV*2;
    float lightFieldIndexY = (cosOmega + halfcosFOV)/ halfcosFOV*2;

    //This is the hogel we are working in.
    float outPutPixelX = hitData.widthInHogel * uv.x;
    float outPutPixelY = hitData.heightInHogels * uv.y;

    //This is the position within a hogel an angle corresponds too.
    float inHogelPosX = hitData.texWidth / hitData.widthInHogel * lightFieldIndexX;
    float inHogelPosY = hitData.texHeight / hitData.heightInHogels * lightFieldIndexX;




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

    if (index == 0)
    {

        //length(norm)* length(dir);
        //make_float3(0.1f, 0.2f, 0.1f));

       // cudaResourceDesc* texDesc;
       // cudaGetTextureObjectResourceDesc(texDesc, hitData.tex)

        //Note each texture element is only 1 value of r|g|b|a in that order so 
        //to get the pixel at location 3,3 we actually need R -[3*4] G- [3*4 + 1] B- [3*4 + 2] A [3*4 + 3]
        float2 inverseUV = make_float2(1 - uv.x, 1 - uv.y);

        float4 r = tex2D<float4>(hitData.tex, w* inverseUV.x     , h * inverseUV.y);
        //unsigned char r = tex2D<float>(hitData.tex, 0, 0);
        //unsigned char g = tex2D<float>(hitData.tex, 0.5f / (80+1) , locY);
        //float b = (float) tex2D<float>(hitData.tex, 0.5f / (80 +2), locY);
        //float a = (float) tex2D<float>(hitData.tex, locX * 4 + 3, locY);
        //whitted::setPayloadResult(abc);
        //const char *a = new char(abc.x);
       //printf("r: %4.5f , g: %4.5f, B: %4.5f    |||", r.x , r.y, r.z);
      //  float fr = ((float)r) / 255;
       // float fr = ((float)r) / 255;
        //float fr = ((float)r) / 255;

        whitted::setPayloadResult(make_float3(r.x, r.y, r.z));
    }
    else
    {
        //float2 inverseUV = make_float2(1 - uv.x, 1 - uv.y);

        float4 r = tex2D<float4>(hitData.tex, w * uv.x, h * uv.y);
        whitted::setPayloadResult(make_float3(r.x, r.y, r.z));
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
