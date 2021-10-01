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
    

    if (optixGetPrimitiveIndex() == 0)
    {
        //make_float3(0.1f, 0.2f, 0.1f));
        const HitGroupData hitData = *(const HitGroupData*)optixGetSbtDataPointer();

        //Note each texture element is only 1 value of r|g|b|a in that order so 
        //to get the pixel at location 3,3 we actually need R -[3*4] G- [3*4 + 1] B- [3*4 + 2] A [3*4 + 3]
        float locX = 5;
        float locY = 5;
        

        float4 r = tex2D<float4>(hitData.tex, 15     , 15);
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
        whitted::setPayloadResult(make_float3(0.0f,0.0f,255.0f));
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
