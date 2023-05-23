#pragma once 
//#include <glad/glad.h>  // Needs to be included before gl_interop
//#include <GLFW/glfw3.h>
//#include <sutil/Camera.h>
//#include <sutil/CUDAOutputBuffer.h>
//#include <cuda_gl_interop.h>

#include <optix.h>
#include <cuda_runtime.h>
#include "TextureDataTypes.h"
#include <map>
#include <whitted.h>
#include "RecordData.h"

/// <summary>
/// Contains all basic datastructures needed for optix to function
/// including all modules and gas handle (for geometry)
/// </summary>
struct RenderState
{
    OptixDeviceContext          context = 0;
    OptixTraversableHandle      gas_handle = 0;
    CUdeviceptr                 d_gas_output_buffer = 0;

    OptixModule                 camera_module = 0;
    OptixModule                 geometry_module = 0;
    OptixModule                 shading_module = 0;

    OptixProgramGroup           raygen_prog_group = 0;
    OptixProgramGroup           miss_prog_group = 0;
    OptixProgramGroup           hitgroup_prog_group = 0;
    OptixProgramGroup           hitgroup_prog_group1 = 0;

    OptixPipeline               pipeline = 0;
    OptixPipelineCompileOptions pipeline_compile_options = {};

    CUstream                    stream = 0;
    whitted::LaunchParams       params = {};
    whitted::LaunchParams* d_params = 0;
    OptixShaderBindingTable     sbt = {};

    //Simple map to store textures object within the engine, based off the file they were inputted from
    std::map<std::string, std::shared_ptr<TextureBase>> texObjects;

    //In order to speed up video changing we need to store the current hit record being used for video. If adding multiple lightfields, would need to move this into a class specific to the lightfield being displayed, same as the textures wed use for the lf 
    HitGroupRecord m_hitRecord;
    HitGroupRecord m_hitRecord1;
    HitGroupRecordFloat m_hitRecordFloat;
};
