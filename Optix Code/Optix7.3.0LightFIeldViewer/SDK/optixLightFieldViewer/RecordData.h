#pragma once
#include <cuda_runtime.h>
#include <optix.h>

struct EmptyData {};

struct HitGroupData
{
    unsigned m_widthInHogel;
    unsigned m_heightInHogel;
    float m_fov;

    unsigned m_texWidth;
    unsigned m_texHeight;

    cudaTextureObject_t m_tex;
};

struct HitGroupDataTexture
{
//    unsigned m_texWidth;
//    unsigned m_texHeight;

    cudaTextureObject_t m_tex;
};


struct HitGroupDataFloat
{
    //    unsigned m_texWidth;
    //    unsigned m_texHeight;

    float m_val;
};


//CUdeviceptr d_hitgroup_record;

// This should be re-evaluated as this basic template is good, but leads to programming overhead elsewhere 
template <typename T>
struct Record
{
    __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

typedef Record<EmptyData>    RayGenRecord;
typedef Record<EmptyData>    MissRecord;
typedef Record<HitGroupData> HitGroupRecord;


typedef Record<HitGroupDataTexture> HitGroupRecordTexture;
typedef Record<HitGroupDataFloat> HitGroupRecordFloat;
typedef Record<CUdeviceptr> HitGroupRecordCudaPointer;
