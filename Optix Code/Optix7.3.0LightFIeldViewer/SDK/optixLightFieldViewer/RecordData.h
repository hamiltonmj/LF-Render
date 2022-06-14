#pragma once
#include <cuda_runtime.h>
#include <optix.h>

struct EmptyData {};

class HitGroupData
{
public:

    unsigned m_widthInHogel;
    unsigned m_heightInHogels;
    float m_fov;

    unsigned m_texWidth;
    unsigned m_texHeight;

    cudaTextureObject_t m_tex;
};

// This should be re-evaluated as this basic template is good, but leads to programming overhead elsewhere 
template <typename T>
struct Record
{
    __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};
