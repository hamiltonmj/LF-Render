#pragma once
#include <cuda_runtime.h>
#include "opencv2/core.hpp" 
#include "RecordData.h"


/// <summary>
/// This is a basic virtual class to use when generating textures for our program. generates the basic data all textures will have, such as 
/// the pointers to the texture on the graphics card
///
/// THis is potentially "Bad" could use some other data type to store textures to reduce code complexity if further textures are added
/// BUt for now this does the trick using some facotry based mechansim could have good potential
/// </summary>
class TextureBase
{
public:
    cudaArray_t m_cuArray;
    cudaTextureObject_t* m_texObject;
    unsigned m_texWidth;
    unsigned m_texHeight;
//    HitGroupRecord m_hitRecord; 
    TextureBase(cv::Mat image);
    ~TextureBase();

    virtual HitGroupData* TextureBase::toHitRecord() = 0;
    virtual CUdeviceptr TextureBase::toDeviceHitRecord() = 0;
};

/// <summary>
/// Basic texture object which when created will generate an image texture on the graphics card
/// </summary>
class TextureData : public TextureBase
{
public:
    TextureData(cv::Mat image);

    HitGroupData* TextureData::toHitRecord();
    CUdeviceptr TextureData::toDeviceHitRecord();
};

/// <summary>
/// Advanced texture object which when generated will create a texture object on the graphics card that will be
/// interpretted as a light field by the hit record 
/// </summary>
class LightFieldData : public TextureBase
{
public:
    unsigned m_widthInHogels;
    unsigned m_heightInHogels;
    float m_fov;

    LightFieldData::LightFieldData(cv::Mat image,unsigned inWidthInHogel = 0, unsigned inHeightInHogel = 0,unsigned infov = 180);
    HitGroupData* LightFieldData::toHitRecord();
    CUdeviceptr LightFieldData::toDeviceHitRecord();
};

/*
class LFVideoData : public LightFieldData
{
    int m_curFrame;
public:

    LFVideoData::LFVideoData(cv::Mat image, unsigned inWidthInHogel = 0, unsigned inHeightInHogel = 0, unsigned infov = 180, int curFrame = 0);
    HitGroupData LFVideoData::toHitRecord();
};
*/