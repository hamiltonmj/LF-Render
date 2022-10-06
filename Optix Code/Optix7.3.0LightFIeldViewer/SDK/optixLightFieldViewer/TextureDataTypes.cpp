#include <cuda_runtime.h>
#include "opencv2/core.hpp" 
#include <sutil/Exception.h>
#include <iostream>
#include "RecordData.h"
#include "TextureDataTypes.h"

TextureBase::TextureBase(cv::Mat image)
    :m_cuArray(cudaArray_t()), m_texObject(new cudaTextureObject_t()), m_width(image.cols), m_height(image.rows)
{
    //Desribes how cuda should iterpret the array its being given
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<uchar4>();

    //Allocate Array on gpu in the same shape as described by the channel descriptor
    CUDA_CHECK(cudaMallocArray(&m_cuArray, &channelDesc, m_width, m_height));

    //Describes the width of the 2D array, in terms of a 1d array
    const size_t spitch = m_width * sizeof(uchar4);

    //Copies the data on ram into vram, in the format generated above
    CUDA_CHECK(cudaMemcpy2DToArray(m_cuArray, 0, 0, image.data, spitch, m_width * sizeof(uchar4), m_height, cudaMemcpyHostToDevice));

    //Describes how cuda should operate on the texture within the gpu
    cudaTextureDesc texDescr;
    memset(&texDescr, 0, sizeof(cudaTextureDesc));
    texDescr.normalizedCoords = 0;
    texDescr.filterMode = cudaFilterModePoint;
    texDescr.addressMode[0] = cudaAddressModeWrap;
    texDescr.addressMode[1] = cudaAddressModeWrap;
    texDescr.readMode = cudaReadModeElementType;

    //Describes how the texture should recognise the array it holds
    cudaResourceDesc texRes;
    memset(&texRes, 0, sizeof(cudaResourceDesc));
    texRes.resType = cudaResourceTypeArray;
    texRes.res.array.array = m_cuArray;

    //Assign the texture object all the produced properties
    CUDA_CHECK(cudaCreateTextureObject(m_texObject, &texRes, &texDescr, NULL));
}

TextureBase::~TextureBase()
{
    CUDA_CHECK(cudaFreeArray(m_cuArray));
    CUDA_CHECK(cudaDestroyTextureObject(*m_texObject));
};


/// <summary>
/// Texture Data: Storage class used to store pointer information about a texture on the graphics card also stores general information about the 
/// texture. THis is to describe none lightfield textures
/// </summary>
/// <param name="image"> Given an image matrix will generate a texture o nthe gpu representing it</param>
TextureData::TextureData(cv::Mat image)
    : TextureBase::TextureBase(image) {}

/// <summary>
/// In order for optix to understand how to represent the texzture on an object it needs it in a hitrecord, this method converts our texture into a hit record,
/// Note this object still posses the ability to destroy the texture object, and as such can be dangerous if we destroy a texture still being expected by the hit record
/// </summary>
/// <returns> A hitRecord object which can be placed into a hitrecord for optix to use </returns>
HitGroupData TextureData::toHitRecord()
{
    HitGroupData rec;
    rec.m_texWidth = m_width;
    rec.m_texHeight = m_height;
    rec.m_tex = *m_texObject;
    rec.m_widthInHogel = m_width;
    rec.m_heightInHogel = m_height;
    rec.m_fov = 180;
    return rec;
}

/// <summary>
/// 
/// </summary>
/// <param name="image">Given an image matrix will generate a texture on the gpu representing it</param>
/// <param name="inWidthInHogel"> Specific light field parameters, width in hogels of image | Note: this represents the view resolution</param>
/// <param name="inHeightInHogel">Specific light field parameters, height in hogels of image| Note: this represents the view resolution</param>
/// <param name="infov">Specific light field parameters, fov of lightfield</param>
LightFieldData::LightFieldData(cv::Mat image, unsigned inWidthInHogel, unsigned inHeightInHogel, unsigned infov)
    : TextureBase::TextureBase(image), m_widthInHogels(inWidthInHogel), m_heightInHogels(inHeightInHogel), m_fov(infov)
{
    if (m_heightInHogels < 1)
    {
        m_heightInHogels = image.rows;
    }
    if (m_widthInHogels < 1)
    {
        m_widthInHogels = image.cols;
    }
}

/// <summary>
/// In order for optix to understand how to represent the texzture on an object it needs it in a hitrecord, this method converts our texture into a hit record,
/// Note this object still posses the ability to destroy the texture object, and as such can be dangerous if we destroy a texture still being expected by the hit record
/// </summary>
/// <returns> A hitRecord object which can be placed into a hitrecord for optix to use </returns>
HitGroupData LightFieldData::toHitRecord()
{
    HitGroupData rec;
    rec.m_texWidth = m_width;
    rec.m_texHeight = m_height;
    rec.m_tex = *m_texObject;

    rec.m_widthInHogel = m_widthInHogels;
    rec.m_heightInHogel = m_heightInHogels;
    rec.m_fov = m_fov;
    return rec;
}

/*
LFVideoData::LFVideoData(cv::Mat image, unsigned inWidthInHogel, unsigned inHeightInHogel, unsigned infov, int curFrame)
    : LightFieldData::LightFieldData(image, inWidthInHogel, inHeightInHogel, infov), m_curFrame(curFrame)
{
    if (m_heightInHogels < 1)
    {
        m_heightInHogels = image.rows;
    }
    if (m_widthInHogels < 1)
    {
        m_widthInHogels = image.cols;
    }
}
*/
/*
class LFVideoData : public LightFieldData
{
    unsigned m_widthInHogels;
    unsigned m_heightInHogels;
    float m_fov;
public:

    LFVideoData::LFVideoData(cv::Mat image, unsigned inWidthInHogel = 0, unsigned inHeightInHogel = 0, unsigned infov = 180, size_t curFrame = 0);
    HitGroupData LFVideoData::toHitRecord();
};
*/
