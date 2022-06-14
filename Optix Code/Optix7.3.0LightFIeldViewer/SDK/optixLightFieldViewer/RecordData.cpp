#include <cuda_runtime.h>
//#include "TextureDataTypes.h"
#include "RecordData.h"

/*
HitGroupData::HitGroupData(TextureData* texData)
    : m_texWidth(texData->m_width), m_texHeight(texData->m_height), m_widthInHogel(texData->m_height), m_heightInHogels(texData->m_width), m_fov(180), m_tex(texData->m_texObject)
{};

HitGroupData::HitGroupData(LightFieldData* texData)
    : m_texWidth(texData->m_width), m_texHeight(texData->m_height), m_widthInHogel(texData->m_widthInHogels),
    m_heightInHogels(texData->m_widthInHogels), m_fov(texData->m_fov), m_tex(texData->m_texObject)
{};

void HitGroupData::update(TextureData* texData)
{
    m_texWidth = (texData->m_width);
    m_texHeight = (texData->m_height);
    m_tex = (texData->m_texObject);
    m_widthInHogel = (texData->m_width);
    m_heightInHogels = (texData->m_height);
    m_fov = (180);
}
void HitGroupData::update(LightFieldData* texData)
{
    update((TextureData*)texData);
    m_widthInHogel = (texData->m_widthInHogels);
    m_heightInHogels = (texData->m_widthInHogels);
    m_fov = (texData->m_fov);
}
*/

