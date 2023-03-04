#pragma once

#include <vk_types.h>
#include <vk_engine.h>

namespace vkutil
{
    bool load_image_from_files(VulkanEngine& engine, const char* file, AllocatedImage& outImage);   
}
