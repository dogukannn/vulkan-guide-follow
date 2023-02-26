﻿// vulkan_guide.h : Include file for standard system include files,
// or project specific include files.

#pragma once

#include <vk_types.h>

namespace vkinit {

	//vulkan init code goes here

	VkCommandPoolCreateInfo command_pool_create_info(uint32_t queueFamilyIndex, VkCommandPoolCreateFlags flags = 0);

	VkCommandBufferAllocateInfo command_buffer_allocate_info(VkCommandPool pool, uint32_t count = 1, VkCommandBufferLevel level = VK_COMMAND_BUFFER_LEVEL_PRIMARY);

	VkPipelineShaderStageCreateInfo pipeline_shader_stage_create_info(VkShaderStageFlagBits stage, VkShaderModule shaderModule);

	VkPipelineVertexInputStateCreateInfo pipeline_vertex_input_state_create_info();

	VkPipelineInputAssemblyStateCreateInfo pipeline_input_assembly_state_create_info(VkPrimitiveTopology topology);

	VkPipelineRasterizationStateCreateInfo pipeline_rasterization_state_create_info(VkPolygonMode polygonMode);

	VkPipelineMultisampleStateCreateInfo pipeline_multisample_state_create_info();

	VkPipelineColorBlendAttachmentState pipeline_color_blend_attachment_state();

	VkPipelineLayoutCreateInfo pipeline_layout_create_info();

	VkFenceCreateInfo fence_create_info(VkFenceCreateFlags flags);

	VkSemaphoreCreateInfo semaphore_create_info(VkSemaphoreCreateFlags flags);
}

