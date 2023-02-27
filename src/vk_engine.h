// vulkan_guide.h : Include file for standard system include files,
// or project specific include files.

#pragma once

#include <vk_types.h>
#include <vector>
#include <functional>
#include "vk_mesh.h"
#include <glm/glm.hpp>
#include "glm/gtc/matrix_transform.hpp"

#define VK_1SEC 1000000000

class DeletionQueue
{
	std::vector<std::function<void()>> deletionQueue;
public:
	void PushFunction(std::function<void()>&& function);
	void Flush();
};

struct MeshPushConstants
{
	glm::vec4 data;
	glm::mat4 render_matrix;
};

class PipelineBuilder
{
public:
	std::vector<VkPipelineShaderStageCreateInfo> _shaderStages;
	VkPipelineVertexInputStateCreateInfo _vertexInputInfo;
	VkPipelineInputAssemblyStateCreateInfo _intputAssembly;
	VkViewport _viewport;
	VkRect2D _scissor;
	VkPipelineRasterizationStateCreateInfo _rasterizer;
	VkPipelineColorBlendAttachmentState _colorBlendAttachmentState;
	VkPipelineMultisampleStateCreateInfo _multisampling;
	VkPipelineLayout _pipelineLayout;
	VkPipelineDepthStencilStateCreateInfo _depthStencil;
	
	VkPipeline build_pipeline(VkDevice device, VkRenderPass pass);
};

class VulkanEngine {
public:
	int shaderIndex = 0;
	bool _isInitialized{ false };
	int _frameNumber {0};

	VmaAllocator _allocator;
	
	DeletionQueue _mainDeletionQueue;
	
	VkExtent2D _windowExtent{ 1700 , 900 };

	struct SDL_Window* _window{ nullptr };

	VkInstance _instance;
	VkDebugUtilsMessengerEXT _debug_messenger;
	VkPhysicalDevice _chosenGPU;
	VkDevice _device;
	VkSurfaceKHR _surface;

	VkSwapchainKHR _swapchain;
	VkFormat _swapchainImageFormat;

	std::vector<VkImage> _swapchainImages;
	std::vector<VkImageView> _swapchainImageViews;

	VkImageView _depthImageView;
	AllocatedImage _depthImage;

	VkFormat _depthFormat;
	
	VkQueue _graphicsQueue;
	uint32_t _graphicsQueueFamily;

	VkCommandPool _commandPool;
	VkCommandBuffer _mainCommandBuffer;

	VkRenderPass _renderPass;

	std::vector<VkFramebuffer> _framebuffers;
	
	VkSemaphore _presentSemaphore, _renderSemaphore;
	VkFence _renderFence;

	VkPipelineLayout _trianglePipelineLayout;
	VkPipelineLayout _meshPipelineLayout;
	VkPipeline _coloredTrianglePipeline;
	VkPipeline _redTrianglePipeline;
	VkPipeline _meshPipeline;
	Mesh _triangleMesh;
	Mesh _monkeyMesh;
	
	void load_meshes();
	void upload_mesh(Mesh& mesh);
	
	void init_vulkan();
	void init_swapchain();
	void init_commands();
	void init_default_renderpass();
	void init_framebuffers();
	void init_sync_structures();

	bool load_shader_module(const char* filePath, VkShaderModule& outShaderMoudle);
	void init_pipeline();
	
	//initializes everything in the engine
	void init();

	//shuts down the engine
	void cleanup();

	//draw loop
	void draw();

	//run main loop
	void run();
};
