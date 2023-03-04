// vulkan_guide.h : Include file for standard system include files,
// or project specific include files.

#pragma once

#include <vk_types.h>
#include <vector>
#include <functional>
#include <cvt/wstring>

#include "vk_mesh.h"
#include <glm/glm.hpp>
#include "glm/gtc/matrix_transform.hpp"

#define VK_1SEC 1000000000

struct Material
{
	VkPipeline pipeline;
	VkPipelineLayout pipelineLayout;
};

struct RenderObject
{
	Mesh* mesh;
	Material* material;
	glm::mat4 transformMatrix;
};

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

struct GPUCameraData
{
	glm::mat4 view;
	glm::mat4 proj;
	glm::mat4 viewProj;
};

struct GPUSceneData
{
	glm::vec4 fogColor;
	glm::vec4 fogDistance;
	glm::vec4 ambientColor;
	glm::vec4 sunlightDirection;
	glm::vec4 sunlightColor;
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

struct FrameData
{
	VkSemaphore _presentSemaphore, _renderSemaphore;
	VkFence _renderFence;

	VkCommandPool _commandPool;
	VkCommandBuffer _mainCommandBuffer;

	AllocatedBuffer cameraBuffer;

	VkDescriptorSet globalDescriptor;
};
constexpr unsigned int FRAME_OVERLAP = 2;

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
	VkPhysicalDeviceProperties _gpuProperties;
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

	VkRenderPass _renderPass;

	std::vector<VkFramebuffer> _framebuffers;
	
	FrameData _frames[FRAME_OVERLAP];

	VkDescriptorSetLayout _globalSetLayout;
	VkDescriptorPool _descriptorPool;

	GPUSceneData _sceneParameters;
	AllocatedBuffer _sceneParameterBuffer;
	
	VkPipelineLayout _trianglePipelineLayout;
	VkPipelineLayout _meshPipelineLayout;
	VkPipeline _coloredTrianglePipeline;
	VkPipeline _redTrianglePipeline;
	VkPipeline _meshPipeline;
	Mesh _triangleMesh;
	Mesh _monkeyMesh;

	std::vector<RenderObject> _renderables;
	std::unordered_map<std::string, Material> _materials;
	std::unordered_map<std::string, Mesh> _meshes;

	FrameData& get_current_frame();
	
	void load_meshes();
	void init_scene();
	void upload_mesh(Mesh& mesh);
	Material* create_material(VkPipeline pipeline, VkPipelineLayout layout, const std::string& name);
	Material* get_material(const std::string& name);
	Mesh* get_mesh(const std::string& name);
	void draw_objects(VkCommandBuffer cmd, RenderObject* first, int count);	

	AllocatedBuffer create_buffer(size_t allocSize, VkBufferUsageFlags usage, VmaMemoryUsage memoryUsage);
	void init_descriptors();
	size_t pad_uniform_buffer_size(size_t originalSize);
	
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
