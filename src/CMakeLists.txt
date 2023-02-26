
# Add source to this project's executable.
add_executable(vulkan_start
    main.cpp
    vk_engine.cpp
    vk_engine.h
    vk_types.h
    vk_initializers.cpp
    vk_initializers.h
    vk_mesh.h
    vk_mesh.cpp)


set_property(TARGET vulkan_start PROPERTY VS_DEBUGGER_WORKING_DIRECTORY "$<TARGET_FILE_DIR:vulkan_start>")

target_include_directories(vulkan_start PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(vulkan_start vkbootstrap vma glm tinyobjloader imgui stb_image)

target_link_libraries(vulkan_start Vulkan::Vulkan sdl2)

add_dependencies(vulkan_start Shaders)
