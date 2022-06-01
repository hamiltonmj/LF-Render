//
// Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//

#include <glad/glad.h>  // Needs to be included before gl_interop

#include "opencv2/core.hpp" 
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/opencv.hpp"

#include <cuda_gl_interop.h>
#include <cuda_runtime.h>

#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stack_size.h>
#include <optix_stubs.h>

#include <sampleConfig.h>

#include <sutil/CUDAOutputBuffer.h>
#include <sutil/Camera.h>
#include <sutil/Exception.h>
#include <sutil/GLDisplay.h>
#include <sutil/Matrix.h>
#include <sutil/Trackball.h>
#include <sutil/sutil.h>

#include <array>
#include <GLFW/glfw3.h>
#include <cstring>
#include <iomanip>

#include <imgui/imgui.h>
#include <imgui/imgui_impl_glfw.h>
#include <imgui/imgui_impl_opengl3.h>


#include <cuda/whitted.h>

#include "optixLightFieldViewer.h"

#include <fstream>

struct Config
{
    std::string imageName;
    unsigned int widthInHogels = 1;
    unsigned int heightInHogels = 1;
    float fov;
    unsigned int lightFieldWorldWidth;
    unsigned int lightFieldWorldHeight;

};

//------------------------------------------------------------------------------
//
// Globals
//
//------------------------------------------------------------------------------
Config config = Config();

//Can eventually convert all these seperate vectors into a vector of a struct containing this info
std::vector<cudaArray_t>         textureArrays;
std::vector<cudaTextureObject_t> textureObjects;

std::vector<unsigned int>         textureWidths;
std::vector<unsigned int>         textureHeights;

bool resize_dirty = false;
bool minimized    = false;

//image matrix
cv::Mat image1;

// Camera state
bool             camera_changed = true;
sutil::Camera    camera;
sutil::Trackball trackball;

// Shading state
bool         shading_changed = false;
unsigned int dc_index        = 0;

// Mouse state
int32_t mouse_button = -1;

//------------------------------------------------------------------------------
//
// Local types
//
//------------------------------------------------------------------------------
template <typename T>
struct Record
{
    __align__( OPTIX_SBT_RECORD_ALIGNMENT ) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

typedef Record<EmptyData>    RayGenRecord;
typedef Record<EmptyData>    MissRecord;
typedef Record<HitGroupData> HitGroupRecord;
typedef Record<EmptyData>    CallablesRecord;

struct CallableProgramsState
{
    OptixDeviceContext          context                  = 0;
    OptixTraversableHandle      gas_handle               = 0;
    CUdeviceptr                 d_gas_output_buffer      = 0;

    OptixModule                 camera_module            = 0;
    OptixModule                 geometry_module          = 0;
    OptixModule                 shading_module           = 0;

    OptixProgramGroup           raygen_prog_group        = 0;
    OptixProgramGroup           miss_prog_group          = 0;
    OptixProgramGroup           hitgroup_prog_group      = 0;

    OptixPipeline               pipeline                 = 0;
    OptixPipelineCompileOptions pipeline_compile_options = {};

    CUstream                    stream                   = 0;
   whitted::LaunchParams       params                   = {};
   whitted::LaunchParams*      d_params                 = 0;
    OptixShaderBindingTable     sbt                      = {};
};

//------------------------------------------------------------------------------
//
// GLFW callbacks
//
//------------------------------------------------------------------------------

static void mouseButtonCallback( GLFWwindow* window, int button, int action, int mods )
{
    double xpos, ypos;
    glfwGetCursorPos( window, &xpos, &ypos );
    
    //checks ImGui is using mouse or not
    if (!sutil::Get_is_ImGuiActive()) {
        if (action == GLFW_PRESS)
        {
            mouse_button = button;
            trackball.startTracking(static_cast<int>(xpos), static_cast<int>(ypos));
        }

        else
        {
            mouse_button = -1;
        }
    }
}


static void cursorPosCallback( GLFWwindow* window, double xpos, double ypos )
{
   whitted::LaunchParams* params = static_cast<whitted::LaunchParams*>( glfwGetWindowUserPointer( window ) );
  
   //checks ImGui is using mouse or not
   if (!sutil::Get_is_ImGuiActive()) {
       if (mouse_button == GLFW_MOUSE_BUTTON_LEFT)
       {
           trackball.setViewMode(sutil::Trackball::LookAtFixed);
           trackball.updateTracking(static_cast<int>(xpos), static_cast<int>(ypos), params->width, params->height);
           camera_changed = true;
       }

       else if (mouse_button == GLFW_MOUSE_BUTTON_RIGHT)
       {
           trackball.setViewMode(sutil::Trackball::EyeFixed);
           trackball.updateTracking(static_cast<int>(xpos), static_cast<int>(ypos), params->width, params->height);
           camera_changed = true;
       }
   }
   
}


static void windowSizeCallback( GLFWwindow* window, int32_t res_x, int32_t res_y )
{
    // Keep rendering at the current resolution when the window is minimized.
    if( minimized )
        return;

    // Output dimensions must be at least 1 in both x and y.
    sutil::ensureMinimumSize( res_x, res_y );

   whitted::LaunchParams* params = static_cast<whitted::LaunchParams*>( glfwGetWindowUserPointer( window ) );
    params->width                 = res_x;
    params->height                = res_y;
    camera_changed                = true;
    resize_dirty                  = true;
}


static void windowIconifyCallback( GLFWwindow* window, int32_t iconified )
{
    minimized = ( iconified > 0 );
}

int shifted = 1;
static void keyCallback( GLFWwindow* window, int32_t key, int32_t /*scancode*/, int32_t action, int32_t mods )
{
    if(key == GLFW_KEY_ESCAPE )
    {
        if(action == GLFW_PRESS)
        {
            glfwSetWindowShouldClose( window, true );
        }
    }
    /*else*/ if (key == GLFW_KEY_SPACE)
    {
        shading_changed = true;
        dc_index        = ( dc_index + 1 ) % 3;
    }

    int moveRatio = 10  * shifted;
    {
        switch (key)
        {
        //Forward
        case GLFW_KEY_W:
            camera.setEye(camera.eye() + camera.direction() / moveRatio);
            camera.setLookat(camera.lookat() + camera.direction() / moveRatio);
            camera_changed = true;
            break;
        //Backwards
        case GLFW_KEY_S:
            camera.setEye(camera.eye() - camera.direction() / moveRatio);
            camera.setLookat(camera.lookat() - camera.direction() / moveRatio);
            camera_changed = true;
            break;
        //Left
        case GLFW_KEY_A:
            float3 right = normalize(cross(camera.direction(), camera.up()));
            camera.setEye(camera.eye() - right / moveRatio);
            camera.setLookat(camera.lookat() - right / moveRatio);
            camera_changed = true;
            break;
        //Right
        case GLFW_KEY_D:
        {
            float3 right = normalize(cross(camera.direction(), camera.up())); 
            camera.setEye(camera.eye() + right / moveRatio);
            camera.setLookat(camera.lookat() + right / moveRatio);
            camera_changed = true;
            break;
        }
        //Up
        case GLFW_KEY_E:
            camera.setEye(camera.eye() + camera.up() / moveRatio);
            camera.setLookat(camera.lookat() + camera.up() / moveRatio);
            camera_changed = true;
            break;
        //Down
        case GLFW_KEY_Q:
            camera.setEye(camera.eye() - camera.up() / moveRatio);
            camera.setLookat(camera.lookat() - camera.up() / moveRatio);
            camera_changed = true;
            break;

        default:
            break;
        }
    }
}


static void scrollCallback( GLFWwindow* window, double xscroll, double yscroll )
{
    if( trackball.wheelEvent( (int)yscroll ) )
        camera_changed = true;
}


//------------------------------------------------------------------------------
//
//  Helper functions
//
//------------------------------------------------------------------------------

void printUsageAndExit( const char* argv0 )
{
    std::cerr << "Usage  : " << argv0 << " [options]\n";
    std::cerr << "Options: --file | -f <filename>      File for image output\n";
    std::cerr << "         --no-gl-interop             Disable GL interop for display\n";
    std::cerr << "         --dim=<width>x<height>      Set image dimensions; defaults to 768x768\n";
    std::cerr << "         --help | -h                 Print this usage message\n";
    exit( 0 );
}

//return true if fileName has .txt file extension
bool is_textFile(std::string fileName)
{
    std::size_t found = fileName.find("txt");
    if (found != std::string::npos)
    {
        std::string fileExtension = fileName.substr(found, fileName.length());

        if (fileExtension == "txt") {
            return true;
        }
    }
    else return false;
}


void load_Image(std::string fileName)
{
    std::cout << fileName << " :Reading\n";
    std::cout << "Attempting to load Image: " << config.imageName << "\n";
    image1 = cv::imread(fileName, cv::IMREAD_UNCHANGED);
    if (image1.empty())
    {
        std::cout << "Error: Failed to load image - " << fileName << "\n";
        exit;
    }
}


void setConfig(std::string fileName)
{
    //if file selected is .txt file, then directly load the file.
    
    if (is_textFile(fileName)) {
        std::ifstream file;
        file.open(fileName);
        if (!file.is_open())
        {
            std::cout << "Error Opening config\n";
            exit(-1);
        }

        file >> config.imageName >> config.widthInHogels >> config.heightInHogels >> config.fov;
        load_Image(config.imageName);
    }
    else {
        load_Image(fileName);
        config.imageName = fileName;
        config.widthInHogels = image1.cols;
        config.heightInHogels = image1.rows;
        config.fov = 180;

    }


}


void createTexObject(CallableProgramsState& state, const char* filename)
{
    
    int numOfTex = 1;
    std::vector<unsigned char> image;

    textureArrays.resize(textureArrays.size() + numOfTex);
    textureObjects.resize(textureObjects.size() + numOfTex);
    textureWidths.resize(textureWidths.size() + numOfTex);
    textureHeights.resize(textureHeights.size() + numOfTex);

    



    //Cuda textures need 4 channels, so if image does not have them we add an alpha channel
    if (image1.channels() != 4)
    {
        std::cout << "Warning: Invalid Number of Image Channels, attempting to convert BGR to BGRA \n";
        cv::cvtColor(image1, image1, cv::COLOR_BGR2BGRA);
    }

    std::cout << "Succesfully loaded: " << filename << "\n";

    unsigned width = image1.cols;
    unsigned height = image1.rows;

    //Desribes how cuda should iterpret the array its being given
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<uchar4>();

    //Assigns the image to the first texture object within the simulator
    //Currently thhis is not utilized well, but if more lihgt fields are 
    //rednered at once this will be used
    cudaArray_t &cuArray = textureArrays[0];

    //Allocate Array on gpu in the same shape as described by the channel descriptor
    CUDA_CHECK(cudaMallocArray(&cuArray, &channelDesc, width, height));

    //Describes the width of the 2D array, in terms of a 1d array
    const size_t spitch = width  * sizeof(uchar4);

    //Copies the data on ram into vram, in the format generated above
    CUDA_CHECK(cudaMemcpy2DToArray(cuArray, 0, 0, image1.data, spitch, width * sizeof(uchar4), height, cudaMemcpyHostToDevice));
    
    //Describes how cuda should operate on the texture within the gpu
    cudaTextureDesc texDescr;
    memset(&texDescr, 0, sizeof(cudaTextureDesc));
    texDescr.normalizedCoords = 0;
    texDescr.filterMode = cudaFilterModePoint; //cudaFilterModeLinear;
    texDescr.addressMode[0] = cudaAddressModeWrap;
    texDescr.addressMode[1] = cudaAddressModeWrap;
    texDescr.readMode = cudaReadModeElementType;//cudaReadModeNormalizedFloat;

    //Describes how the texture should recognise the array it holds
    cudaResourceDesc texRes;
    memset(&texRes, 0, sizeof(cudaResourceDesc));
    texRes.resType = cudaResourceTypeArray;
    texRes.res.array.array = cuArray;
    
    //Create a blank texture object to be assigned 
    cudaTextureObject_t tex = 0;

    //Assign the texture object all the produced properties
    CUDA_CHECK(cudaCreateTextureObject(&tex, &texRes, &texDescr, NULL));

    // Store the textrue information within the program
    textureObjects[0] = tex;
    textureWidths[0] = width;
    textureHeights[0] = height;
}

void initLaunchParams( CallableProgramsState& state )
{
    CUDA_CHECK(cudaMalloc( reinterpret_cast<void**>( &state.params.accum_buffer ),
                            state.params.width * state.params.height * sizeof( float4 ) ) );
    state.params.frame_buffer = nullptr;  // Will be set when output buffer is mapped

    state.params.subframe_index = 0u;

    // Set ambient light color and point light position
    std::vector<Light> lights( 2 );
    lights[0].type            = Light::Type::AMBIENT;
    lights[0].ambient.color   = make_float3( 0.4f, 0.4f, 0.4f );
    lights[1].type            = Light::Type::POINT;
    lights[1].point.color     = make_float3( 1.0f, 1.0f, 1.0f );
    lights[1].point.intensity = 1.0f;
    lights[1].point.position  = make_float3( 10.0f, 10.0f, -10.0f );
    lights[1].point.falloff   = Light::Falloff::QUADRATIC;

    state.params.lights.count = static_cast<unsigned int>( lights.size() );
    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &state.params.lights.data ), lights.size() * sizeof( Light ) ) );
    CUDA_CHECK( cudaMemcpy( reinterpret_cast<void*>( state.params.lights.data ), lights.data(),
                            lights.size() * sizeof( Light ), cudaMemcpyHostToDevice ) );

    CUDA_CHECK( cudaStreamCreate( &state.stream ) );
    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &state.d_params ), sizeof(whitted::LaunchParams ) ) );

    state.params.handle = state.gas_handle;
}

static void buildGas( const CallableProgramsState&  state,
                      const OptixAccelBuildOptions& accel_options,
                      const OptixBuildInput&        build_input,
                      OptixTraversableHandle&       gas_handle,
                      CUdeviceptr&                  d_gas_output_buffer )
{
    OptixAccelBufferSizes gas_buffer_sizes;
    CUdeviceptr           d_temp_buffer_gas;

    OPTIX_CHECK( optixAccelComputeMemoryUsage( state.context, &accel_options, &build_input, 1, &gas_buffer_sizes ) );

    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_temp_buffer_gas ), gas_buffer_sizes.tempSizeInBytes ) );

    // non-compacted output and size of compacted GAS
    CUdeviceptr d_buffer_temp_output_gas_and_compacted_size;
    size_t      compactedSizeOffset = roundUp<size_t>( gas_buffer_sizes.outputSizeInBytes, 8ull );
    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_buffer_temp_output_gas_and_compacted_size ), compactedSizeOffset + 8 ) );

    OptixAccelEmitDesc emitProperty = {};
    emitProperty.type               = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
    emitProperty.result = ( CUdeviceptr )( (char*)d_buffer_temp_output_gas_and_compacted_size + compactedSizeOffset );

    OPTIX_CHECK( optixAccelBuild( state.context, 0, &accel_options, &build_input, 1, d_temp_buffer_gas,
                                  gas_buffer_sizes.tempSizeInBytes, d_buffer_temp_output_gas_and_compacted_size,
                                  gas_buffer_sizes.outputSizeInBytes, &gas_handle, &emitProperty, 1 ) );

    CUDA_CHECK( cudaFree( (void*)d_temp_buffer_gas ) );

    size_t compacted_gas_size;
    CUDA_CHECK( cudaMemcpy( &compacted_gas_size, (void*)emitProperty.result, sizeof( size_t ), cudaMemcpyDeviceToHost ) );

    if( compacted_gas_size < gas_buffer_sizes.outputSizeInBytes )
    {
        CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_gas_output_buffer ), compacted_gas_size ) );

        // use handle as input and output
        OPTIX_CHECK( optixAccelCompact( state.context, 0, gas_handle, d_gas_output_buffer, compacted_gas_size, &gas_handle ) );

        CUDA_CHECK( cudaFree( (void*)d_buffer_temp_output_gas_and_compacted_size ) );
    }
    else
    {
        d_gas_output_buffer = d_buffer_temp_output_gas_and_compacted_size;
    }
}

void createGeometry( CallableProgramsState& state )
{
//
   // accel handling
   //
    OptixTraversableHandle gas_handle;
    CUdeviceptr            d_gas_output_buffer;
    {
        // Use default options for simplicity.  In a real use case we would want to
        // enable compaction, etc
        OptixAccelBuildOptions accel_options = {};
        //accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS;
        accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;

        // To ADD Triangles to be rendered simply add them to the vertices array and change the size of the array to match the new inputs 
        // Triangle build input: simple list of three vertices
        const std::array<float3, 6> vertices =
        { {
              { -1.0f, -1.0f, 0.0f },
              {  1.0f, -1.0f, 0.0f },
              { -1.0f,  1.0f, 0.0f },

              {  1.0f,  1.0f, 0.0f },
              { -1.0f,  1.0f, 0.0f },
              {  1.0f, -1.0f, 0.0f }
        } };

        const size_t vertices_size = sizeof(float3) * vertices.size();
        CUdeviceptr d_vertices = 0;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_vertices), vertices_size));
        CUDA_CHECK(cudaMemcpy(
            reinterpret_cast<void*>(d_vertices),
            vertices.data(),
            vertices_size,
            cudaMemcpyHostToDevice
        ));

        // Our build input is a simple list of non-indexed triangle vertices
        const uint32_t triangle_input_flags[1] = { OPTIX_GEOMETRY_FLAG_NONE };
        OptixBuildInput triangle_input = {};
        triangle_input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
        triangle_input.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
        triangle_input.triangleArray.numVertices = static_cast<uint32_t>(vertices.size());
        triangle_input.triangleArray.vertexBuffers = &d_vertices;
        triangle_input.triangleArray.flags = triangle_input_flags;
        triangle_input.triangleArray.numSbtRecords = 1;

        OptixAccelBufferSizes gas_buffer_sizes;
        OPTIX_CHECK(optixAccelComputeMemoryUsage(
            state.context,
            &accel_options,
            &triangle_input,
            1, // Number of build inputs
            &gas_buffer_sizes
        ));
        CUdeviceptr d_temp_buffer_gas;
        CUDA_CHECK(cudaMalloc(
            reinterpret_cast<void**>(&d_temp_buffer_gas),
            gas_buffer_sizes.tempSizeInBytes
        ));
        CUDA_CHECK(cudaMalloc(
            reinterpret_cast<void**>(&d_gas_output_buffer),
            gas_buffer_sizes.outputSizeInBytes
        ));

        OPTIX_CHECK(optixAccelBuild(
            state.context,
            0,                  // CUDA stream
            &accel_options,
            &triangle_input,
            1,                  // num build inputs
            d_temp_buffer_gas,
            gas_buffer_sizes.tempSizeInBytes,
            d_gas_output_buffer,
            gas_buffer_sizes.outputSizeInBytes,
            &state.gas_handle,
            nullptr,            // emitted property list
            0                   // num emitted properties
        ));

        // We can now free the scratch space buffer used during build and the vertex
        // inputs, since they are not needed by our trivial shading method
        CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_temp_buffer_gas)));
        CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_vertices)));
    }

}

void createModules( CallableProgramsState& state )
{
    OptixModuleCompileOptions module_compile_options = {};

    char   log[2048];
    size_t sizeof_log = sizeof( log );

    {
        size_t      inputSize = 0;
        const char* input     = sutil::getInputData( nullptr, nullptr, "whitted.cu", inputSize );
        OPTIX_CHECK_LOG( optixModuleCreateFromPTX( state.context, &module_compile_options, &state.pipeline_compile_options,
                                                   input, inputSize, log, &sizeof_log, &state.camera_module ) );
    }

    {
        size_t      inputSize = 0;
        const char* input     = sutil::getInputData( OPTIX_SAMPLE_NAME, OPTIX_SAMPLE_DIR, "optixLightFieldViewer.cu", inputSize );
        OPTIX_CHECK_LOG( optixModuleCreateFromPTX( state.context, &module_compile_options, &state.pipeline_compile_options,
                                                   input, inputSize, log, &sizeof_log, &state.shading_module ) );
    }
}

static void createCameraProgram( CallableProgramsState& state, std::vector<OptixProgramGroup>& program_groups )
{
    OptixProgramGroup        cam_prog_group;
    OptixProgramGroupOptions cam_prog_group_options = {};
    OptixProgramGroupDesc    cam_prog_group_desc    = {};
    cam_prog_group_desc.kind                        = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    cam_prog_group_desc.raygen.module               = state.camera_module;
    cam_prog_group_desc.raygen.entryFunctionName    = "__raygen__pinhole";

    char   log[2048];
    size_t sizeof_log = sizeof( log );
    OPTIX_CHECK_LOG( optixProgramGroupCreate( state.context, &cam_prog_group_desc, 1, &cam_prog_group_options, log,
                                              &sizeof_log, &cam_prog_group ) );

    program_groups.push_back( cam_prog_group );
    state.raygen_prog_group = cam_prog_group;
}

static void createHitProgram( CallableProgramsState& state, std::vector<OptixProgramGroup>& program_groups )
{
    OptixProgramGroup        hitgroup_prog_group;
    OptixProgramGroupOptions hitgroup_prog_group_options  = {};
    OptixProgramGroupDesc    hitgroup_prog_group_desc     = {};

    hitgroup_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    hitgroup_prog_group_desc.hitgroup.moduleCH = state.shading_module;
    hitgroup_prog_group_desc.hitgroup.entryFunctionNameCH = "__closesthit__ch1";

    char   log[2048];
    size_t sizeof_log = sizeof( log );
    OPTIX_CHECK_LOG( optixProgramGroupCreate( state.context, &hitgroup_prog_group_desc, 1, &hitgroup_prog_group_options,
                                              log, &sizeof_log, &hitgroup_prog_group ) );

    program_groups.push_back( hitgroup_prog_group );
    state.hitgroup_prog_group = hitgroup_prog_group;

}

static void createMissProgram( CallableProgramsState& state, std::vector<OptixProgramGroup>& program_groups )
{
    OptixProgramGroupOptions miss_prog_group_options = {};
    OptixProgramGroupDesc    miss_prog_group_desc    = {};
    miss_prog_group_desc.kind                        = OPTIX_PROGRAM_GROUP_KIND_MISS;
    miss_prog_group_desc.miss.module                 = state.shading_module;
    miss_prog_group_desc.miss.entryFunctionName      = "__miss__raydir_shade";

    char   log[2048];
    size_t sizeof_log = sizeof( log );
    OPTIX_CHECK_LOG( optixProgramGroupCreate( state.context, &miss_prog_group_desc, 1, &miss_prog_group_options, log,
                                              &sizeof_log, &state.miss_prog_group ) );

    program_groups.push_back( state.miss_prog_group );
}

void createPipeline( CallableProgramsState& state )
{
    const uint32_t max_trace_depth     = 1;
    const uint32_t max_cc_depth        = 1;
    const uint32_t max_dc_depth        = 1;
    const uint32_t max_traversal_depth = 1;

    std::vector<OptixProgramGroup> program_groups;

    state.pipeline_compile_options = {
        false,                                          // usesMotionBlur
        OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS,  // traversableGraphFlags
        whitted::NUM_PAYLOAD_VALUES,                    // numPayloadValues
        3,                                              // numAttributeValues
        OPTIX_EXCEPTION_FLAG_NONE,                      // exceptionFlags
        "params"                                       // pipelineLaunchParamsVariableName
    };
    state.pipeline_compile_options.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;


    // Prepare program groups
    createModules( state );
    createCameraProgram( state, program_groups );
    createHitProgram( state, program_groups );
    createMissProgram( state, program_groups );

    // Link program groups to pipeline
    OptixPipelineLinkOptions pipeline_link_options = {
        max_trace_depth,                // maxTraceDepth
        OPTIX_COMPILE_DEBUG_LEVEL_FULL  // debugLevel
    };
    char   log[2048];
    size_t sizeof_log = sizeof( log );
    OPTIX_CHECK_LOG( optixPipelineCreate( state.context, &state.pipeline_compile_options, &pipeline_link_options,
                                          program_groups.data(), static_cast<unsigned int>( program_groups.size() ),
                                          log, &sizeof_log, &state.pipeline ) );

    OptixStackSizes stack_sizes = {};
    for( auto& prog_group : program_groups )
    {
        OPTIX_CHECK( optixUtilAccumulateStackSizes( prog_group, &stack_sizes ) );
    }

    uint32_t direct_callable_stack_size_from_traversal;
    uint32_t direct_callable_stack_size_from_state;
    uint32_t continuation_stack_size;
    OPTIX_CHECK( optixUtilComputeStackSizes( &stack_sizes, max_trace_depth, max_cc_depth, max_dc_depth, &direct_callable_stack_size_from_traversal,
                                             &direct_callable_stack_size_from_state, &continuation_stack_size ) );
    OPTIX_CHECK( optixPipelineSetStackSize( state.pipeline, direct_callable_stack_size_from_traversal,
                                            direct_callable_stack_size_from_state, continuation_stack_size, max_traversal_depth ) );
}

void createSBT( CallableProgramsState& state )
{
    // Raygen program record
    {
        RayGenRecord raygen_record;
        OPTIX_CHECK( optixSbtRecordPackHeader( state.raygen_prog_group, &raygen_record ) );

        CUdeviceptr d_raygen_record;
        size_t      sizeof_raygen_record = sizeof( RayGenRecord );
        CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_raygen_record ), sizeof_raygen_record ) );
        CUDA_CHECK( cudaMemcpy( reinterpret_cast<void*>( d_raygen_record ), &raygen_record, sizeof_raygen_record, cudaMemcpyHostToDevice ) );

        state.sbt.raygenRecord = d_raygen_record;
    }

    // Miss program record
    {
        MissRecord miss_record;
        OPTIX_CHECK( optixSbtRecordPackHeader( state.miss_prog_group, &miss_record ) );

        CUdeviceptr d_miss_record;
        size_t      sizeof_miss_record = sizeof( MissRecord );
        CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_miss_record ), sizeof_miss_record ) );
        CUDA_CHECK( cudaMemcpy( reinterpret_cast<void*>( d_miss_record ), &miss_record, sizeof_miss_record, cudaMemcpyHostToDevice ) );

        state.sbt.missRecordBase          = d_miss_record;
        state.sbt.missRecordCount         = 1;
        state.sbt.missRecordStrideInBytes = static_cast<uint32_t>( sizeof_miss_record );
    }

    // Hitgroup program record
    {
        HitGroupRecord hitgroup_record;
        OPTIX_CHECK( optixSbtRecordPackHeader( state.hitgroup_prog_group, &hitgroup_record ) );
        hitgroup_record.data.color   = 2;

        
        createTexObject(state, config.imageName.c_str()); //MOVE THIS LATER

        hitgroup_record.data.widthInHogel = config.widthInHogels;
        hitgroup_record.data.heightInHogels = config.heightInHogels;
        hitgroup_record.data.fov = config.fov;
        hitgroup_record.data.texWidth = textureWidths[0];
        hitgroup_record.data.texHeight = textureHeights[0];
        hitgroup_record.data.tex = textureObjects[0];

        CUdeviceptr d_hitgroup_record;
        size_t      sizeof_hitgroup_record = sizeof( HitGroupRecord );
        CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_hitgroup_record ), sizeof_hitgroup_record ) );
        CUDA_CHECK( cudaMemcpy( reinterpret_cast<void*>( d_hitgroup_record ), &hitgroup_record, sizeof_hitgroup_record,
                                cudaMemcpyHostToDevice ) );

        state.sbt.hitgroupRecordBase          = d_hitgroup_record;
        state.sbt.hitgroupRecordCount         = 1;
        state.sbt.hitgroupRecordStrideInBytes = static_cast<uint32_t>( sizeof_hitgroup_record );
    }
}

static void context_log_cb( unsigned int level, const char* tag, const char* message, void* /*cbdata */ )
{
    std::cerr << "[" << std::setw( 2 ) << level << "][" << std::setw( 12 ) << tag << "]: " << message << "\n";
}

void createContext( CallableProgramsState& state )
{
    // Initialize CUDA
    CUDA_CHECK( cudaFree( 0 ) );

    OptixDeviceContext context;
    CUcontext          cuCtx = 0;  // zero means take the current context
    OPTIX_CHECK( optixInit() );
    OptixDeviceContextOptions options = {};
    options.logCallbackFunction       = &context_log_cb;
    options.logCallbackLevel          = 4;
    OPTIX_CHECK( optixDeviceContextCreate( cuCtx, &options, &context ) );

    state.context = context;
}

//
// Handle updates
//

void initCameraState()
{
    camera.setEye( make_float3( 0.0f, 0.0f, 3.0f ) );
    camera.setLookat( make_float3( 0.0f, 0.0f, 0.0f ) );
    camera.setUp( make_float3( 0.0f, 1.0f, 0.0f ) );
    camera.setFovY( 60.0f );
    camera_changed = true;

    trackball.setCamera( &camera );
    trackball.setMoveSpeed( 10.0f );
    trackball.setReferenceFrame( make_float3( 1.0f, 0.0f, 0.0f ), make_float3( 0.0f, 0.0f, 1.0f ), make_float3( 0.0f, 1.0f, 0.0f ) );
    trackball.setGimbalLock( true );
}

void handleCameraUpdate( CallableProgramsState& state )
{
    if( !camera_changed )
        return;
    camera_changed = false;

    camera.setAspectRatio( static_cast<float>( state.params.width ) / static_cast<float>( state.params.height ) );
    state.params.eye = camera.eye();
    camera.UVWFrame( state.params.U, state.params.V, state.params.W );
}

void handleResize( sutil::CUDAOutputBuffer<uchar4>& output_buffer,whitted::LaunchParams& params )
{
    if( !resize_dirty )
        return;
    resize_dirty = false;

    output_buffer.resize( params.width, params.height );

    // Realloc accumulation buffer
    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( params.accum_buffer ) ) );
    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &params.accum_buffer ), params.width * params.height * sizeof( float4 ) ) );
}

void updateState( sutil::CUDAOutputBuffer<uchar4>& output_buffer, CallableProgramsState& state )
{
    // Update params on device
    if( camera_changed || resize_dirty || shading_changed )
        state.params.subframe_index = 0;

    handleCameraUpdate( state );
    handleResize( output_buffer, state.params );
}

void launchSubframe( sutil::CUDAOutputBuffer<uchar4>& output_buffer, CallableProgramsState& state )
{

    // Launch
    uchar4* result_buffer_data = output_buffer.map();
    state.params.frame_buffer  = result_buffer_data;
    
    CUDA_CHECK( cudaMemcpyAsync( reinterpret_cast<void*>( state.d_params ), &state.params,
                                 sizeof(whitted::LaunchParams ), cudaMemcpyHostToDevice, state.stream ) );
    
    OPTIX_CHECK( optixLaunch( state.pipeline, state.stream, reinterpret_cast<CUdeviceptr>( state.d_params ),
                              sizeof(whitted::LaunchParams ), &state.sbt,
                              state.params.width,   // launch width
                              state.params.height,  // launch height
                              1                     // launch depth
                              ) );

    output_buffer.unmap();
    CUDA_SYNC_CHECK();
}


void displaySubframe( sutil::CUDAOutputBuffer<uchar4>& output_buffer, sutil::GLDisplay& gl_display, GLFWwindow* window )
{
    // Display
    int framebuf_res_x = 0;  // The display's resolution (could be HDPI res)
    int framebuf_res_y = 0;  //
    glfwGetFramebufferSize( window, &framebuf_res_x, &framebuf_res_y );
    gl_display.display( output_buffer.width(), output_buffer.height(), framebuf_res_x, framebuf_res_y, output_buffer.getPBO() );
}


void cleanupState( CallableProgramsState& state )
{
    OPTIX_CHECK( optixPipelineDestroy( state.pipeline ) );
    OPTIX_CHECK( optixProgramGroupDestroy( state.raygen_prog_group ) );
    OPTIX_CHECK( optixProgramGroupDestroy( state.hitgroup_prog_group ) );
    OPTIX_CHECK( optixProgramGroupDestroy( state.miss_prog_group ) );

    OPTIX_CHECK( optixModuleDestroy( state.shading_module ) );
    OPTIX_CHECK( optixModuleDestroy( state.camera_module ) );
    OPTIX_CHECK( optixDeviceContextDestroy( state.context ) );


    CUDA_CHECK( cudaStreamDestroy( state.stream ) );
    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( state.sbt.raygenRecord ) ) );
    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( state.sbt.missRecordBase ) ) );
    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( state.sbt.hitgroupRecordBase ) ) );
    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( state.d_gas_output_buffer ) ) );
    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( state.params.accum_buffer ) ) );
    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( state.params.lights.data ) ) );
    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( state.d_params ) ) );
}

int main( int argc, char* argv[] )
{
    CallableProgramsState state;
    state.params.width                             = 768;
    state.params.height                            = 768;
    sutil::CUDAOutputBufferType output_buffer_type = sutil::CUDAOutputBufferType::GL_INTEROP;

    //
    // Parse command line options
    //
    std::string outfile;

    for( int i = 1; i < argc; ++i )
    {
        const std::string arg = argv[i];
        if( arg == "--help" || arg == "-h" )
        {
            printUsageAndExit( argv[0] );
        }
        else if( arg == "--no-gl-interop" )
        {
            output_buffer_type = sutil::CUDAOutputBufferType::CUDA_DEVICE;
        }
        else if( arg == "--file" || arg == "-f" )
        {
            if( i >= argc - 1 )
                printUsageAndExit( argv[0] );
            outfile = argv[++i];
        }
        else if( arg.substr( 0, 6 ) == "--dim=" )
        {
            const std::string dims_arg = arg.substr( 6 );
            int               w, h;
            sutil::parseDimensions( dims_arg.c_str(), w, h );
            state.params.width  = w;
            state.params.height = h;
        }
        else
        {
            std::cerr << "Unknown option '" << argv[i] << "'\n";
            printUsageAndExit( argv[0] );
        }
    }

    try
    {
        initCameraState();

       // setConfig("Config.txt");
        setConfig("Config.txt");
        //
        // Set up OptiX state
        //
        createContext( state );
        createGeometry( state );
        createPipeline( state );
        createSBT( state );

        initLaunchParams( state );

        //
        // Render loop
        //
        if( outfile.empty() )
        {
            GLFWwindow* window = sutil::initUI( "Real Time Lightfield Render", state.params.width, state.params.height );
           
                glfwSetCursorPosCallback(window, cursorPosCallback);
                glfwSetMouseButtonCallback(window, mouseButtonCallback);
                glfwSetWindowSizeCallback(window, windowSizeCallback);
                glfwSetWindowIconifyCallback(window, windowIconifyCallback);
                glfwSetKeyCallback(window, keyCallback);
                glfwSetScrollCallback(window, scrollCallback);
                glfwSetWindowUserPointer(window, &state.params);


            {
                // output_buffer needs to be destroyed before cleanupUI is called
                sutil::CUDAOutputBuffer<uchar4> output_buffer( output_buffer_type, state.params.width, state.params.height );

                output_buffer.setStream( state.stream );
                sutil::GLDisplay gl_display;

                std::chrono::duration<double> state_update_time( 0.0 );
                std::chrono::duration<double> render_time( 0.0 );
                std::chrono::duration<double> display_time( 0.0 );

                do
                {

                    auto t0 = std::chrono::steady_clock::now();
                    
                    glfwPollEvents();
             
                    updateState( output_buffer, state );
                    auto t1 = std::chrono::steady_clock::now();
                    state_update_time += t1 - t0;
                    t0 = t1;

                    launchSubframe( output_buffer, state );
                    t1 = std::chrono::steady_clock::now();
                    render_time += t1 - t0;
                    t0 = t1;

                    displaySubframe( output_buffer, gl_display, window );
                    t1 = std::chrono::steady_clock::now();
                    display_time += t1 - t0;
                   
                    sutil::displayStats( state_update_time, render_time, display_time );

                    
                    bool changeState = sutil::getChangeState();
                    
                    if (changeState) 
                    {
                        std::string filePath = sutil::getCurrFilename();
                        setConfig(filePath);

                        createSBT(state);
                        initLaunchParams(state);

                    }
                  
                    
                    glfwSwapBuffers( window );

                   

                    ++state.params.subframe_index;
                    

                    
                } while( !glfwWindowShouldClose( window ) );
            }
            sutil::cleanupUI( window );
            
            
        }
        else
        {
            if( output_buffer_type == sutil::CUDAOutputBufferType::GL_INTEROP )
            {
                sutil::initGLFW();  // For GL context
                sutil::initGL();
            }

            sutil::CUDAOutputBuffer<uchar4> output_buffer( output_buffer_type, state.params.width, state.params.height );

            handleCameraUpdate( state );
            handleResize( output_buffer, state.params );
            launchSubframe( output_buffer, state );

            sutil::ImageBuffer buffer;
            buffer.data         = output_buffer.getHostPointer();
            buffer.width        = output_buffer.width();
            buffer.height       = output_buffer.height();
            buffer.pixel_format = sutil::BufferImageFormat::UNSIGNED_BYTE4;
            sutil::saveImage( outfile.c_str(), buffer, false );

            if( output_buffer_type == sutil::CUDAOutputBufferType::GL_INTEROP )
            {
                glfwTerminate();
            }
        }

        cleanupState( state );
    }
    catch( std::exception& e )
    {
        std::cerr << "Caught exception: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
