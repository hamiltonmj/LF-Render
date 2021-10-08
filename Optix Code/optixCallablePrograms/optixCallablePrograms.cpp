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

#include "lodepng.h"

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

#include <cuda/whitted.h>

#include "optixCallablePrograms.h"

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

    if( action == GLFW_PRESS )
    {
        mouse_button = button;
        trackball.startTracking( static_cast<int>( xpos ), static_cast<int>( ypos ) );
    }
    else
    {
        mouse_button = -1;
    }
}


static void cursorPosCallback( GLFWwindow* window, double xpos, double ypos )
{
   whitted::LaunchParams* params = static_cast<whitted::LaunchParams*>( glfwGetWindowUserPointer( window ) );

    if( mouse_button == GLFW_MOUSE_BUTTON_LEFT )
    {
        trackball.setViewMode( sutil::Trackball::LookAtFixed );
        trackball.updateTracking( static_cast<int>( xpos ), static_cast<int>( ypos ), params->width, params->height );
        camera_changed = true;
    }
    else if( mouse_button == GLFW_MOUSE_BUTTON_RIGHT )
    {
        trackball.setViewMode( sutil::Trackball::EyeFixed );
        trackball.updateTracking( static_cast<int>( xpos ), static_cast<int>( ypos ), params->width, params->height );
        camera_changed = true;
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

    /*
    if (mods == GLFW_MOD_SHIFT)
    {
        shifted = 2;
    }
    else
    {
        shifted = 1;
    }
    */
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

std::vector<std::vector<float4>> convertTo2dMatrix(const std::vector<unsigned char>& image, unsigned width, unsigned height)
{
    std::vector<std::vector<float4>> outImage(width, std::vector<float4>(height));
    size_t imagePos = 0;
    for (size_t y = 0; y < width; y++)
    {
        for (size_t x = 0; x < height; x++)
        {
            outImage[x][y].x = (float) image[imagePos]     / 255;
            outImage[x][y].y = (float) image[imagePos + 1] / 255;
            outImage[x][y].z = (float) image[imagePos + 2] / 255;
            outImage[x][y].w = (float) image[imagePos + 3] /255;
            imagePos += 4;
        }
    }
    return outImage;
}

float4Array convertToFloat4Vec(const std::vector<unsigned char>& image, unsigned width, unsigned height)
{
    float4Array outImage;
    outImage.data = new float4[width * height];
    outImage.size = width * height;
  //  std::vector<float4> outImage((int)width * height, float4());
    size_t imagePos = 0;
    for (size_t i = 0; i < (int)width * height; i++)
    {
            outImage.data[i].x = (float)image[imagePos] / 255;
            outImage.data[i].y = (float)image[imagePos + 1] / 255;
            outImage.data[i].z = (float)image[imagePos + 2] / 255;
            outImage.data[i].w = (float)image[imagePos + 3] / 255;

            imagePos += 4;
    }
    return outImage;
}

FloatArray convertToFloatArray(const std::vector<unsigned char>& image, unsigned width, unsigned height)
{
    FloatArray outImage;
    outImage.data = (float *) malloc(sizeof(float) * width * height);
    outImage.size = width * height;
    size_t imagePos = 0;
    for (size_t i = 0; i < (int)width * height; i++)
    {
        outImage.data[i] = (float)(image[i] );
    }
    return outImage;
}


void setConfig(std::string fileName)
{
    std::ifstream file;
    file.open(fileName);
    if (!file.is_open())
    {
        std::cout <<"Error Opening config\n";
        exit(-1);
    }
    std::string ab;
   // const char a[];
    file >> config.imageName >> config.widthInHogels >> config.heightInHogels >> config.fov;

    //std::strcpy(config.imageName, ab.c_str());
    //config.imageName = new const char[ab.size()];

    std::cout << config.imageName << "   HIOGHAWIGHIOAWHGIAHWGHW(Ag\n";
}

void initlightfieldPparameters(CallableProgramsState& state)
{


    std::vector<unsigned char> image;
    unsigned width;
    unsigned height;
    //const char* filename = "inputData/HugeMultiColor.png";

   // unsigned error = lodepng::decode(image, width, height, filename);
    //std::cout << "error:" << lodepng_error_text(error) <<" || Image.length:" << image.size() << " || width: " << width << " height: " << height << "\n";

    std::cout << "r:" <<(unsigned) image[0] << "g:" << (unsigned)image[1] << "b:" << (unsigned)image[2] << "\n";

    unsigned int size = width * height * sizeof(float) * 4;

    



    //Allocates device memory for size of inputted image
   // float* dData = NULL;
   // CUDA_CHECK(cudaMalloc( (void **) dData, size));
    
    int count = 0;
    float4Array a = convertToFloat4Vec(image, width, height);
    std::vector<std::vector<float4>> image2d = convertTo2dMatrix(image, width, height);
/*
    for (int i = 0; i < a.size; i++)
    {
        std::cout << "ByLINE r:" << a.data[i].x << "g:" << a.data[i].y << "b:" << a.data[i].z << "\n";
        count++;
    }
    std::cout << count << "number of pixels \n";
*/
    float4* imagePointer = a.data;

    /*
    cudaChannelFormatDesc channelDesc= cudaCreateChannelDesc<float4>();
    cudaArray *cuArray;
    CUDA_CHECK(cudaMallocArray(&cuArray, &channelDesc, width, height));

    CUDA_CHECK(cudaMemcpyToArray(cuArray, 0, 0, imagePointer, size, cudaMemcpyHostToDevice));
    */
    //FloatArray floatImage = convertToFloatArray(image, width * 4, height);

    float* floatImage;
    floatImage = (float*)malloc(sizeof(float) * width * height *4);
    size_t imagePos = 0;
    for (size_t i = 0; i < (int)width * height *4; i++)
    {
        floatImage[i] = (float)(image[i]);
    }

    for (size_t i = 0; i < size; i += 4)
    {
        std::cout << "ByPixel r:" << floatImage[i] << "g:" << floatImage[i+1] << "b:" << floatImage[i+2] << "\n";
    }

    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
    cudaArray *cuArray;
    CUDA_CHECK(cudaMallocArray(&cuArray, &channelDesc, width *4, height));

    const size_t spitch = width * 4 * sizeof(float);
    CUDA_CHECK(cudaMemcpy2DToArray(cuArray, 0, 0, floatImage, spitch, width * 4 * sizeof(float), height, cudaMemcpyHostToDevice));

    //CUDA_CHECK(cudaMemcpyToArray(cuArray, 0, 0, floatImage, size, cudaMemcpyHostToDevice));




    
    cudaTextureDesc texDescr;
   // memset(&texDescr, 0, sizeof(cudaTextureDesc));
    texDescr.normalizedCoords = 1;
    texDescr.filterMode = cudaFilterModeLinear;
    texDescr.addressMode[0] = cudaAddressModeWrap;
    texDescr.addressMode[1] = cudaAddressModeWrap;
    texDescr.readMode = cudaReadModeElementType;

    cudaResourceDesc texRes;
    //memset(&texRes, 0, sizeof(cudaResourceDesc));
    texRes.resType = cudaResourceTypeArray;    
    texRes.res.array.array = cuArray;

    //state.params.tex = 0;
    //cudaTextureObject_t* localTex = (cudaTextureObject_t*) malloc(sizeof(cudaTextureObject_t));
   // state.params.tex = new cudaTextureObject_t;

    CUDA_CHECK(cudaCreateTextureObject(&state.params.tex, &texRes, &texDescr, NULL));
    // = &tex;

   // CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&state.params.tex), sizeof(float3)));
    //CUDA_CHECK(cudaMemcpy(&state.params.tex, localTex, sizeof(cudaUserObject_t), cudaMemcpyHostToDevice));



//    cudaThreadSynchronize();

    //std::cout << "2d Array r:" << image2d[15][0].x << "g:" << image2d[15][0].y << "b:" << image2d[15][0].z << "\n";

    float3* tColor1Pointer = new float3(make_float3(image2d[15][0]));
    float3* tColor2Pointer = new float3(make_float3(0.01f, 0.01f, 0.9f));
//    tColor1Pointer = 360 , 400
//    tColor1Pointer = 

   // float4 abc = tex2D<float4>(*params.texture, 18.0f, 18.0f);

    //std::cout << abc.x << ;;
    //std::cout << "texReturn r:" << abc.x << "g:" << abc.y << "b:" << abc.z << "\n";
    //tex2d
    //tex2D(*state.params.texture, 0.0f, 0.0f);



    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&state.params.testColor1), sizeof(float3)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&state.params.testColor2), sizeof(float3)));
   // CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&state.params.tex), sizeof(cudaTextureObject_t)));
    
    CUDA_CHECK(cudaMemcpy(state.params.testColor1, tColor1Pointer, sizeof(float3), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(state.params.testColor2, tColor2Pointer, sizeof(float3), cudaMemcpyHostToDevice));
 //   CUDA_CHECK(cudaMemcpy(state.params.tex, tex, sizeof(cudaTextureObject_t), cudaMemcpyHostToDevice));

    delete(tColor1Pointer);
    delete(tColor2Pointer);

    //TODO :: WE need to allocate memory onto the graphics card initially as the same size as the image we load onto device memory. 
    // so we allocate memory onto both host memory and device memory then copy over the details then use the device pointer as an launchParameter
    // since the resize modifies the init Params going to the device need to make sure its not going to reallocate the memory
    //
    //
    //  Step 1: load image onto host and get into fastest format that can be used (this is a problem within itself). This may involve decompression or other methods
    //  Step 2: allocate Device memory based on size of host memory from the above step
    //  step 3: load image onto device memory and potentially free the space then on the host memory
    //  step 4: within Device now access the image and display the results.


     /*
        Appears we will first malloc memory on the host then cudamalloc onto the graphics card and retrieve the pointer to the graphics card memory. Once allocated on the
        grapghics card we should also deallocate the memory on the host. After this we need to add it to the parameter struct which is used to define the pipeline
        parameters used within the optix program, from there we need to tell the launch command to use our new parameters which we created above. This should in theory
        allow us to then call a params.imageDataPointer or whatever the variable is called and retrieve the device pointer of the image data. From there we should be
        able to directly acces the loaded data on the graphics card,  this process may also change so we dont get a pointer to the default data but maybe to the texture
        object we create to better access the data. At the end we will then need to destroy our texture object and de-allocate the memory from the graphics card.
     */
     

         CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&state.params.accum_buffer),
             state.params.width * state.params.height * sizeof(float4)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&state.d_params), sizeof(whitted::LaunchParams)));
}

/*
    TODO:
        Look through new functions to remove unused bits, deallocate host side memory usage of textures once finished loading them
        Look at not doubleing image array when processing on device 

        Figure out bary-centric coordinates, or figure out how to get hit coordinates system another way.

        Understand how records work to be able to create multiple lightfield at once

        can we use normalized coordinates with barycentric coordinates to solve our problem?

        Begin remembering/solving how to actually use interpolate our rays into the lightfield

        add timer functions within texture loading to better communicate its not crashing

        add ability for multiple lightfields,  work partly done by creating an array of textures and using the hit record to access them.
        */

void createTexObject(CallableProgramsState& state, const char* filename)
{
    int numOfTex = 1;
    std::vector<unsigned char> image;

    textureArrays.resize(textureArrays.size() + numOfTex);
    textureObjects.resize(textureObjects.size() + numOfTex);
    textureWidths.resize(textureWidths.size() + numOfTex);
    textureHeights.resize(textureHeights.size() + numOfTex);

    unsigned width;
    unsigned height;
    std::cout << filename << "   :FILENAME\n";
    unsigned error = lodepng::decode(image, width, height, filename);
    std::cout << "error:" << lodepng_error_text(error) << " || Image.length:" << image.size() << " || width: " << width << " height: " << height << "\n";

    unsigned int size = width * height * sizeof(float) * 4;

    int count = 0;
    std::vector<std::vector<float4>> image2d = convertTo2dMatrix(image, width, height);

    float* floatImage;
    floatImage = (float*)malloc(sizeof(float) * width * height * 4);
    size_t imagePos = 0;
    for (size_t i = 0; i < (int)width * height * 4; i++)
    {
        floatImage[i] = (float)(image[i]);
    }

   // for (size_t i = 0; i < size; i += 4)
    //{
     //   std::cout << "ByPixel r:" << floatImage[i] << "g:" << floatImage[i + 1] << "b:" << floatImage[i + 2] << "\n";
   // }

    float4Array imageFloat4 = convertToFloat4Vec(image, width, height);

    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float4>(); //(32, 0, 0, 0, cudaChannelFormatKindFloat);
    cudaArray_t &cuArray = textureArrays[0];
    CUDA_CHECK(cudaMallocArray(&cuArray, &channelDesc, width, height));

    const size_t spitch = width  * sizeof(float4);
    CUDA_CHECK(cudaMemcpy2DToArray(cuArray, 0, 0, imageFloat4.data, spitch, width * sizeof(float4), height, cudaMemcpyHostToDevice));

    //CUDA_CHECK(cudaMemcpyToArray(cuArray, 0, 0, floatImage, size, cudaMemcpyHostToDevice));

    cudaTextureDesc texDescr;
    memset(&texDescr, 0, sizeof(cudaTextureDesc));
    texDescr.normalizedCoords = 0;
    texDescr.filterMode = cudaFilterModeLinear;
    texDescr.addressMode[0] = cudaAddressModeWrap;
    texDescr.addressMode[1] = cudaAddressModeWrap;
    texDescr.readMode = cudaReadModeElementType;

    cudaResourceDesc texRes;
    memset(&texRes, 0, sizeof(cudaResourceDesc));
    texRes.resType = cudaResourceTypeArray;
    texRes.res.array.array = cuArray;

    cudaTextureObject_t tex = 0;

    CUDA_CHECK(cudaCreateTextureObject(&tex, &texRes, &texDescr, NULL));
    textureObjects[0] = tex;
    textureWidths[0] = width;
    textureHeights[0] = height;
}





void initLaunchParams( CallableProgramsState& state )
{
    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &state.params.accum_buffer ),
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

//TODO :: WE need to allocate memory onto the graphics card initially as the same size as the image we load onto device memory. 
// so we allocate memory onto both host memory and device memory then copy over the details then use the device pointer as an launchParameter
// since the resize modifies the init Params going to the device need to make sure its not going to reallocate the memory
//
//
//  Step 1: load image onto host and get into fastest format that can be used (this is a problem within itself). This may involve decompression or other methods
//  Step 2: allocate Device memory based on size of host memory from the above step
//  step 3: load image onto device memory and potentially free the space then on the host memory
//  step 4: within Device now access the image and display the results.


 /*
    Appears we will first malloc memory on the host then cudamalloc onto the graphics card and retrieve the pointer to the graphics card memory. Once allocated on the 
    grapghics card we should also deallocate the memory on the host. After this we need to add it to the parameter struct which is used to define the pipeline 
    parameters used within the optix program, from there we need to tell the launch command to use our new parameters which we created above. This should in theory 
    allow us to then call a params.imageDataPointer or whatever the variable is called and retrieve the device pointer of the image data. From there we should be 
    able to directly acces the loaded data on the graphics card,  this process may also change so we dont get a pointer to the default data but maybe to the texture 
    object we create to better access the data. At the end we will then need to destroy our texture object and de-allocate the memory from the graphics card.
    
 */

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
        accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS;
        accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;



        // To ADD Triangles to be rendered simply add them to the vertices array and change the size of the array to match the new inputs 


        // Triangle build input: simple list of three vertices
        const std::array<float3, 6> vertices =
        { {
              { -0.5f, -0.5f, 0.0f },
              {  0.5f, -0.5f, 0.0f },
              { -0.5f,  0.5f, 0.0f },

              {  0.5f,  0.5f, 0.0f },
              { -0.5f,  0.5f, 0.0f },
              {  0.5f, -0.5f, 0.0f }
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
        const char* input     = sutil::getInputData( OPTIX_SAMPLE_NAME, OPTIX_SAMPLE_DIR, "optixCallablePrograms.cu", inputSize );
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
    //hitgroup_prog_group_desc.hitgroup.moduleAH = nullptr;
    //hitgroup_prog_group_desc.hitgroup.entryFunctionNameAH = nullptr;

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
    camera.setEye( make_float3( 0.0f, 0.0f, -3.0f ) );
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


    //CUDA_CHECK(cudaFree(reinterpret_cast<void*>(state.params.testColor1)));
    //CUDA_CHECK(cudaFree(reinterpret_cast<void*>(state.params.testColor2)));


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

        setConfig("Config.txt");

        //
        // Set up OptiX state
        //
        createContext( state );
        createGeometry( state );
        createPipeline( state );
        createSBT( state );

        //initlightfieldPparameters(state);
        initLaunchParams( state );

        //
        // Render loop
        //
        if( outfile.empty() )
        {
            GLFWwindow* window = sutil::initUI( "optixCallablePrograms", state.params.width, state.params.height );
            glfwSetMouseButtonCallback( window, mouseButtonCallback );
            glfwSetCursorPosCallback( window, cursorPosCallback );
            glfwSetWindowSizeCallback( window, windowSizeCallback );
            glfwSetWindowIconifyCallback( window, windowIconifyCallback );
            glfwSetKeyCallback( window, keyCallback );
            glfwSetScrollCallback( window, scrollCallback );
            glfwSetWindowUserPointer( window, &state.params );

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
