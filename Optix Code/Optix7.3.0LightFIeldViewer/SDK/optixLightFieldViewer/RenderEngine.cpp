//#include <glad/glad.h>  // Needs to be included before gl_interop

#include "opencv2/core.hpp" 
#include "opencv2/opencv.hpp"

//#include <cuda_gl_interop.h>
#include <cuda_runtime.h>

#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stack_size.h>
#include <optix_stubs.h>

#include <sutil/CUDAOutputBuffer.h>
#include <sutil/Camera.h>
#include <sutil/Exception.h>

#include <cuda/whitted.h>
#include <fstream>

#include "TextureDataTypes.h"
#include "RecordData.h"
#include "RenderEngine.h"

#include <filesystem.>

RenderEngine::RenderEngine(size_t w, size_t h)
    :m_state(RenderState()), m_output_buffer(sutil::CUDAOutputBufferType::CUDA_DEVICE, w,h)
  //  :m_state(), m_output_buffer(sutil::CUDAOutputBufferType::GL_INTEROP, w, h)
{
   // m_state = ;
    m_state.params.width = w;
    m_state.params.height = h;
}

RenderEngine::RenderEngine()
    :m_state(RenderState()), m_output_buffer(sutil::CUDAOutputBufferType::CUDA_DEVICE, 768, 768)
//    :m_state(), m_output_buffer(sutil::CUDAOutputBufferType::GL_INTEROP, 768, 768)
{
   // m_state = RenderState();
    m_state.params.width = 768;
    m_state.params.height = 768;
}

void RenderEngine::handleResize(size_t width, size_t height)
{
    setDisplayDimensions(width, height);
    //m_state.params.subframe_index = 0;

    m_output_buffer.resize(m_state.params.width, m_state.params.height);

    // Realloc accumulation buffer
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(m_state.params.accum_buffer)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&m_state.params.accum_buffer), m_state.params.width * m_state.params.height * sizeof(float4)));
}

/// <summary>
/// Given a filename will generate a texture based on it, if given a .txt file will generate a light field given the parameters provided in the file, if given a 
/// png will load the png image as a texture 
/// </summary>
/// <param name="fileName"></param>
void RenderEngine::loadTexture(std::string fileName)
{
    //if file selected is .txt file, then directly load the file.
    std::shared_ptr<TextureBase> tex;
    if (RenderEngine::is_textFile(fileName)) {
        std::ifstream file;
        file.open(fileName);
        if (!file.is_open())
        {
            std::cout << "Error Opening Lightfield Config: " << fileName << "\n";
            exit(-1);
        }
        std::string name;
        unsigned width, height, fov;
        file >> name >> width >> height >> fov;

        
        if (name == "!VIDEO!")
        {
            size_t pathEnd = fileName.rfind("\\");
            pathEnd = (std::string::npos == pathEnd) ? fileName.rfind("/") : pathEnd;
                
            fileName = fileName.substr(0, pathEnd);
            std::vector<cv::String> framePaths;
            cv::glob(fileName, framePaths, false);

            for (size_t x = 0; x < framePaths.size(); x++)
            {
                cv::String imageName = framePaths[x];
                if (RenderEngine::is_textFile(imageName)) continue;

                std::cout << "Opening Texture As LightField: " << imageName << "\n";
                tex = std::make_shared<LightFieldData>(loadImageToRam(imageName), width, height, fov);

                    
                m_state.texObjects.insert({ "f" + std::to_string(x),tex});
                std::cout << "hi";
            }
        }
        else
        {
            std::cout << "Opening Texture As LightField: " << fileName << "\n";
            tex = std::make_shared<LightFieldData>(loadImageToRam(name), width, height, fov);
            m_state.texObjects.insert({ fileName,tex });
        }
    }
    else
    {
        std::cout << "Opening Texture As Image: " << fileName << "\n";
        tex = std::make_shared<TextureData>(loadImageToRam(fileName));
        m_state.texObjects.insert({ fileName,tex });
    }

    //TODO: ADD MULTI TEXTURE SUPPORT
    //This will need to be modified once more txtures are added, but also a better referning system will need to be implemented to better control how textures are refernced
    //per face/ object 
    //IE: a factory could be good here
}


void RenderEngine::initLaunchParams()
{
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&m_state.params.accum_buffer),
        m_state.params.width * m_state.params.height * sizeof(float4)));
    m_state.params.frame_buffer = nullptr;  // Will be set when output buffer is mapped

    m_state.params.subframe_index = 0u;

    // Set ambient light color and point light position
    std::vector<Light> lights(2);
    lights[0].type = Light::Type::AMBIENT;
    lights[0].ambient.color = make_float3(0.4f, 0.4f, 0.4f);
    lights[1].type = Light::Type::POINT;
    lights[1].point.color = make_float3(1.0f, 1.0f, 1.0f);
    lights[1].point.intensity = 1.0f;
    lights[1].point.position = make_float3(10.0f, 10.0f, -10.0f);
    lights[1].point.falloff = Light::Falloff::QUADRATIC;

    m_state.params.lights.count = static_cast<unsigned int>(lights.size());
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&m_state.params.lights.data), lights.size() * sizeof(Light)));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(m_state.params.lights.data), lights.data(),
        lights.size() * sizeof(Light), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaStreamCreate(&m_state.stream));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&m_state.d_params), sizeof(whitted::LaunchParams)));

    m_state.params.handle = m_state.gas_handle;

//    m_output_buffer = sutil::CUDAOutputBuffer<uchar4>(outBuffType, m_state.params.width, m_state.params.height);
//    m_output_buffer.setStream(m_state.stream);

}

void RenderEngine::handleCameraUpdate(sutil::Camera* cam)
{
    m_state.params.subframe_index = 0;
    cam->setAspectRatio(static_cast<float>(m_state.params.width) / static_cast<float>(m_state.params.height));
    auto x  = cam->eye();
    m_state.params.eye = cam->eye();
    cam->UVWFrame(m_state.params.U, m_state.params.V, m_state.params.W);
}

void RenderEngine::handleResize()
{
    //m_state.params.subframe_index = 0;
    m_output_buffer.resize(m_state.params.width, m_state.params.height);

    // Realloc accumulation buffer
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(m_state.params.accum_buffer)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&m_state.params.accum_buffer), m_state.params.width * m_state.params.height * sizeof(float4)));
}

void RenderEngine::buildGas( const OptixAccelBuildOptions& accel_options,
    const OptixBuildInput& build_input,
    OptixTraversableHandle& gas_handle,
    CUdeviceptr& d_gas_output_buffer)
{
    OptixAccelBufferSizes gas_buffer_sizes;
    CUdeviceptr           d_temp_buffer_gas;

    OPTIX_CHECK(optixAccelComputeMemoryUsage(m_state.context, &accel_options, &build_input, 1, &gas_buffer_sizes));

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_temp_buffer_gas), gas_buffer_sizes.tempSizeInBytes));

    // non-compacted output and size of compacted GAS
    CUdeviceptr d_buffer_temp_output_gas_and_compacted_size;
    size_t      compactedSizeOffset = roundUp<size_t>(gas_buffer_sizes.outputSizeInBytes, 8ull);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_buffer_temp_output_gas_and_compacted_size), compactedSizeOffset + 8));

    OptixAccelEmitDesc emitProperty = {};
    emitProperty.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
    emitProperty.result = (CUdeviceptr)((char*)d_buffer_temp_output_gas_and_compacted_size + compactedSizeOffset);

    OPTIX_CHECK(optixAccelBuild(m_state.context, 0, &accel_options, &build_input, 1, d_temp_buffer_gas,
        gas_buffer_sizes.tempSizeInBytes, d_buffer_temp_output_gas_and_compacted_size,
        gas_buffer_sizes.outputSizeInBytes, &gas_handle, &emitProperty, 1));

    CUDA_CHECK(cudaFree((void*)d_temp_buffer_gas));

    size_t compacted_gas_size;
    CUDA_CHECK(cudaMemcpy(&compacted_gas_size, (void*)emitProperty.result, sizeof(size_t), cudaMemcpyDeviceToHost));

    if (compacted_gas_size < gas_buffer_sizes.outputSizeInBytes)
    {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_gas_output_buffer), compacted_gas_size));

        // use handle as input and output
        OPTIX_CHECK(optixAccelCompact(m_state.context, 0, gas_handle, d_gas_output_buffer, compacted_gas_size, &gas_handle));

        CUDA_CHECK(cudaFree((void*)d_buffer_temp_output_gas_and_compacted_size));
    }
    else
    {
        d_gas_output_buffer = d_buffer_temp_output_gas_and_compacted_size;
    }
}

void RenderEngine::createGeometry()
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
         /*{ -1.0f, -1.0f , -3.0f },
                {  1.0f, -1.0f , -3.0f },
                { -1.0f,  1.0f , -3.0f},

                {  1.0f,  1.0f , -3.0f },
                { -1.0,  1.0f  , -3.0f },
                {  1.0f, -1.0f , -3.0f }*/
        const std::array<float3, 15> vertices =
        { {
               
                  { -1.0f / 10, -1.0f / 10 , -6.0f / 10 },
                  {  1.0f / 10, -1.0f / 10 , -6.0f / 10},
                  { -1.0f / 10,  1.0f / 10 , -6.0f / 10},

                  {  1.0f / 10,  1.0f / 10 , -6.0f / 10},
                  { -1.0 / 10,  1.0f / 10  , -6.0f / 10},
                  {  1.0f / 10, -1.0f / 10 , -6.0f / 10},
       
                //refrence triangle
                // left
                { -1.0f ,  0.0f , 0.0f },
                { -1.0f  ,  1.0f , 0.0f},
                {  -1.0f , 0.0f , 1.0f},

                // right
                {  1.0f ,  0.0f , 0.0f },
                {   1.0f,  1.0f , 0.0f},
                {   1.0f , 0.0f , 1.0f},

                // back
               {  0.0f ,  0.0f , 1.0f},
               { 1.0f  ,  0.0f ,  1.0f},
               {  0.0f , 1.0f ,  1.0f}

            } 
        

        };

        
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
            m_state.context,
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
            m_state.context,
            0,                  // CUDA stream
            &accel_options,
            &triangle_input,
            1,                  // num build inputs
            d_temp_buffer_gas,
            gas_buffer_sizes.tempSizeInBytes,
            d_gas_output_buffer,
            gas_buffer_sizes.outputSizeInBytes,
            &m_state.gas_handle,
            nullptr,            // emitted property list
            0                   // num emitted properties
        ));

        // We can now free the scratch space buffer used during build and the vertex
        // inputs, since they are not needed by our trivial shading method
        CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_temp_buffer_gas)));
        CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_vertices)));
    }

}

void RenderEngine::createModules()
{
    OptixModuleCompileOptions module_compile_options = {};

    char   log[2048];
    size_t sizeof_log = sizeof(log);

    {
        size_t      inputSize = 0;
        const char* input = sutil::getInputData(nullptr, nullptr, "whitted.cu", inputSize);
        OPTIX_CHECK_LOG(optixModuleCreateFromPTX(m_state.context, &module_compile_options, &m_state.pipeline_compile_options,
            input, inputSize, log, &sizeof_log, &m_state.camera_module));
    }

    {
        size_t      inputSize = 0;
        const char* input = sutil::getInputData(OPTIX_SAMPLE_NAME, OPTIX_SAMPLE_DIR, "optixLightFieldViewer.cu", inputSize);
        OPTIX_CHECK_LOG(optixModuleCreateFromPTX(m_state.context, &module_compile_options, &m_state.pipeline_compile_options,
            input, inputSize, log, &sizeof_log, &m_state.shading_module));
    }
}

void RenderEngine::createCameraProgram(std::vector<OptixProgramGroup>& program_groups)
{
    OptixProgramGroup        cam_prog_group;
    OptixProgramGroupOptions cam_prog_group_options = {};
    OptixProgramGroupDesc    cam_prog_group_desc = {};
    cam_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    cam_prog_group_desc.raygen.module = m_state.camera_module;
    cam_prog_group_desc.raygen.entryFunctionName = "__raygen__pinhole";

    char   log[2048];
    size_t sizeof_log = sizeof(log);
    OPTIX_CHECK_LOG(optixProgramGroupCreate(m_state.context, &cam_prog_group_desc, 1, &cam_prog_group_options, log,
        &sizeof_log, &cam_prog_group));

    program_groups.push_back(cam_prog_group);
    m_state.raygen_prog_group = cam_prog_group;
}

void RenderEngine::createHitProgram(std::vector<OptixProgramGroup>& program_groups)
{
    OptixProgramGroup        hitgroup_prog_group;
    OptixProgramGroupOptions hitgroup_prog_group_options = {};
    OptixProgramGroupDesc    hitgroup_prog_group_desc = {};

    hitgroup_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    hitgroup_prog_group_desc.hitgroup.moduleCH = m_state.shading_module;
    hitgroup_prog_group_desc.hitgroup.entryFunctionNameCH = "__closesthit__ch1";

    char   log[2048];
    size_t sizeof_log = sizeof(log);
    OPTIX_CHECK_LOG(optixProgramGroupCreate(m_state.context, &hitgroup_prog_group_desc, 1, &hitgroup_prog_group_options,
        log, &sizeof_log, &hitgroup_prog_group));

    program_groups.push_back(hitgroup_prog_group);
    m_state.hitgroup_prog_group = hitgroup_prog_group;

}

void RenderEngine::createMissProgram(std::vector<OptixProgramGroup>& program_groups)
{
    OptixProgramGroupOptions miss_prog_group_options = {};
    OptixProgramGroupDesc    miss_prog_group_desc = {};
    miss_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    miss_prog_group_desc.miss.module = m_state.shading_module;
    miss_prog_group_desc.miss.entryFunctionName = "__miss__raydir_shade";

    char   log[2048];
    size_t sizeof_log = sizeof(log);
    OPTIX_CHECK_LOG(optixProgramGroupCreate(m_state.context, &miss_prog_group_desc, 1, &miss_prog_group_options, log,
        &sizeof_log, &m_state.miss_prog_group));

    program_groups.push_back(m_state.miss_prog_group);
}

void RenderEngine::createPipeline()
{
    const uint32_t max_trace_depth = 1;
    const uint32_t max_cc_depth = 1;
    const uint32_t max_dc_depth = 1;
    const uint32_t max_traversal_depth = 1;

    std::vector<OptixProgramGroup> program_groups;

    m_state.pipeline_compile_options = {
        false,                                          // usesMotionBlur
        OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS,  // traversableGraphFlags
        whitted::NUM_PAYLOAD_VALUES,                    // numPayloadValues
        3,                                              // numAttributeValues
        OPTIX_EXCEPTION_FLAG_NONE,                      // exceptionFlags
        "params"                                       // pipelineLaunchParamsVariableName
    };
    m_state.pipeline_compile_options.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;


    // Prepare program groups
    createModules();
    createCameraProgram( program_groups);
    createHitProgram(program_groups);
    createMissProgram(program_groups);

    // Link program groups to pipeline
    OptixPipelineLinkOptions pipeline_link_options = {
        max_trace_depth,                // maxTraceDepth
        OPTIX_COMPILE_DEBUG_LEVEL_FULL  // debugLevel
    };
    char   log[2048];
    size_t sizeof_log = sizeof(log);
    OPTIX_CHECK_LOG(optixPipelineCreate(m_state.context, &m_state.pipeline_compile_options, &pipeline_link_options,
        program_groups.data(), static_cast<unsigned int>(program_groups.size()),
        log, &sizeof_log, &m_state.pipeline));

    OptixStackSizes stack_sizes = {};
    for (auto& prog_group : program_groups)
    {
        OPTIX_CHECK(optixUtilAccumulateStackSizes(prog_group, &stack_sizes));
    }

    uint32_t direct_callable_stack_size_from_traversal;
    uint32_t direct_callable_stack_size_from_m_state;
    uint32_t continuation_stack_size;
    OPTIX_CHECK(optixUtilComputeStackSizes(&stack_sizes, max_trace_depth, max_cc_depth, max_dc_depth, &direct_callable_stack_size_from_traversal,
        &direct_callable_stack_size_from_m_state, &continuation_stack_size));
    OPTIX_CHECK(optixPipelineSetStackSize(m_state.pipeline, direct_callable_stack_size_from_traversal,
        direct_callable_stack_size_from_m_state, continuation_stack_size, max_traversal_depth));
}

void RenderEngine::createSBT()
{
    // Raygen program record
    {
        RayGenRecord raygen_record;
        OPTIX_CHECK(optixSbtRecordPackHeader(m_state.raygen_prog_group, &raygen_record));

        CUdeviceptr d_raygen_record;
        size_t      sizeof_raygen_record = sizeof(RayGenRecord);
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_raygen_record), sizeof_raygen_record));
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_raygen_record), &raygen_record, sizeof_raygen_record, cudaMemcpyHostToDevice));

        m_state.sbt.raygenRecord = d_raygen_record;
    }

    // Miss program record
    {
        MissRecord miss_record;
        OPTIX_CHECK(optixSbtRecordPackHeader(m_state.miss_prog_group, &miss_record));

        CUdeviceptr d_miss_record;
        size_t      sizeof_miss_record = sizeof(MissRecord);
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_miss_record), sizeof_miss_record));
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_miss_record), &miss_record, sizeof_miss_record, cudaMemcpyHostToDevice));

        m_state.sbt.missRecordBase = d_miss_record;
        m_state.sbt.missRecordCount = 1;
        m_state.sbt.missRecordStrideInBytes = static_cast<uint32_t>(sizeof_miss_record);
    }

    // Hitgroup program record
    {
        if (m_state.texObjects.size() < 1)
        {
            std::cout << "Error No Textures Loaded, Cannot Generate HitRecord ... Exiting\n";
            exit(-1);
        }

        //TODO: When adding more then one texture this needs to be adjusted as currently
        // will just update the single hit record for the "last" texture added
        //HitGroupRecord hitgroup_record;
        OPTIX_CHECK(optixSbtRecordPackHeader(m_state.hitgroup_prog_group, &m_state.m_hitRecord));
        for (auto texObject : m_state.texObjects)
        {
            auto texture = texObject.second;
            m_state.m_hitRecord.data = texture->toHitRecord();
            std::cout << "H: " << texture->m_height << " W:" << texture->m_width << "\n";

        }

        CUdeviceptr d_hitgroup_record;
        size_t      sizeof_hitgroup_record = sizeof(HitGroupRecord);
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_hitgroup_record), sizeof_hitgroup_record));
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_hitgroup_record), &m_state.m_hitRecord, sizeof_hitgroup_record,
            cudaMemcpyHostToDevice));

        m_state.sbt.hitgroupRecordBase = d_hitgroup_record;
        m_state.sbt.hitgroupRecordCount = 1;
        m_state.sbt.hitgroupRecordStrideInBytes = static_cast<uint32_t>(sizeof_hitgroup_record);
    }
}


void RenderEngine::updateVideo(size_t currentFrame)
{
    std::string name = "f" + std::to_string(currentFrame);
    size_t      sizeof_hitgroup_record = sizeof(HitGroupRecord);

    auto x = m_state.texObjects.find(name);

    if ( x  != m_state.texObjects.end())
    {
        std::shared_ptr<TextureBase> tex =x->second;
        //HitGroupRecord hitgroup_record;
        //OPTIX_CHECK(optixSbtRecordPackHeader(m_state.hitgroup_prog_group, &hitgroup_record));
//        hitgroup_record.data = tex->toHitRecord();
        m_state.m_hitRecord.data.m_tex = *tex->m_texObject;
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(m_state.sbt.hitgroupRecordBase), &m_state.m_hitRecord, sizeof_hitgroup_record,
            cudaMemcpyHostToDevice));
    }
}


void RenderEngine::updateVideo(float elapsedTime)
{
    //TODO: Investigate if casting to an int is faster, and predictable 
    float DecimalSec = elapsedTime - std::trunc(elapsedTime);


    float FramePerSec = 30;
    //Currently this is hardcoded for a seven second vido, need to update this eventually
    float currentSec = ((int)(elapsedTime - DecimalSec) % 7) + DecimalSec;
   // std::cout << (currentSec * FramePerSec) << " : " << (size_t) (currentSec * FramePerSec) << "\n";

    updateVideo( (size_t) (currentSec * FramePerSec));
}


//RenderEngine::handleTime()

void RenderEngine::createContext()
{
    // Initialize CUDA
    CUDA_CHECK(cudaFree(0));

    OptixDeviceContext context;
    CUcontext          cuCtx = 0;  // zero means take the current context
    OPTIX_CHECK(optixInit());
    OptixDeviceContextOptions options = {};
    options.logCallbackFunction = &context_log_cb;
    options.logCallbackLevel = 4;
    OPTIX_CHECK(optixDeviceContextCreate(cuCtx, &options, &context));

    m_state.context = context;
}

void RenderEngine::updateTexture(std::string file)
{
    //TODO: Remove this if working with more then one texture
    m_state.texObjects.clear();
    loadTexture(sutil::getCurrFilename());
    createSBT();
}


void RenderEngine::buildEngine(std::string defaultTexture)
{
    loadTexture(defaultTexture);

    createContext();
    createGeometry();
    createPipeline();
    createSBT();
    initLaunchParams();
}


cv::Mat RenderEngine::loadImageToRam(std::string fileName)
{
    std::cout << fileName << " :Reading\n";
//    std::cout << "Attempting to load Image: " << fileName << "\n";
    cv::Mat image = cv::imread(fileName, cv::IMREAD_UNCHANGED);
    if (image.empty())
    {
        std::cout << "Error: Failed to load image - " << fileName << "\n";
        exit;
    }

    //Cuda textures need 4 channels, so if image does not have them we add an alpha channel
    if (image.channels() != 4)
    {
        std::cout << "Warning: Invalid Number of Image Channels, attempting to convert BGR to BGRA \n";
        cv::cvtColor(image, image, cv::COLOR_BGR2BGRA);
    }
    std::cout << "Succesfully loaded: " << fileName << "\n";

    return image;
}


bool RenderEngine::is_textFile(std::string fileName)
{
    std::size_t found = fileName.find("txt");
    if (found != std::string::npos)
    {
        return fileName.substr(found, fileName.length()) == "txt";
    }
    return false;
}

void RenderEngine::launchSubframe()
{

    // Launch
    uchar4* result_buffer_data = m_output_buffer.map();
    m_state.params.frame_buffer = result_buffer_data;

    CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void*>(m_state.d_params), &m_state.params,
        sizeof(whitted::LaunchParams), cudaMemcpyHostToDevice, m_state.stream));

    OPTIX_CHECK(optixLaunch(m_state.pipeline, m_state.stream, reinterpret_cast<CUdeviceptr>(m_state.d_params),
        sizeof(whitted::LaunchParams), &m_state.sbt,
        m_state.params.width,   // launch width
        m_state.params.height,  // launch height
        1                     // launch depth
    ));

    m_output_buffer.unmap();
    CUDA_SYNC_CHECK();
}

void RenderEngine::cleanupState()
{
    OPTIX_CHECK(optixPipelineDestroy(m_state.pipeline));
    OPTIX_CHECK(optixProgramGroupDestroy(m_state.raygen_prog_group));
    OPTIX_CHECK(optixProgramGroupDestroy(m_state.hitgroup_prog_group));
    OPTIX_CHECK(optixProgramGroupDestroy(m_state.miss_prog_group));

    OPTIX_CHECK(optixModuleDestroy(m_state.shading_module));
    OPTIX_CHECK(optixModuleDestroy(m_state.camera_module));
    OPTIX_CHECK(optixDeviceContextDestroy(m_state.context));

    CUDA_CHECK(cudaStreamDestroy(m_state.stream));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(m_state.sbt.raygenRecord)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(m_state.sbt.missRecordBase)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(m_state.sbt.hitgroupRecordBase)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(m_state.d_gas_output_buffer)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(m_state.params.accum_buffer)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(m_state.params.lights.data)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(m_state.d_params)));
}

 void RenderEngine::context_log_cb(unsigned int level, const char* tag, const char* message, void* /*cbdata */)
{
    std::cerr << "[" << std::setw(2) << level << "][" << std::setw(12) << tag << "]: " << message << "\n";
}

