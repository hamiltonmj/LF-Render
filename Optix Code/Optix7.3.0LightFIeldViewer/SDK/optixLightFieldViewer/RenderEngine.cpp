
#include "opencv2/core.hpp" 
#include "opencv2/opencv.hpp"

#include <glad/glad.h>  // Needs to be included before gl_interop
#include <cuda_gl_interop.h>
#include <cuda_runtime.h>

#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stack_size.h>
#include <optix_stubs.h>

#include <sutil/CUDAOutputBuffer.h>
#include <sutil/Camera.h>
#include <sutil/Exception.h>

#include <cuda/whitted.h>
#include "RenderEngine.h"

RenderEngine::RenderEngine(size_t w, size_t h)
    :m_state(RenderState()), m_output_buffer(sutil::CUDAOutputBufferType::CUDA_DEVICE, w,h)
  //  :m_state(), m_output_buffer(sutil::CUDAOutputBufferType::GL_INTEROP, w, h)
{
   // m_state = ;
    m_state.params.width = w;
    m_state.params.height = h;
}

//Resolution setting 
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
    m_fileReader.loadTexFile(fileName, m_state);
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

void RenderEngine::buildGas( const OptixBuildInput& build_input)
{

    OptixAccelBuildOptions accel_options = {};// { OPTIX_BUILD_FLAG_ALLOW_COMPACTION, OPTIX_BUILD_OPERATION_BUILD };
    accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
    accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;

    // Use default options for simplicity.  In a real use case we would want to
// enable compaction, etc

   // OptixAccelBufferSizes gas_buffer_sizes;
    OptixAccelBufferSizes gas_buffer_sizes;
    OPTIX_CHECK(optixAccelComputeMemoryUsage(
        m_state.context,
        &accel_options,
        &build_input,
        1, // Number of build inputs
        &gas_buffer_sizes
    ));
    
    CUdeviceptr d_temp_buffer_gas;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_temp_buffer_gas), gas_buffer_sizes.tempSizeInBytes));

    CUdeviceptr d_buffer_temp_output_gas_and_compacted_size;
    size_t      compactedSizeOffset = roundUp<size_t>(gas_buffer_sizes.outputSizeInBytes, 8ull);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_buffer_temp_output_gas_and_compacted_size), compactedSizeOffset + 8));

    OptixAccelEmitDesc emitProperty = {};
    emitProperty.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
    emitProperty.result = (CUdeviceptr)((char*)d_buffer_temp_output_gas_and_compacted_size + compactedSizeOffset);

    OPTIX_CHECK(optixAccelBuild(
        m_state.context,
        0,                  // CUDA stream
        &accel_options,
        &build_input,
        1,                  // num build inputs
        d_temp_buffer_gas,
        gas_buffer_sizes.tempSizeInBytes,
        d_buffer_temp_output_gas_and_compacted_size,
        gas_buffer_sizes.outputSizeInBytes,
        &m_state.gas_handle,
        &emitProperty,            // emitted property list
        1                   // num emitted properties
    ));

    // We can now free the scratch space buffer used during build and the vertex
    // inputs, since they are not needed by our trivial shading method
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_temp_buffer_gas)));

    size_t compacted_gas_size;
    CUDA_CHECK(cudaMemcpy(&compacted_gas_size, (void*)emitProperty.result, sizeof(size_t), cudaMemcpyDeviceToHost));

    if (compacted_gas_size < gas_buffer_sizes.outputSizeInBytes)
    {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&m_state.d_gas_output_buffer), compacted_gas_size));

        // use handle as input and output     // change gas_handle to stat.gas_handle
        OPTIX_CHECK(optixAccelCompact(m_state.context, 0, m_state.gas_handle, m_state.d_gas_output_buffer, compacted_gas_size, &m_state.gas_handle));

        CUDA_CHECK(cudaFree((void*)d_buffer_temp_output_gas_and_compacted_size));
    }
    else
    {
        m_state.d_gas_output_buffer = d_buffer_temp_output_gas_and_compacted_size;
    }
}
//Eventually should seperate the geomertry creation and the building of the gas 
void RenderEngine::createGeometry()
{
    //
    // To ADD Triangles to be rendered simply add them to the vertices array and change the size of the array to match the new inputs 
    // Triangle build input: simple list of three vertices

    float w, h, d;
    float3 Pos = make_float3(0.0, 0.0, 0.0);

    w = 0.5;
    h = 0.5;
    d = 0.0;
    //w = 1.0;
    // h = 1.0;
    // d = -10.0;

    float halfW = w / 2;
    float halfH = h / 2;

    const std::array<float3, 6> vertices =
    { {
            make_float3( -halfW, -halfH, d) + Pos,
            make_float3(  halfW, -halfH, d) + Pos,
            make_float3( -halfW,  halfH, d) + Pos,

            make_float3(  halfW,  halfH, d) + Pos,
            make_float3( -halfW,  halfH, d) + Pos,
            make_float3(  halfW, -halfH, d) + Pos
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

    //Mulit-shaders we create a list of indexs which correspond between what primitve connects with what sbt record by index
//        const uint32_t sbt_index[] = { 0, 0, 1, 1 };
    const uint32_t sbt_index[] = { 0, 1 }; //1 };
    CUdeviceptr d_sbt_index;
    
    //Mulit-shaders we nopw take this mapping and store it on the graphics card for optix to use during launches
//        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_sbt_index), sizeof(sbt_index) *2));
//        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_sbt_index), sbt_index, sizeof(sbt_index) *2, cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_sbt_index), sizeof(sbt_index)));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_sbt_index), sbt_index, sizeof(sbt_index), cudaMemcpyHostToDevice));

    //multi-shader  Need to include flags for each hit record given:
    uint32_t flagsPerSBTRecord[2];
    flagsPerSBTRecord[0] = OPTIX_GEOMETRY_FLAG_NONE;
    flagsPerSBTRecord[1] = OPTIX_GEOMETRY_FLAG_NONE;

    // Our build input is a simple list of non-indexed triangle vertices
    // const uint32_t triangle_input_flags[1] = { OPTIX_GEOMETRY_FLAG_NONE };

    OptixBuildInput triangle_input = {};
    triangle_input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
    triangle_input.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
    triangle_input.triangleArray.numVertices = static_cast<uint32_t>(vertices.size());
    triangle_input.triangleArray.vertexBuffers = &d_vertices;
    triangle_input.triangleArray.vertexStrideInBytes = sizeof(float3);
    //Multi-Shader: telling optix that i have mutiple shaders to be used within this acceleration structure
    uint32_t numShaders = 2;
    triangle_input.triangleArray.numSbtRecords = numShaders;
    triangle_input.triangleArray.sbtIndexOffsetBuffer = d_sbt_index;
    triangle_input.triangleArray.sbtIndexOffsetSizeInBytes = sizeof(uint32_t);//sizeof(uint32_t);
    triangle_input.triangleArray.sbtIndexOffsetStrideInBytes = sizeof(uint32_t);
    triangle_input.triangleArray.flags = flagsPerSBTRecord;
        
    buildGas(triangle_input);
    CUDA_CHECK(cudaFree((void*)d_sbt_index));
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

void RenderEngine::createHitProgram(std::vector<OptixProgramGroup>& program_groups, std::string hitFuncName)
{
    OptixProgramGroup        hitgroup_prog_group;
    OptixProgramGroupOptions hitgroup_prog_group_options = {};
    OptixProgramGroupDesc    hitgroup_prog_group_desc = {};

    hitgroup_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    hitgroup_prog_group_desc.hitgroup.moduleCH = m_state.shading_module;
    hitgroup_prog_group_desc.hitgroup.entryFunctionNameCH = hitFuncName.c_str();

    char   log[2048];
    size_t sizeof_log = sizeof(log);
    OPTIX_CHECK_LOG(optixProgramGroupCreate(m_state.context, &hitgroup_prog_group_desc, 1, &hitgroup_prog_group_options,
        log, &sizeof_log, &hitgroup_prog_group));

    program_groups.push_back(hitgroup_prog_group);
    m_state.hitgroup_prog_group = hitgroup_prog_group;


    //Multi Shader temproray code to create a second hit shader for use within the application 
    OptixProgramGroup        hitgroup_prog_group1;
    OptixProgramGroupOptions hitgroup_prog_group_options1 = {};
    OptixProgramGroupDesc    hitgroup_prog_group_desc1 = {};

    hitgroup_prog_group_desc1.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    hitgroup_prog_group_desc1.hitgroup.moduleCH = m_state.shading_module;
    hitgroup_prog_group_desc1.hitgroup.entryFunctionNameCH = "__closesthit__ch1";

    char   log1[2048];
    size_t sizeof_log1 = sizeof(log1);
    OPTIX_CHECK_LOG(optixProgramGroupCreate(m_state.context, &hitgroup_prog_group_desc1, 1, &hitgroup_prog_group_options1,
        log1, &sizeof_log1, &hitgroup_prog_group1));

    program_groups.push_back(hitgroup_prog_group1);
    m_state.hitgroup_prog_group1 = hitgroup_prog_group1;

    std::cout << log1;

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
    const uint32_t max_trace_depth = 2;
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
    m_state.pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_DEBUG | OPTIX_EXCEPTION_FLAG_TRACE_DEPTH | OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW;// | OPTIX_EXCEPTION_FLAG_USER;
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

        size_t const numRecords = 3;
        HitGroupRecordCudaPointer hitGroupsPointer[numRecords];
        HitGroupRecord hitGroups[numRecords];

        size_t sizeof_hitgroup_record = sizeof(HitGroupRecord);//sizeof(hitGroups[0]);
        size_t sizeof_hitgroupPointer_record = sizeof(HitGroupRecordCudaPointer);//sizeof(hitGroups[0]);


        //Multi Shader adding [] to each hit group to refernce correct one 

//        OPTIX_CHECK(optixSbtRecordPackHeader(m_state.hitgroup_prog_group, &m_state.m_hitRecord));
        OPTIX_CHECK(optixSbtRecordPackHeader(m_state.hitgroup_prog_group, &hitGroups[0]));
        OPTIX_CHECK(optixSbtRecordPackHeader(m_state.hitgroup_prog_group1, &hitGroups[1]));
        OPTIX_CHECK(optixSbtRecordPackHeader(m_state.hitgroup_prog_group1, &hitGroups[2]));

        OPTIX_CHECK(optixSbtRecordPackHeader(m_state.hitgroup_prog_group1, &hitGroupsPointer[0]));
        OPTIX_CHECK(optixSbtRecordPackHeader(m_state.hitgroup_prog_group1, &hitGroupsPointer[1]));
        OPTIX_CHECK(optixSbtRecordPackHeader(m_state.hitgroup_prog_group1, &hitGroupsPointer[2]));


        // CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&hitGroups[0].data), sizeof(HitGroupData)));
        //CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&hitGroups[1].data), sizeof(HitGroupData)));

        for (auto texObject : m_state.texObjects)
        {
            auto texture = texObject.second;
            HitGroupData temp1 = *texture->toHitRecord();
//            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&hitGroupsPointer[0].data), sizeof(HitGroupData)));
//            CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(hitGroupsPointer[0].data), texture->toHitRecord(), sizeof(HitGroupData), cudaMemcpyHostToDevice));


            hitGroups[0].data = *(texture->toHitRecord());
            hitGroups[1].data = *(texture->toHitRecord());
            hitGroups[2].data = *(texture->toHitRecord());

            hitGroupsPointer[0].data = texture->toDeviceHitRecord();
            hitGroupsPointer[1].data = texture->toDeviceHitRecord();
            hitGroupsPointer[2].data = texture->toDeviceHitRecord();

            std::cout << "H: " << texture->m_texHeight << " W:" << texture->m_texWidth << "\n";
        }

        CUdeviceptr d_hitgroup_record;
//        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_hitgroup_record), sizeof_hitgroup_record * numRecords));
//        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_hitgroup_record), &hitGroups, sizeof_hitgroup_record * numRecords,
//            cudaMemcpyHostToDevice));

        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_hitgroup_record), sizeof_hitgroupPointer_record * numRecords));
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_hitgroup_record), &hitGroupsPointer, sizeof_hitgroupPointer_record * numRecords,
            cudaMemcpyHostToDevice));


        m_state.sbt.hitgroupRecordBase = d_hitgroup_record;
        m_state.sbt.hitgroupRecordCount = numRecords;
        m_state.sbt.hitgroupRecordStrideInBytes = static_cast<uint32_t>(sizeof_hitgroup_record);
        m_state.sbt.hitgroupRecordStrideInBytes = static_cast<uint32_t>(sizeof_hitgroupPointer_record);


//        m_state.sbt.hitgroupRecordBase = d_hitgroup_record;
//        m_state.sbt.hitgroupRecordCount = 2;// numRecords;
//        m_state.sbt.hitgroupRecordStrideInBytes = static_cast<uint32_t>(sizeof_hitgroup_record);

        // HitGroupDataFloat tempX;
        // tempX.m_val = 1;
        // CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&(hitGroups[1].data)), sizeof(HitGroupDataFloat)));
        // CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(hitGroups[1].data), &tempX, sizeof(HitGroupDataFloat), cudaMemcpyHostToDevice));

 //        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_hitgroup_record), sizeof_hitgroup_record * numRecords));
 //        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_hitgroup_record), hitGroups, sizeof_hitgroup_record * numRecords,
 //            cudaMemcpyHostToDevice));


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
    //    m_state.m_hitRecord.data.m_tex = *tex->m_texObject;
     //   CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(m_state.sbt.hitgroupRecordBase), &m_state.m_hitRecord, sizeof_hitgroup_record,
      //      cudaMemcpyHostToDevice));
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
    //options.validationMode = OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL;
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

