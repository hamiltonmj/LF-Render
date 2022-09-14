


#pragma once

// Tell OpenXR what platform code we'll be using
#define XR_USE_PLATFORM_WIN32
#define XR_USE_GRAPHICS_API_OPENGL

#include <openxr/include/openxr.h>
#include <openxr/include/openxr_platform.h>
#include <sutil/sutil.h>
#include <glad/glad.h>
#include <sutil/Camera.h>
#include <sutil/CUDAOutputBuffer.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include "RenderEngine.h"
#include <sutil/GLDisplay.h>

#include <sutil/Exception.h>
#include <vector>
#include <thread>
#include <algorithm>


//Swapchain management
struct swapchain_t {
	XrSwapchain handle;
	int32_t     width;
	int32_t     height;
	std::vector<XrSwapchainImageOpenGLKHR> swapchain_images;
};




///////////////////////////////////////////////////////////////
// 
///////////////////////////////////////////////////////////////

class openXR_app
	{

	//sutil::GLDisplay gl_display;
	//sutil::BufferImageFormat m_image_format;
	RenderEngine m_optixEngine = RenderEngine::RenderEngine(1568,1568);
	sutil::Camera    m_camera;

	
	

	sutil::Camera    Left_m_camera;
	sutil::Camera    Right_m_camera;
	float4 lookDirection;

	XrResult result;

	////////////////////////////////////////////


	///////////////////////////////////////////

	// Function pointers for some OpenXR extension methods we'll use.
	PFN_xrGetOpenGLGraphicsRequirementsKHR ext_xrGetOpenGLGraphicsRequirementsKHR = nullptr;
	PFN_xrCreateDebugUtilsMessengerEXT    ext_xrCreateDebugUtilsMessengerEXT = nullptr;
	PFN_xrDestroyDebugUtilsMessengerEXT   ext_xrDestroyDebugUtilsMessengerEXT = nullptr;

	XrFormFactor	app_config_form = XR_FORM_FACTOR_HEAD_MOUNTED_DISPLAY;
	XrViewConfigurationType app_config_view = XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO;


	XrInstance	xr_instance = {};
	XrDebugUtilsMessengerEXT xr_debug = {};
	XrSystemId     xr_system_id = XR_NULL_SYSTEM_ID;
	XrEnvironmentBlendMode   xr_blend = {};
	
	XrSession      xr_session = {};
	bool xr_running = false;
	XrSessionState xr_session_state = XR_SESSION_STATE_UNKNOWN;



	const XrPosef  xr_pose_identity = { {0,0,0,1}, {0,0,0} };
	XrSpace        xr_app_space = {};
	std::vector<XrView>                  xr_views;
	std::vector<XrViewConfigurationView> xr_config_views;
	
	float3 left_eye_dir;
	float3 right_eye_dir;
	
	int renderTargetWidth;
	int renderTargetHeight;

	//Swapchain management
	// "swapchain_t" stuct holds actual swapchain
	std::vector<swapchain_t>             xr_swapchains; // Vector that holds Two swapchains
	uint32_t* swapchain_lengths; //Two swapchain for 2 views (2 eyes), length of xr_swapchain

	struct
	{
		// To render into a texture we need a framebuffer (one per texture to make it easy)
		GLuint** framebuffers;

		float near_z;
		float far_z;

		GLuint shader_program_id;
		GLuint VAO;
	} gl_rendering;
	//

	
	GLFWwindow* window;

	///////////////////////////////////////////



	public:


		bool prepareSwapchain();
		bool openxr_init(const char* app_name);
		bool renderLayer(XrTime predictedTime, std::vector<XrCompositionLayerProjectionView> &views, XrCompositionLayerProjection &layer);
		void renderFrame();
		void openxr_poll_events(bool& exit);
		void openxr_shutdown();
		bool get_xr_running();
		bool prepareGLFramebufer(uint32_t view_count, uint32_t* swapchain_lengths, GLuint*** framebuffers, GLuint* shader_program_id,
			GLuint* VAO);
		void GLrendering(XrCompositionLayerProjectionView &view, GLuint surface, GLuint swapchainImage, int eye, GLuint shader_program_id,
			GLuint VAO);
		XrSessionState get_xr_session_state();
		void swapchain_destroy(swapchain_t &swapchain);
		bool buildEngine();
		//helper functions
		static void print_instance_properties(XrInstance instance);
		static void print_system_properties(XrSystemProperties* system_properties);
		static void print_viewconfig_view_info(uint32_t view_count, XrViewConfigurationView* viewconfig_views);
		void displayFrame_onWindow(sutil::CUDAOutputBuffer<uchar4>& output_buffer, sutil::GLDisplay& gl_display, GLFWwindow* window);
		std::vector<float> makeRotationMatrix4x4(float4 quaternion);
		float4 MatrixMul(std::vector<float> rotationMatrix, float4 vector4);
		float3 amplifyPos(float x, float y, float z, float amplifyBy);
		

		
		
		

};


