


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

	RenderEngine m_optixEngine = RenderEngine::RenderEngine();
	sutil::Camera    m_camera;
	float4 lookDirection;

	XrResult result;
	
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



	const XrPosef  xr_pose_identity = { {0,0.705,0,0.709}, {0,0,0} };
	XrSpace        xr_app_space = {};
	std::vector<XrView>                  xr_views;
	std::vector<XrViewConfigurationView> xr_config_views;
	

	
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

		openXR_app::openXR_app(GLFWwindow* window);
		bool launchApp();


		//prepares Swapchain. XR applications will want to present rendered images to the user. 
		//To allow this, the runtime provides images organized in swapchains for the application to render into.
		//return true if it succesfully creates swapchain.
		bool prepareSwapchain();

		//facilates in swapchain image rendering. Swapchain image is attached to framebuffer before rendering into swapchain image.
		bool prepareGLFramebufer(uint32_t view_count, uint32_t* swapchain_lengths, GLuint*** framebuffers, GLuint* shader_program_id,
			GLuint* VAO);

		//Distroy swapchain just before application is terminated.
		void swapchain_destroy(swapchain_t& swapchain); 

		//renders one frame of VR application, asks OpenXR runtime for position, orientation, and what swapchain Image to render into. 
		
		bool renderLayer(XrTime predictedTime, std::vector<XrCompositionLayerProjectionView>& views, XrCompositionLayerProjection& layer);
		
		//actual rendering on the swapchain image happens here.
		void GLrendering(XrCompositionLayerProjectionView& view, GLuint surface, GLuint swapchainImage, int eye, GLuint shader_program_id,
			GLuint VAO);

		
		//helper functions

		static void print_instance_properties(XrInstance instance);
		
		static void print_system_properties(XrSystemProperties* system_properties);
		
		static void print_viewconfig_view_info(uint32_t view_count, XrViewConfigurationView* viewconfig_views);
		
		//converts Quaternion into its rotationMatrix. 
		std::vector<float> makeRotationMatrix4x4(float4 quaternion);
		
		// Rotation matrix is multiplied by coloumn vector
		float4 MatrixMul(std::vector<float> rotationMatrix, float4 vector4);

		//initalizes openXR
		bool openxr_init(const char* app_name);
		
		//Poll for headset state
		void openxr_poll_events(bool& exit);

		//render frame from actual rendering loop
		void renderFrame();
		

		bool get_xr_running();
	
		//builds Optix Rendering engine
		bool buildEngine();

		XrSessionState get_xr_session_state();

		//terminates OpenXR
		void openxr_shutdown();
		

};


