


#include "optixLightFieldViewer/VirtualReality.h"

//#include "openxr/include/openxr.h"
//#include "openxr/include/openxr_platform.h"
#include <glad/glad.h>
#include <cmath>
#include <vector>
#include <stdio.h>
#include <stdlib.h>

#include <algorithm>
#include <iostream>


openXR_app::openXR_app(GLFWwindow* window1)
{
	//window = window1;
	glfwMakeContextCurrent(window1);
}
////////////////////////////
bool openXR_app::launchApp() 
{

	if (!openxr_init("VR_app")) {
		MessageBox(nullptr, "OpenXR initialization failed\n", "Error", 1);
		return 1;
	}

	buildEngine();
	bool quit = false;

	while (!quit) {

		openxr_poll_events(quit);

		if (get_xr_running()) {
			renderFrame();
			
			if (get_xr_session_state() != XR_SESSION_STATE_VISIBLE &&
				get_xr_session_state() != XR_SESSION_STATE_FOCUSED) {

				std::this_thread::sleep_for(std::chrono::milliseconds(250));
			}
		}





	} 

	openxr_shutdown();

	return 0;


}



void openXR_app::prepareGLFramebufer( uint32_t swapchain_length, GLuint*& framebuffers)
{
	framebuffers = (GLuint*) malloc(sizeof(GLuint) * swapchain_length);
	glGenFramebuffers(swapchain_length, framebuffers);



	/* Allocate resources that we use for our own rendering.
	 * We will bind framebuffers to the runtime provided textures for rendering.
	 * For this, we create one framebuffer per OpenGL texture.
	 * This is not mandated by OpenXR, other ways to render to textures will work too.
	 */
	 /**framebuffers = (GLuint**) malloc(sizeof(GLuint*) * view_count);
	 for (uint32_t i = 0; i < view_count; i++) {
		 (*framebuffers)[i] = (GLuint*) malloc(sizeof(GLuint) * swapchain_lengths[i]);
		 glGenFramebuffers(swapchain_lengths[i], (*framebuffers)[i]);
	 }*/
	/*
	* framebuffers = (GLuint*) malloc(sizeof(GLuint*) * view_count);
	for (uint32_t i = 0; i < view_count; i++)
	{
		(*framebuffers)[i] = malloc(sizeof(GLuint) * swapchain_lengths[i]);
		glGenFramebuffers(swapchain_lengths[i], (*framebuffers)[i]);
	}
	*/
}


bool openXR_app::prepareGLFramebufer(uint32_t view_count, uint32_t* swapchain_lengths, GLuint*** framebuffers )
{
	/* Allocate resources that we use for our own rendering.
	 * We will bind framebuffers to the runtime provided textures for rendering.
	 * For this, we create one framebuffer per OpenGL texture.
	 * This is not mandated by OpenXR, other ways to render to textures will work too.
	 */
	/**framebuffers = (GLuint**) malloc(sizeof(GLuint*) * view_count);
	for (uint32_t i = 0; i < view_count; i++) {
		(*framebuffers)[i] = (GLuint*) malloc(sizeof(GLuint) * swapchain_lengths[i]);
		glGenFramebuffers(swapchain_lengths[i], (*framebuffers)[i]);
	}*/
	/*
	*framebuffers = malloc(sizeof(GLuint*) * view_count);
	for (uint32_t i = 0; i < view_count; i++) 
	{
		(*framebuffers)[i] = malloc(sizeof(GLuint) * swapchain_lengths[i]);
		glGenFramebuffers(swapchain_lengths[i], (*framebuffers)[i]);
	}
	*/

//	*framebuffers = (GLuint**) malloc(sizeof(GLuint*) * view_count);
	//for (uint32_t i = 0; i < view_count; i++) {
//		(*framebuffers)[i] = (GLuint*) malloc(sizeof(GLuint) * swapchain_lengths[i]);
	//	glGenFramebuffers(swapchain_lengths[i], (*framebuffers)[i]);
	//}



	return 0;


}

//##################################################################################
// create swapchain
bool openXR_app::prepareSwapchain()
{
	
	// Now we need to find all the viewpoints we need to take care of! For a stereo headset, this should be 2.
	// Similarly, for an AR phone, we'll need 1, and a VR cave could have 6, or even 12!
	uint32_t view_count = 0;
	xrEnumerateViewConfigurationViews(xr_instance, xr_system_id, app_config_view, 0, &view_count, nullptr);
	xr_config_views.resize(view_count, { XR_TYPE_VIEW_CONFIGURATION_VIEW });
	xr_views.resize(view_count, { XR_TYPE_VIEW });
    result = xrEnumerateViewConfigurationViews(xr_instance, xr_system_id, app_config_view, view_count, &view_count, xr_config_views.data());

	if (XR_SUCCESS != result) {
		std::cout << "Failed to get ViewConfigurationViews.  xrEnumerateViewConfigurationViews() returned XrResult :" << result << "\n";
		return false;
	};

	print_viewconfig_view_info(view_count, xr_config_views.data());

	
	//prepare two seperate swapchains for Left and Right Views.
	swapchain_lengths =  (uint32_t*)malloc(sizeof(uint32_t) * view_count);
	for (uint32_t i = 0; i < view_count; i++) {

		XrViewConfigurationView& view = xr_config_views[i];
		XrSwapchainCreateInfo    swapchain_info = { XR_TYPE_SWAPCHAIN_CREATE_INFO };
		XrSwapchain              handle;
		swapchain_info.arraySize = 1;
		swapchain_info.mipCount = 1;
		swapchain_info.faceCount = 1;
		swapchain_info.format = (int64_t)GL_SRGB8_ALPHA8;
//		swapchain_info.format = (int64_t) GL_SRGB;
		swapchain_info.width = view.recommendedImageRectWidth; 
		renderTargetWidth = view.recommendedImageRectWidth;
		
		swapchain_info.height = view.recommendedImageRectHeight; 
		renderTargetHeight = view.recommendedImageRectHeight;
		
		swapchain_info.sampleCount = view.recommendedSwapchainSampleCount;
		swapchain_info.usageFlags = XR_SWAPCHAIN_USAGE_SAMPLED_BIT | XR_SWAPCHAIN_USAGE_COLOR_ATTACHMENT_BIT;
		xrCreateSwapchain(xr_session, &swapchain_info, &handle);
		
		// Find out how many textures were generated for the swapchain
		swapchain_lengths[i] = 0;
		xrEnumerateSwapchainImages(handle, 0, &swapchain_lengths[i], nullptr);
		//swapchain_lengths[i] = (uint32_t) &surface_count;
		// We'll want to track our own information about the swapchain, so we can draw stuff onto it! We'll also create
		// a depth buffer for each generated texture here as well with make_surfacedata.
		swapchain_t swapchain = {};
		swapchain.width = swapchain_info.width;
		swapchain.height = swapchain_info.height;
		swapchain.handle = handle;
		swapchain.swapchain_images.resize(swapchain_lengths[i], { XR_TYPE_SWAPCHAIN_IMAGE_OPENGL_KHR });
		
		
		xrEnumerateSwapchainImages(swapchain.handle, swapchain_lengths[i], &swapchain_lengths[i], (XrSwapchainImageBaseHeader*)swapchain.swapchain_images.data());

		prepareGLFramebufer(swapchain_lengths[i], swapchain.glFrameBuffers);

		xr_swapchains.push_back(swapchain);

	}
	
	//prepareGLFramebufer(view_count, swapchain_lengths, &gl_rendering.framebuffers, &gl_rendering.shader_program_id, &gl_rendering.VAO);
	
	return true;
	
}




/// <summary>
/// Initialize OPENXR and bind OpenGL to our app.
/// </summary>
/// <param name="app_name"></param>
/// <returns></returns>
bool openXR_app::openxr_init(const char* app_name)
{
	// OpenXR will fail to initialize if we ask for an extension that OpenXR
	// can't provide! So we need to check our all extensions before 
	// initializing OpenXR with them. Note that even if the extension is 
	// present, it's still possible you may not be able to use it. For 
	// example: the hand tracking extension may be present, but the hand
	// sensor might not be plugged in or turned on. There are often 
	// additional checks that should be made before using certain features!
	
	std::vector<const char*> use_extensions;
	const char *ask_extensions[] = {
		XR_KHR_OPENGL_ENABLE_EXTENSION_NAME, // Use OpenGL for rendering
		XR_EXT_DEBUG_UTILS_EXTENSION_NAME,  // Debug utils for extra info
	};


	// We'll get a list of extensions that OpenXR provides using this 
	// enumerate pattern. OpenXR often uses a two-call enumeration pattern 
	// where the first call will tell you how much memory to allocate, and
	// the second call will provide you with the actual data!
	uint32_t ext_count = 0;
	xrEnumerateInstanceExtensionProperties(nullptr, 0, &ext_count, nullptr);
	std::vector<XrExtensionProperties> xr_exts(ext_count, { XR_TYPE_EXTENSION_PROPERTIES });
	xrEnumerateInstanceExtensionProperties(nullptr, ext_count, &ext_count, xr_exts.data());

	printf("OpenXR extensions available:\n");
	for (size_t i = 0; i < xr_exts.size(); i++) {
		printf("- %s\n", xr_exts[i].extensionName);

		// Check if we're asking for this extensions, and add it to our use 
		// list!
		for (int32_t ask = 0; ask < _countof(ask_extensions); ask++) {
			if (strcmp(ask_extensions[ask], xr_exts[i].extensionName) == 0) {
				use_extensions.push_back(ask_extensions[ask]);
				break;
			}
		}
	}

	// If a required extension isn't present, you want to ditch out here!
	// It's possible something like your rendering API might not be provided
	// by the active runtime. APIs like OpenGL don't have universal support.
	if (!std::any_of(use_extensions.begin(), use_extensions.end(),
		[](const char* ext) {
			return strcmp(ext, XR_KHR_OPENGL_ENABLE_EXTENSION_NAME) == 0;
		}))
		return false;



	// Initialize OpenXR with the extensions we've found!
	XrInstanceCreateInfo createInfo = { XR_TYPE_INSTANCE_CREATE_INFO };
	createInfo.enabledExtensionCount = use_extensions.size();
	createInfo.enabledExtensionNames = use_extensions.data();
	createInfo.applicationInfo.apiVersion = XR_CURRENT_API_VERSION;
	strcpy_s(createInfo.applicationInfo.applicationName, app_name);
	result = xrCreateInstance(&createInfo, &xr_instance);
	// Check if OpenXR is on this system, if this is null here, the user 
	// needs to install an OpenXR runtime and ensure it's active!

	if (XR_SUCCESS !=  result) {
		std::cout << "Failed to create XR Instance.  xrCreateInstance() returned XrResult :" << result << "\n";
		return false;
	}
	else print_instance_properties(xr_instance);

	// Load extension methods that we'll need for this application! There's a
	// couple ways to do this, and this is a fairly manual one. Chek out this
	// file for another way to do it:
	// https://github.com/maluoi/StereoKit/blob/master/StereoKitC/systems/platform/openxr_extensions.h
	xrGetInstanceProcAddr(xr_instance, "xrCreateDebugUtilsMessengerEXT", (PFN_xrVoidFunction*)(&ext_xrCreateDebugUtilsMessengerEXT));
	xrGetInstanceProcAddr(xr_instance, "xrDestroyDebugUtilsMessengerEXT", (PFN_xrVoidFunction*)(&ext_xrDestroyDebugUtilsMessengerEXT));
	xrGetInstanceProcAddr(xr_instance, "xrGetOpenGLGraphicsRequirementsKHR", (PFN_xrVoidFunction*)(&ext_xrGetOpenGLGraphicsRequirementsKHR));


	// Set up a really verbose debug log! Great for dev, but turn this off or
	// down for final builds. WMR doesn't produce much output here, but it
	// may be more useful for other runtimes?
	// Here's some extra information about the message types and severities:
	// https://www.khronos.org/registry/OpenXR/specs/1.0/html/xrspec.html#debug-message-categorization
	XrDebugUtilsMessengerCreateInfoEXT debug_info = { XR_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT };
	debug_info.messageTypes =
		XR_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
		XR_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
		XR_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT |
		XR_DEBUG_UTILS_MESSAGE_TYPE_CONFORMANCE_BIT_EXT;
	debug_info.messageSeverities =
		XR_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT |
		XR_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT |
		XR_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
		XR_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
	debug_info.userCallback = [](XrDebugUtilsMessageSeverityFlagsEXT severity, XrDebugUtilsMessageTypeFlagsEXT types, const XrDebugUtilsMessengerCallbackDataEXT* msg, void* user_data) {
		// Print the debug message we got! There's a bunch more info we could
		// add here too, but this is a pretty good start, and you can always
		// add a breakpoint this line!
		printf("%s: %s\n", msg->functionName, msg->message);

		// Output to debug window
		char text[512];
		sprintf_s(text, "%s: %s", msg->functionName, msg->message);
		OutputDebugStringA(text);

		// Returning XR_TRUE here will force the calling function to fail
		return (XrBool32)XR_FALSE;
	};

	// Start up the debug utils!
	if (ext_xrCreateDebugUtilsMessengerEXT)
		ext_xrCreateDebugUtilsMessengerEXT(xr_instance, &debug_info, &xr_debug);

	// Request a form factor from the device (HMD, Handheld, etc.)
	XrSystemGetInfo systemInfo = { XR_TYPE_SYSTEM_GET_INFO };
	systemInfo.formFactor = app_config_form;
	result =  xrGetSystem(xr_instance, &systemInfo, &xr_system_id);
	if (result != XR_SUCCESS) {
		printf("Failed to get system for HMD form factor.  xrGetSystem() returned XrResult :" + result);
		return false;
	}
	else printf("Successfully got XrSystem with id %lu for HMD form factor\n", xr_system_id);

	// Check what blend mode is valid for this device (opaque vs transparent displays)
	// We'll just take the first one available!
	uint32_t blend_count = 0;
	xrEnumerateEnvironmentBlendModes(xr_instance, xr_system_id, app_config_view, 1, &blend_count, &xr_blend);

	// OpenXR wants to ensure apps are using the correct graphics card, so this MUST be called 
	// before xrCreateSession. This is crucial on devices that have multiple graphics cards, 
	// like laptops with integrated graphics chips in addition to dedicated graphics cards.
	XrGraphicsRequirementsOpenGLKHR requirement = { XR_TYPE_GRAPHICS_REQUIREMENTS_OPENGL_KHR };
	ext_xrGetOpenGLGraphicsRequirementsKHR(xr_instance, xr_system_id, &requirement);
	
	//window = sutil::initUI("Real Time Lightfield Rendering in VR", 1568, 1568);
	

	
	
	XrSystemProperties systemProperties = { XR_TYPE_SYSTEM_PROPERTIES , NULL };
	result = xrGetSystemProperties(xr_instance, xr_system_id, &systemProperties);
	if (result != XR_SUCCESS) {
		std::cout << "Failed to get system properties.  xrGetSystemProperties() returned XrResult :" << result << "\n";
		return false;
	}
	else print_system_properties(&systemProperties);

	//
	
	// A session represents this application's desire to display things! This is where we hook up our graphics API.
	// This does not start the session, for that, you'll need a call to xrBeginSession, which we do in openxr_poll_events
	XrGraphicsBindingOpenGLWin32KHR graphicsBinding = { XR_TYPE_GRAPHICS_BINDING_OPENGL_WIN32_KHR, nullptr, wglGetCurrentDC(), wglGetCurrentContext() };
	
	

	XrSessionCreateInfo sessionInfo = { XR_TYPE_SESSION_CREATE_INFO };
	sessionInfo.next = &graphicsBinding;
	sessionInfo.systemId = xr_system_id;
	result = xrCreateSession(xr_instance, &sessionInfo, &xr_session);
	if (result != XR_SUCCESS) {
		std::cout << "Failed to create Session.  xrCreateSession() returned XrResult :" << result << "\n";
		return false;
	}
	else printf("Successfully created a session with OpenGL!\n");
	

	

	// OpenXR uses a couple different types of reference frames for positioning content, we need to choose one for
	// displaying our content! STAGE would be relative to the center of your guardian system's bounds, and LOCAL
	// would be relative to your device's starting location. HoloLens doesn't have a STAGE, so we'll use LOCAL.
	XrReferenceSpaceCreateInfo ref_space = { XR_TYPE_REFERENCE_SPACE_CREATE_INFO };
	ref_space.poseInReferenceSpace = xr_pose_identity;
	ref_space.referenceSpaceType = XR_REFERENCE_SPACE_TYPE_LOCAL;
	result = xrCreateReferenceSpace(xr_session, &ref_space, &xr_app_space);
	if (result != XR_SUCCESS) {
		printf("Failed to create play space!\n");
		return false;
	}

	
	if (prepareSwapchain()) {
		return true;
	}
	else return false; //swapchain preparation failed.
	
	
}


void openXR_app::renderFrame() 
{
	// Block until the previous frame is finished displaying, and is ready for another one.
	// Also returns a prediction of when the next frame will be displayed, for use with predicting
	// locations of controllers, viewpoints, etc.
	XrFrameState frame_state = { XR_TYPE_FRAME_STATE };
	xrWaitFrame(xr_session, nullptr, &frame_state);
	// Must be called before any rendering is done! This can return some interesting flags, like 
	// XR_SESSION_VISIBILITY_UNAVAILABLE, which means we could skip rendering this frame and call
	// xrEndFrame right away.
	xrBeginFrame(xr_session, nullptr);
	
	
	// If the session is active, lets render our layer in the compositor!
	XrCompositionLayerBaseHeader* layer = nullptr;
	XrCompositionLayerProjection  layer_proj = { XR_TYPE_COMPOSITION_LAYER_PROJECTION };
	std::vector<XrCompositionLayerProjectionView> views;

	bool session_active = xr_session_state == XR_SESSION_STATE_VISIBLE || xr_session_state == XR_SESSION_STATE_FOCUSED;
	if (session_active && renderLayer(frame_state.predictedDisplayTime, views, layer_proj)) 
	{
		layer = (XrCompositionLayerBaseHeader*)&layer_proj;
	}
	
	// We're finished with rendering our layer, so send it off for display!
	XrFrameEndInfo end_info{ XR_TYPE_FRAME_END_INFO };
	end_info.displayTime = frame_state.predictedDisplayTime;
	end_info.environmentBlendMode = xr_blend;
	end_info.layerCount = layer == nullptr ? 0 : 1;
	end_info.layers = &layer;
	result = xrEndFrame(xr_session, &end_info);
	if (result != XR_SUCCESS) {
		std::cout << "Failed to submit frame.  xrEndFrame() returned XrResult :" << result << "\n";
	}

}



void openXR_app::GLrendering(XrCompositionLayerProjectionView &view, GLuint surface, GLuint swapchainImage, int eye)
{

	std::vector<float> rotationMatrix = makeRotationMatrix4x4(make_float4(view.pose.orientation.w, view.pose.orientation.x, view.pose.orientation.y, view.pose.orientation.z));
	float4 newUP = MatrixMul(rotationMatrix, make_float4(0, 1, 0, 0));
	float4 newLookDirection = MatrixMul(rotationMatrix, lookDirection);

	m_camera.setEye( make_float3(view.pose.position.x, view.pose.position.y, view.pose.position.z ));	
	m_camera.setFovY((view.fov.angleRight - view.fov.angleLeft) * 180 / M_PI);
	m_camera.setUp(make_float3(newUP.x, newUP.y, newUP.z));
	m_camera.setDirection(normalize(make_float3(newLookDirection.x, newLookDirection.y, newLookDirection.z)));

	m_optixEngine.handleCameraUpdate(&m_camera);
	
	m_optixEngine.launchSubframe();
	sutil::CUDAOutputBuffer<uchar4>* output_buffer = m_optixEngine.GetOutputBuffer();
	GLenum x = glGetError();
	

	glBindFramebuffer(GL_FRAMEBUFFER, m_optixEngine.GetOutputBuffer()->getPBO());
	x = glGetError();

	//glEnable(GL_TEXTURE_2D); // gl error code 1280!
	x = glGetError();

	//Used to Crash caused Here 
	//glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, swapchainImage, 0);
	//and sometimes Crash caused Here 
	//glClear(GL_COLOR_BUFFER_BIT);
 
	glBindTexture(GL_TEXTURE_2D, swapchainImage);

	//Crash Now Caused Here 
	//glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, renderTargetWidth, renderTargetHeight, GL_RGBA, GL_UNSIGNED_BYTE, output_buffer->getHostPointer());
	
	//display left eye view on the desktop window
	//if(eye == 0){	
	//	glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
	//	glBlitFramebuffer(0, 0, 1568, 1568, 0, 0, 1568, 1568, GL_COLOR_BUFFER_BIT, GL_NEAREST);
		
//	}
	glfwSwapBuffers(window);
}

bool openXR_app::renderLayer(XrTime predictedTime, std::vector<XrCompositionLayerProjectionView> &views, XrCompositionLayerProjection &layer)
{

	// Find the state and location of each viewpoint at the predicted time
	uint32_t         view_count = 0;
	XrViewState      view_state = { XR_TYPE_VIEW_STATE };
	XrViewLocateInfo locate_info = { XR_TYPE_VIEW_LOCATE_INFO };
	locate_info.viewConfigurationType = app_config_view;
	locate_info.displayTime = predictedTime;
	locate_info.space = xr_app_space;
	xrLocateViews(xr_session, &locate_info, &view_state, (uint32_t)xr_views.size(), &view_count, xr_views.data());
	views.resize(view_count);
	

	// And now we'll iterate through each viewpoint, and render it!
	for (uint32_t i = 0; i < view_count; i++) {

		// We need to ask which swapchain image to use for rendering! Which one will we get?
		// Who knows! It's up to the runtime to decide.
		uint32_t                    img_id;
		XrSwapchainImageAcquireInfo acquire_info = { XR_TYPE_SWAPCHAIN_IMAGE_ACQUIRE_INFO };
		xrAcquireSwapchainImage(xr_swapchains[i].handle, &acquire_info, &img_id);

		// Wait until the image is available to render to. The compositor could still be
		// reading from it.
		XrSwapchainImageWaitInfo wait_info = { XR_TYPE_SWAPCHAIN_IMAGE_WAIT_INFO };
		wait_info.timeout = XR_INFINITE_DURATION;
		xrWaitSwapchainImage(xr_swapchains[i].handle, &wait_info);

		// Set up our rendering information for the viewpoint we're using right now!
		views[i] = { XR_TYPE_COMPOSITION_LAYER_PROJECTION_VIEW };
		views[i].pose = xr_views[i].pose;
		views[i].fov = xr_views[i].fov;
		views[i].subImage.swapchain = xr_swapchains[i].handle;
		views[i].subImage.imageRect.offset = { 0, 0 };
		views[i].subImage.imageRect.extent = { xr_swapchains[i].width, xr_swapchains[i].height };

		// Call the rendering callback with our view and swapchain info
//		openXR_app::GLrendering(views[i], gl_rendering.framebuffers[i][img_id], xr_swapchains[i].swapchain_images[img_id].image, i, gl_rendering.shader_program_id, gl_rendering.VAO);
		openXR_app::GLrendering(views[i], *(xr_swapchains[i].glFrameBuffers), xr_swapchains[i].swapchain_images[img_id].image, i);

		// And tell OpenXR we're done with rendering to this one!
		XrSwapchainImageReleaseInfo release_info = { XR_TYPE_SWAPCHAIN_IMAGE_RELEASE_INFO };
		xrReleaseSwapchainImage(xr_swapchains[i].handle, &release_info);
	}

	layer.space = xr_app_space;
	layer.viewCount = (uint32_t)views.size();
	layer.views = views.data();
	return true;
	
}



void openXR_app::openxr_poll_events(bool& exit) {
	exit = false;

	XrEventDataBuffer event_buffer = { XR_TYPE_EVENT_DATA_BUFFER };

	while (xrPollEvent(xr_instance, &event_buffer) == XR_SUCCESS) {
		switch (event_buffer.type) {
		case XR_TYPE_EVENT_DATA_SESSION_STATE_CHANGED: {
			XrEventDataSessionStateChanged* changed = (XrEventDataSessionStateChanged*)&event_buffer;
			xr_session_state = changed->state;

			// Session state change is where we can begin and end sessions, as well as find quit messages!
			switch (xr_session_state) {
			case XR_SESSION_STATE_READY: {
				XrSessionBeginInfo begin_info = { XR_TYPE_SESSION_BEGIN_INFO };
				begin_info.primaryViewConfigurationType = app_config_view;
				xrBeginSession(xr_session, &begin_info);
				xr_running = true;
			} break;
			case XR_SESSION_STATE_STOPPING: {
				xr_running = false;
				xrEndSession(xr_session);
			} break;
			case XR_SESSION_STATE_EXITING:      exit = true;              break;
			case XR_SESSION_STATE_LOSS_PENDING: exit = true;              break;
			}
		} break;
		case XR_TYPE_EVENT_DATA_INSTANCE_LOSS_PENDING: exit = true; return;
		}
		event_buffer = { XR_TYPE_EVENT_DATA_BUFFER };
	}
}

bool openXR_app::buildEngine() 
{
	
	m_camera.setEye(make_float3(0.0f, 0.0f, 0.0f));
	m_camera.setLookat(make_float3(0.0f, 0.0f, -1.0f));
	m_camera.setUp(make_float3(0.0f, 1.0f, 0.0f));
	m_camera.setFovY(60.0f);
	lookDirection = make_float4(m_camera.direction().x, m_camera.direction().y, m_camera.direction().z, 0);
	m_camera.setAspectRatio(1);

	//m_optixEngine = RenderEngine::RenderEngine(1568, 1568);
	m_optixEngine = RenderEngine::RenderEngine(renderTargetWidth, renderTargetHeight);
	m_optixEngine.buildEngine();
	m_optixEngine.handleCameraUpdate(&m_camera);
	
	return true;
}



///////////////////////////////////////////


void openXR_app::swapchain_destroy(swapchain_t &swapchain) {
	
	xr_swapchains.clear();
}

void openXR_app::openxr_shutdown() {
	// We used a graphics API to initialize the swapchain data, so we'll
	// give it a chance to release anythig here!
	for (int32_t i = 0; i < xr_swapchains.size(); i++) {
		xrDestroySwapchain(xr_swapchains[i].handle);
		swapchain_destroy(xr_swapchains[i]);
	}
	xr_swapchains.clear();

		
		

	// Release all the other OpenXR resources that we've created!
	// What gets allocated, must get deallocated!
	
	if (xr_app_space != XR_NULL_HANDLE) xrDestroySpace(xr_app_space);
	if (xr_session != XR_NULL_HANDLE) xrDestroySession(xr_session);
	if (xr_debug != XR_NULL_HANDLE) ext_xrDestroyDebugUtilsMessengerEXT(xr_debug);
	if (xr_instance != XR_NULL_HANDLE) xrDestroyInstance(xr_instance);

	glfwSetWindowShouldClose(window, GL_FALSE);
}


/// helper funcions ///////////////
bool openXR_app::get_xr_running() {
	return xr_running;
}

XrSessionState openXR_app::get_xr_session_state() {
	return xr_session_state;
}


void openXR_app::print_instance_properties(XrInstance instance)
{
	XrResult result;
	XrInstanceProperties instance_props = { XR_TYPE_INSTANCE_PROPERTIES, NULL };

	result = xrGetInstanceProperties(instance, &instance_props);
	if (result != XR_SUCCESS)
		printf("FAILED TO GET XR INSTANCE INFO.");

	printf("Runtime Name: %s\n", instance_props.runtimeName);
	printf("Runtime Version: %d.%d.%d\n", XR_VERSION_MAJOR(instance_props.runtimeVersion),
		XR_VERSION_MINOR(instance_props.runtimeVersion),
		XR_VERSION_PATCH(instance_props.runtimeVersion));
}

void openXR_app::print_system_properties(XrSystemProperties* system_properties)
{
	printf("System properties for system %lu: \"%s\", vendor ID %d\n", system_properties->systemId,
		system_properties->systemName, system_properties->vendorId);
	printf("\tMax layers          : %d\n", system_properties->graphicsProperties.maxLayerCount);
	printf("\tMax swapchain height: %d\n",
		system_properties->graphicsProperties.maxSwapchainImageHeight);
	printf("\tMax swapchain width : %d\n",
		system_properties->graphicsProperties.maxSwapchainImageWidth);
	printf("\tOrientation Tracking: %d\n", system_properties->trackingProperties.orientationTracking);
	printf("\tPosition Tracking   : %d\n", system_properties->trackingProperties.positionTracking);
}

void openXR_app::print_viewconfig_view_info(uint32_t view_count, XrViewConfigurationView* viewconfig_views)
{
	for (uint32_t i = 0; i < view_count; i++) {
		printf("View Configuration View %d:\n", i);
		printf("\tResolution       : Recommended %dx%d, Max: %dx%d\n",
			viewconfig_views[0].recommendedImageRectWidth,
			viewconfig_views[0].recommendedImageRectHeight, viewconfig_views[0].maxImageRectWidth,
			viewconfig_views[0].maxImageRectHeight);
		printf("\tSwapchain Samples: Recommended: %d, Max: %d)\n",
			viewconfig_views[0].recommendedSwapchainSampleCount,
			viewconfig_views[0].maxSwapchainSampleCount);
	}
}
//currently valid for only 4x4 rotation matrix * 4x1 vector matrix
//rotationMatrix Matrix as vector
/// {
/// [0] [1] [2] [3]
/// [4] [5] [6] [7]
/// [8] [9] [10] [11]
/// [12] [13] [14] [15]
/// } * {
/// [w]
/// [x]
/// [y]
/// [z]
/// }

float4 openXR_app::MatrixMul(std::vector<float> rotationMatrix, float4 vector4) {
	float4 newVec;
	newVec.x = (rotationMatrix[0] * vector4.x) + (rotationMatrix[1] * vector4.y) + (rotationMatrix[2] * vector4.z) + (rotationMatrix[3] * vector4.w);
	newVec.y = (rotationMatrix[4] * vector4.x) + (rotationMatrix[5] * vector4.y) + (rotationMatrix[6] * vector4.z) + (rotationMatrix[7] * vector4.w);
	newVec.z = (rotationMatrix[8] * vector4.x) + (rotationMatrix[9] * vector4.y) + (rotationMatrix[10] * vector4.z) + (rotationMatrix[11] * vector4.w);
	newVec.w = (rotationMatrix[12] * vector4.x) + (rotationMatrix[13] * vector4.y) + (rotationMatrix[14] * vector4.z) + (rotationMatrix[15] * vector4.w);
	return newVec;
}

/// <summary>
/// Converts Quaternion w,x,y,z into its 4x4 rotation matrix
/// rotationMatrix Matrix as vector
/// {
/// [0] [1] [2] [3]
/// [4] [5] [6] [7]
/// [8] [9] [10] [11]
/// [12] [13] [14] [15]
/// }
/// </summary>
/// <param name="quaternion"></param>
/// <returns> rotation Matrix </returns>
std::vector<float> openXR_app::makeRotationMatrix4x4(float4 quaternion) {
	std::vector<float> rotationMatrix;
	rotationMatrix.resize(16);
	float q0 = quaternion.x;
	float q1 = quaternion.y;
	float q2 = quaternion.z;
	float q3 = quaternion.w;

	rotationMatrix[0] = 1 - (2 * pow(q2, 2)) - (2 * pow(q3, 2));
	rotationMatrix[1] = (2 * q1 * q2) - (2 * q0 * q3);
	rotationMatrix[2] = (2 * q1 * q3) + (2 * q0 * q2);
	rotationMatrix[3] = 0;

	rotationMatrix[4] = (2 * q1 * q2) + (2 * q0 * q3);
	rotationMatrix[5] = 1 - (2 * pow(q1, 2)) - (2 * pow(q3, 2));
	rotationMatrix[6] = (2 * q2 * q3) - (2 * q0 * q1);
	rotationMatrix[7] = 0;

	rotationMatrix[8] = (2 * q1 * q3) - (2 * q0 * q2);
	rotationMatrix[9] = (2 * q2 * q3) + (2 * q0 * q1);
	rotationMatrix[10] = 1 - (2 * pow(q1, 2)) - (2 * pow(q2, 2));
	rotationMatrix[11] = 0;

	rotationMatrix[12] = 0;
	rotationMatrix[13] = 0;
	rotationMatrix[14] = 0;
	rotationMatrix[15] = 1;

	return rotationMatrix;

}


