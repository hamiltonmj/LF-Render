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

#pragma once
#include <sutil/CUDAOutputBuffer.h>
#include <sutil/Camera.h>
#include "ctrlMap.h"
#include <sutil/GLDisplay.h>
#include "RenderEngine.h"
#include "RecordData.h"


class lightFieldViewer
{
public:
	ctrlMap m_ctrlMapping = ctrlMap();
	GLFWwindow* m_window;
	RenderEngine m_optixEngine;

	sutil::Camera    m_camera;
	sutil::Trackball m_trackball;

	bool m_camera_changed = true;
	bool m_resize_dirty = false;

	bool m_tracking = false;
	float m_shifted = 1;
	bool m_minimized = false;

	// Mouse state
	int32_t m_mouse_button = -1;	
	
	lightFieldViewer()
		: m_optixEngine(RenderEngine()) {}

	/// <summary>
	/// Places Camera in its inital position within the scene and sets up its vectors
	/// </summary>
	void lightFieldViewer::initCameraState();

	/// <summary>
	/// Checks to see if any input has been given and the nupdates the renderEngines parameters to reflect it 
	/// </summary>
	void lightFieldViewer::updateState();

	/// <summary>
	/// Given a ctrl will perform the action given from it 
	/// Specifically this looks at a signle value from the control ie on/off
	/// </summary>
	/// <param name="ctrl"> Control to be performed</param>
	void lightFieldViewer::performDescreteControl(int ctrl);

	/// <summary>
	/// Given a ctrl will perform the action given from it 
	/// Specifically this looks at at continous values given by a controller ie mouse position
	/// and will apply based on non descrete values
	/// </summary>
	/// <param name="ctrl">Control to be performed</param>
	/// <param name="value"> Analog value given from the crtl</param>
	void lightFieldViewer::performAnalogControl(int ctrl, std::pair<float, float> value);

	/// <summary>
	/// Given a buffer will display it to the screen
	/// </summary>
	/// <param name="output_buffer"> Buffer to be displayed</param>
	/// <param name="gl_display"> the gl_display object we are using to display </param>
	/// <param name="window"> The window to display our object in</param>
	void lightFieldViewer::displaySubframe(sutil::CUDAOutputBuffer<uchar4>& output_buffer, sutil::GLDisplay& gl_display, GLFWwindow* window);

	/// <summary>
	/// Builds the entire lightfield viewer application
	/// </summary>
	/// <param name="bufType"> Buffer type being used for output</param>
	/// <param name="file"> if we just wantto save to a file we can</param>
	/// <returns> returns a glfwWindow object which can be used to add eventhandleers and such too</returns>
	GLFWwindow* lightFieldViewer::build( sutil::CUDAOutputBufferType bufType, std::string file);

	/// <summary>
	/// THis function performs the entire renderlooping and will only return when the user closes the lightfield viewer
	/// </summary>
	void lightFieldViewer::renderLoop();
};