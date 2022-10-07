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
#include <optixLightFieldViewer/VirtualReality.h>
#include <glad/glad.h>  // Needs to be included before gl_interop

#include "opencv2/core.hpp" 
#include "opencv2/opencv.hpp"

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <sutil/CUDAOutputBuffer.h>
#include <sutil/Camera.h>
#include <sutil/Exception.h>
#include <sutil/GLDisplay.h>
#include <sutil/Matrix.h>
#include <sutil/Trackball.h>
#include <sutil/sutil.h>

#include <array>
#include <GLFW/glfw3.h>

#include <imgui/imgui.h>
#include <imgui/imgui_impl_glfw.h>
#include <imgui/imgui_impl_opengl3.h>


#include <cuda/whitted.h>

#include <fstream>

#include "optixLightFieldViewer.h"

#include "RenderEngine.h"
#include "RecordData.h"
#include "ctrlMap.h"

void lightFieldViewer::initCameraState()
{
    m_camera.setEye(make_float3(0.0f, 0.0f, 0.0f));
    m_camera.setLookat(make_float3(0.0f, 0.0f, -1.0f/10));
    m_camera.setUp(make_float3(0.0f, 1.0f, 0.0f));
    m_camera.setFovY(60.0f);
    m_camera_changed = true;

    m_trackball.setCamera(&m_camera);
    m_trackball.setMoveSpeed(10.0f);
    m_trackball.setReferenceFrame(make_float3(1.0f, 0.0f, 0.0f), make_float3(0.0f, 0.0f, 1.0f), make_float3(0.0f, 1.0f, 0.0f));
    m_trackball.setGimbalLock(true);
}

void lightFieldViewer::updateState()
{
    // Update params on device
    if (m_camera_changed)
    {
        m_optixEngine.handleCameraUpdate(&m_camera);
        m_camera_changed = false;
    }
    else if (m_resize_dirty)
    {
        m_optixEngine.handleResize();
        m_resize_dirty = false;
    }
}

void lightFieldViewer::performDescreteControl(int ctrl)
{
    float moveRatio = 10 * m_shifted;
    switch (ctrl)
    {
    case ctrlMap::ctrls::exit:
        std::cout << "Hello VR, exit! \n";
        glfwSetWindowShouldClose(m_window, true);
        break;

    case ctrlMap::ctrls::forward:
        m_camera.setEye(m_camera.eye() + m_camera.direction() / moveRatio);
        m_camera.setLookat(m_camera.lookat() + m_camera.direction() / moveRatio);
        m_camera_changed = true;
        break;
        //Backwards

    case ctrlMap::ctrls::back:
        m_camera.setEye(m_camera.eye() - m_camera.direction() / moveRatio);
        m_camera.setLookat(m_camera.lookat() - m_camera.direction() / moveRatio);
        m_camera_changed = true;
        break;
        //Left

    case ctrlMap::ctrls::left:
        float3 right = normalize(cross(m_camera.direction(), m_camera.up()));
        m_camera.setEye(m_camera.eye() - right / moveRatio);
        m_camera.setLookat(m_camera.lookat() - right / moveRatio);
        m_camera_changed = true;
        break;
        //Right

    case ctrlMap::ctrls::right:
    {
        float3 right = normalize(cross(m_camera.direction(), m_camera.up()));
        m_camera.setEye(m_camera.eye() + right / moveRatio);
        m_camera.setLookat(m_camera.lookat() + right / moveRatio);
        m_camera_changed = true;
        break;
    }
    //Up

    case ctrlMap::ctrls::up:
        m_camera.setEye(m_camera.eye() + m_camera.up() / moveRatio);
        m_camera.setLookat(m_camera.lookat() + m_camera.up() / moveRatio);
        m_camera_changed = true;
        break;
        //Down

    case ctrlMap::ctrls::down:
        m_camera.setEye(m_camera.eye() - m_camera.up() / moveRatio);
        m_camera.setLookat(m_camera.lookat() - m_camera.up() / moveRatio);
        m_camera_changed = true;
        break;

    case ctrlMap::ctrls::fastCamOn:
        // is_fastCam = true;
        m_shifted = std::max(0.2f, (float)(m_shifted * 0.8));
        break;

    case ctrlMap::ctrls::fastCamOff:
        m_shifted = 1;
        break;

    case ctrlMap::ctrls::load:
        m_optixEngine.updateTexture(sutil::getCurrFilename());
        break;


    default:
        break;
    }
}
void lightFieldViewer::performAnalogControl(int ctrl, std::pair<float, float> value)
{
    float moveRatio = 10 * m_shifted;
    switch (ctrl)
    {
    case ctrlMap::ctrls::rotAround:
        if (!m_tracking) 
        {
            m_tracking = true;
            m_trackball.startTracking(static_cast<int>(value.first), static_cast<int>(value.second));
            break;
        }
       // m_trackball.setViewMode(sutil::Trackball::LookAtFixed);
      //  m_trackball.updateTracking(static_cast<int>(value.first), static_cast<int>(value.second), m_optixEngine.GetState()->params.width, m_optixEngine.GetState()->params.width);
       // m_camera_changed = true;
        break;

    case ctrlMap::ctrls::turnCamera:
        if (!m_tracking)
        {
            m_tracking = true;
            m_trackball.startTracking(static_cast<int>(value.first), static_cast<int>(value.second));
            break;
        }
        break;

    case ctrlMap::ctrls::scroll:
        if (m_trackball.wheelEvent((int)value.second))
            m_camera_changed = true;
        break;

    case ctrlMap::ctrls::endTracking:
        m_tracking = false;
        break;

    default:
        break;
    }
};


void lightFieldViewer::displaySubframe( sutil::CUDAOutputBuffer<uchar4>& output_buffer, sutil::GLDisplay& gl_display, GLFWwindow* window )
{
    glfwMakeContextCurrent(window);
    // Display
    int framebuf_res_x = 0;  // The display's resolution (could be HDPI res)
    int framebuf_res_y = 0;  //
    glfwGetFramebufferSize( window, &framebuf_res_x, &framebuf_res_y );
    gl_display.display(output_buffer.width(), output_buffer.height(), framebuf_res_x, framebuf_res_y, output_buffer.getPBO() );
}


GLFWwindow* lightFieldViewer::build(sutil::CUDAOutputBufferType bufType, std::string file)
{
    try
    {
        initCameraState();
        m_optixEngine.buildEngine();
        if (file.empty())
        {
             m_window = sutil::initUI("Real Time Lightfield Render", m_optixEngine.GetDisplayWidth(), m_optixEngine.GetDisplayHeight());
            // m_window = sutil::initUI("Real Time Lightfield Render", 768, 768);
          //m_optixEngine.buildEngine();
            return m_window;
        }
        else
        {
            if (bufType == sutil::CUDAOutputBufferType::GL_INTEROP)
            {
                sutil::initGLFW();  // For GL context
                sutil::initGL();
            }

            updateState();
            m_optixEngine.launchSubframe();

            sutil::ImageBuffer buffer;
            buffer.data = m_optixEngine.GetOutputBuffer()->getHostPointer();
            buffer.width = m_optixEngine.GetOutputBuffer()->width();
            buffer.height = m_optixEngine.GetOutputBuffer()->height();
            buffer.pixel_format = sutil::BufferImageFormat::UNSIGNED_BYTE4;
           // sutil::saveImage(outfile.c_str(), buffer, false);

            if (bufType == sutil::CUDAOutputBufferType::GL_INTEROP)
            {
                glfwTerminate();
            }
            return NULL;
        }
    }
    catch (std::exception& e)
    {
        std::cerr << "Caught exception: " << e.what() << "\n";
        exit;
    }
}

void lightFieldViewer::renderLoop()
{
    
    std::chrono::duration<double> state_update_time(0.0);
    std::chrono::duration<double> render_time(0.0);
    std::chrono::duration<double> display_time(0.0);
    //m_optixEngine.setOutputBuffer(sutil::CUDAOutputBufferType::GL_INTEROP);
    sutil::GLDisplay gl_display;

    do
    {
        auto t0 = std::chrono::steady_clock::now();

        glfwPollEvents();

        //updateState( output_buffer, state );
        updateState();
        auto t1 = std::chrono::steady_clock::now();
        state_update_time += t1 - t0;
        t0 = t1;

        m_optixEngine.launchSubframe();
        //                  launchSubframe( output_buffer, state );
        t1 = std::chrono::steady_clock::now();
        render_time += t1 - t0;
        t0 = t1;
        displaySubframe(*m_optixEngine.GetOutputBuffer(), gl_display, m_window);

        t1 = std::chrono::steady_clock::now();
        display_time += t1 - t0;

        sutil::displayStats(state_update_time, render_time, display_time);

        bool changeState = sutil::getChangeState();
        
        if (changeState)
        {
            performDescreteControl(m_ctrlMapping.getctrl(std::pair<int, int>(-6, 1)));
        }

        if (sutil::get_launchVR()) {
            openXR_app app(m_window, &m_optixEngine);
            app.launchApp();
        }

        glfwSwapBuffers(m_window);

        ++m_optixEngine.GetState()->params.subframe_index;
    } while (!glfwWindowShouldClose(m_window));
}
