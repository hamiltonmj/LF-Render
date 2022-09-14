//#include "VirtualReality.cpp"
#include <optixLightFieldViewer/VirtualReality.h>
#include <optix.h>
#include <cuda/whitted.h>
#include <sutil/CUDAOutputBuffer.h>
#include <sutil/Trackball.h>
#include <GLFW/glfw3.h>
#include "optixLightFieldViewer.h"
#include <thread>
#include <chrono> 
#include <iostream>   




lightFieldViewer lFViewer = lightFieldViewer();

//------------------------------------------------------------------------------
//
// GLFW callbacks
//
//------------------------------------------------------------------------------

void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods)
{
    lFViewer.m_mouse_button = (action == 1) ? button : -1;

    double xpos, ypos;
    glfwGetCursorPos(window, &xpos, &ypos);

    //checks ImGui is using mouse or not
    if (!sutil::Get_is_ImGuiActive())
    {
        lFViewer.performAnalogControl(lFViewer.m_ctrlMapping.getctrl(std::pair<int, int>(button, action)), std::pair<double, double>(xpos, ypos));
    }
}

void cursorPosCallback(GLFWwindow* window, double xpos, double ypos)
{
    whitted::LaunchParams* params = static_cast<whitted::LaunchParams*>(glfwGetWindowUserPointer(window));


    //checks ImGui is using mouse or not
    if (!sutil::Get_is_ImGuiActive())
    {
        if (lFViewer.m_mouse_button == GLFW_MOUSE_BUTTON_LEFT)
        {
            lFViewer.m_trackball.setViewMode(sutil::Trackball::LookAtFixed);
            lFViewer.m_trackball.updateTracking(static_cast<int>(xpos), static_cast<int>(ypos), params->width, params->height);
            lFViewer.m_camera_changed = true;
        }

        else if (lFViewer.m_mouse_button == GLFW_MOUSE_BUTTON_RIGHT)
        {
            lFViewer.m_trackball.setViewMode(sutil::Trackball::EyeFixed);
            lFViewer.m_trackball.updateTracking(static_cast<int>(xpos), static_cast<int>(ypos), params->width, params->height);
            lFViewer.m_camera_changed = true;
        }
    }

}

void keyCallback(GLFWwindow* window, int32_t key, int32_t /*scancode*/, int32_t action, int32_t mods)
{
    //Since our application does not have any repeat commands we use this to simplify
    //Really should add acceleration to our code but not worth the hassle
    int act = action > 0;
    lFViewer.performDescreteControl(lFViewer.m_ctrlMapping.getctrl(std::pair<int, int>(key, act)));
}

void scrollCallback(GLFWwindow* window, double xscroll, double yscroll)
{
    lFViewer.performAnalogControl(lFViewer.m_ctrlMapping.getctrl(std::pair<int, int>(-5, 1)), std::pair<float, float>(xscroll, yscroll));
}

void windowSizeCallback(GLFWwindow* window, int32_t res_x, int32_t res_y)
{
    // Keep rendering at the current resolution when the window is minimized.
    if (lFViewer.m_minimized)
        return;

    // Output dimensions must be at least 1 in both x and y.
    sutil::ensureMinimumSize(res_x, res_y);

    whitted::LaunchParams* params = static_cast<whitted::LaunchParams*>(glfwGetWindowUserPointer(window));
    params->width = res_x;
    params->height = res_y;
    lFViewer.m_camera_changed = true;
    lFViewer.m_resize_dirty = true;
}


void windowIconifyCallback(GLFWwindow* window, int32_t iconified)
{
    lFViewer.m_minimized = (iconified > 0);
}






void addCallbacks(GLFWwindow* window)
{
    glfwSetCursorPosCallback(window, cursorPosCallback);
    glfwSetMouseButtonCallback(window, mouseButtonCallback);
    glfwSetWindowSizeCallback(window, windowSizeCallback);
    glfwSetWindowIconifyCallback(window, windowIconifyCallback);
    glfwSetKeyCallback(window, keyCallback);
    glfwSetScrollCallback(window, scrollCallback);
    glfwSetWindowUserPointer(window, lFViewer.m_optixEngine.GetState());
}


//------------------------------------------------------------------------------
//
//  Helper functions
//
//------------------------------------------------------------------------------

void printUsageAndExit(const char* argv0)
{
    std::cerr << "Usage  : " << argv0 << " [options]\n";
    std::cerr << "Options: --file | -f <filename>      File for image output\n";
    std::cerr << "         --no-gl-interop             Disable GL interop for display\n";
    std::cerr << "         --dim=<width>x<height>      Set image dimensions; defaults to 768x768\n";
    std::cerr << "         --help | -h                 Print this usage message\n";
    exit(0);
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

float4 MatrixMul(std::vector<float> rotationMatrix, float4 vector4) {
    float4 newVec;
    newVec.x = (rotationMatrix[0] * vector4.x) + (rotationMatrix[1] * vector4.x) + (rotationMatrix[2] * vector4.x) + (rotationMatrix[3] * vector4.x);
    newVec.y = (rotationMatrix[4] * vector4.y) + (rotationMatrix[5] * vector4.y) + (rotationMatrix[6] * vector4.y) + (rotationMatrix[7] * vector4.y);
    newVec.z = (rotationMatrix[8] * vector4.z) + (rotationMatrix[9] * vector4.z) + (rotationMatrix[10] * vector4.z) + (rotationMatrix[11] * vector4.z);
    newVec.w = (rotationMatrix[12] * vector4.w) + (rotationMatrix[13] * vector4.w) + (rotationMatrix[14] * vector4.w) + (rotationMatrix[15] * vector4.w);

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
/// <returns></returns>
std::vector<float> makeRotationMatrix4x4(float4 quaternion) {
    std::vector<float> rotationMatrix;
    rotationMatrix.resize(16);
    float q0 = quaternion.x;
    float q1 = quaternion.y;
    float q2 = quaternion.z;
    float q3 = quaternion.w;

    rotationMatrix[0] = 1 - (2 * pow(q2,2)) - (2 * pow(q3,2));
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

int main(int argc, char* argv[])
{
    /*
    std::vector<float> rotationMatrix = makeRotationMatrix4x4(make_float4(0.9164, -0.2582, 0.7746, 0.2582));
    std::cout << "matrix rotation \n";
    for (int i = 0; i < 16; i++) {
        std::cout << rotationMatrix[i] << "\n";
    }
    std::cout << "------------" << "\n";

    float4 dir = MatrixMul(rotationMatrix, make_float4(1, 6, -1, 0));
    std::cout << dir.x << "\n";
    std::cout << dir.y << "\n";
    std::cout << dir.z << "\n";
    std::cout << dir.w << "\n";


    */
   
    openXR_app app;
   

    if (!app.openxr_init("VR_app")) {
        MessageBox(nullptr, "OpenXR initialization failed\n", "Error", 1);
        return 1;
    }

    app.buildEngine();
    bool quit = false;

    while(!quit){

   
       

       
       
            app.openxr_poll_events(quit);
            //printf("*********************************App lunched. \n");

            if (app.get_xr_running()) {
                app.renderFrame();
               // printf("#####################################App is running. \n");

                //XrSessionState xr_session_state = app.get_xr_session_state();
                if (app.get_xr_session_state() != XR_SESSION_STATE_VISIBLE &&
                    app.get_xr_session_state() != XR_SESSION_STATE_FOCUSED) {
                
                    std::this_thread::sleep_for(std::chrono::milliseconds(250));
                }
            }
    
        

       
   
    }

    app.openxr_shutdown();
    
    return 0;
    
}




/*
int main(int argc, char* argv[])
{
    sutil::CUDAOutputBufferType output_buffer_type = sutil::CUDAOutputBufferType::GL_INTEROP;

    //
    // Parse command line options
    //
    std::string outfile;

    for (int i = 1; i < argc; ++i)
    {
        const std::string arg = argv[i];
        if (arg == "--help" || arg == "-h")
        {
            printUsageAndExit(argv[0]);
        }
        else if (arg == "--no-gl-interop")
        {
            output_buffer_type = sutil::CUDAOutputBufferType::CUDA_DEVICE;
        }
        else if (arg == "--file" || arg == "-f")
        {
            if (i >= argc - 1)
                printUsageAndExit(argv[0]);
            outfile = argv[++i];
        }
        else if (arg.substr(0, 6) == "--dim=")
        {
            const std::string dims_arg = arg.substr(6);
            int               w, h;
            sutil::parseDimensions(dims_arg.c_str(), w, h);
            lFViewer.m_optixEngine.GetState()->params.width = w;
            lFViewer.m_optixEngine.GetState()->params.height = h;
        }
        else
        {
            std::cerr << "Unknown option '" << argv[i] << "'\n";
            printUsageAndExit(argv[0]);
        }
    }

    try
    {

         //GLFWwindow* window = sutil::initUI("Real Time Lightfield Render", 768, 768);
         lFViewer.build(output_buffer_type, outfile);
         auto window = lFViewer.build(output_buffer_type, outfile);
         if (window)
             {
               addCallbacks(window);
               lFViewer.renderLoop();
               }
      

        
    }
    catch (std::exception& e)
    {
        std::cerr << "Caught exception: " << e.what() << "\n";
        exit;
    }

    return 0;
}
*/
