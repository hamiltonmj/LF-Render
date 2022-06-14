
#include <optix.h>
#include <cuda/whitted.h>
#include <sutil/CUDAOutputBuffer.h>
#include <sutil/Trackball.h>
#include <GLFW/glfw3.h>
#include "optixLightFieldViewer.h"


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
