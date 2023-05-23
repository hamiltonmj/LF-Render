#include "fileReader.h"



void FileReader::loadTexFile(std::string fileName, RenderState &engineState)
{
    //if file selected is .txt file, then directly load the file.
    std::shared_ptr<TextureBase> tex;
    if (is_textFile(fileName)) 
    {
        std::ifstream file;
        file.open(fileName);
        if (!file.is_open())
        {
            std::cout << "Error Opening Lightfield Config: " << fileName << "\n";
            exit(-1);
        }

        std::string name;
//        unsigned width, height, fov;
        file >> name;


        if(name == "!VIDEO!")
        {            
            engineState.texObjects = *readVideoToVram(fileName, file);
/*
            size_t pathEnd = fileName.rfind("\\");
            pathEnd = (std::string::npos == pathEnd) ? fileName.rfind("/") : pathEnd;

            fileName = fileName.substr(0, pathEnd);
            std::vector<cv::String> framePaths;
            cv::glob(fileName, framePaths, false);

            for (size_t x = 0; x < framePaths.size(); x++)
            {
                cv::String imageName = framePaths[x];
                if (is_textFile(imageName)) continue;

                std::cout << "Opening Texture As LightField: " << imageName << "\n";
               // tex = std::make_shared<LightFieldData>(loadImageToRam(imageName), width, height, fov);


               // m_state.texObjects.insert({ "f" + std::to_string(x),tex });
                std::cout << "hi";
            }
*/
        }
        else
        {
            //readLFTexToVram( name, file);
            std::cout << "Opening Texture As LightField: " << fileName << "\n";
            engineState.texObjects.insert({ fileName,readLFTexToVram(name, file) });
//            tex = std::make_shared<LightFieldData>(loadImageToRam(name), width, height, fov);
//            m_state.texObjects.insert({ fileName,tex });        
        }
    }
    else
    {
        //readImageTexToVram(fileName);
        engineState.texObjects.insert({ fileName,readImageTexToVram(fileName) });

/*
        std::cout << "Opening Texture As Image: " << fileName << "\n";
        tex = std::make_shared<TextureData>(loadImageToRam(fileName));
        m_state.texObjects.insert({ fileName,tex });
*/
    }
}



cv::Mat FileReader::loadImageToRam(std::string fileName)
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
    std::cout << "Succesfully loaded : " << fileName << " to ram \n";

    return image;
}

bool FileReader::is_textFile(std::string fileName)
{
    std::size_t found = fileName.find("txt");
    if (found != std::string::npos)
    {
        return fileName.substr(found, fileName.length()) == "txt";
    }
    return false;
}


///TextureAsTexture
std::shared_ptr<TextureBase> FileReader::readImageTexToVram( std::string fileName)
{
    std::cout << "Opening Texture As Image: " << fileName << "\n";
    return std::make_shared<TextureData>(loadImageToRam(fileName));
   
    //HERE  m_state.texObjects.insert({ fileName,tex });
}


//loadLightFieldtoMemory
std::shared_ptr<TextureBase> FileReader::readLFTexToVram(std::string IntegralImage, std::ifstream &file)
{
    unsigned width, height, fov;
    file >> width >> height >> fov;

    return  std::make_shared<LightFieldData>(loadImageToRam(IntegralImage), width, height, fov);
}


//read Video 
std::map<std::string, std::shared_ptr<TextureBase>>* FileReader::readVideoToVram( std::string fileName, std::ifstream &file)
{
    unsigned width, height, fov;
    file >> width >> height >> fov;
    size_t pathEnd = fileName.rfind("\\");

    pathEnd = (std::string::npos == pathEnd) ? fileName.rfind("/") : pathEnd;
    fileName = fileName.substr(0, pathEnd);
    std::vector<cv::String> framePaths;
    cv::glob(fileName, framePaths, false);

    std::shared_ptr<TextureBase> tex;
    std::map<std::string, std::shared_ptr<TextureBase>>* texObjects = new std::map<std::string, std::shared_ptr<TextureBase>>();

    for (size_t x = 0; x < framePaths.size(); x++)
    {
        cv::String imageName = framePaths[x];
        if (is_textFile(imageName)) continue;

        std::cout << "Opening Texture As LightField: " << imageName << "\n";
        tex = std::make_shared<LightFieldData>(loadImageToRam(imageName), width, height, fov);
        texObjects->insert({ "f" + std::to_string(x),tex });
    }
    return texObjects;
}
