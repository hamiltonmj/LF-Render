#pragma once 
#include "opencv2/core.hpp" 
#include "opencv2/opencv.hpp"

#include <filesystem>
#include <fstream>
#include <map>

#include "TextureDataTypes.h"
#include "RecordData.h"
#include "RenderState.h"

class FileReader
{
public:
	FileReader() {};
	void FileReader::loadTexFile(std::string fileName, RenderState &engineState);

	bool FileReader::is_textFile(std::string fileName);
	cv::Mat FileReader::loadImageToRam(std::string fileName);
	
	std::shared_ptr<TextureBase> FileReader::readLFTexToVram(std::string IntegralImage, std::ifstream &file);
	std::shared_ptr<TextureBase> FileReader::readImageTexToVram(std::string fileName);

	std::map<std::string, std::shared_ptr<TextureBase>>* FileReader::readVideoToVram(std::string fileName, std::ifstream &file);
};


