To View lightfields, 
1.) Install Open CV from here: https://opencv.org/releases/

	i.) Once Downloaded, within OpenCV copy build folder and move it to \Optix7.3.0LightFIeldViewer\SDK\lib\
	ii.) Rename folder to OpenCV

2.)Now build visual studio solution files (Visual Studio 2019 used) using CMake.
	- Also need to define an environment Variables related to open cv and add another to Path

		i.) To Path env variable add : PATHTOSimulator + \Optix7.3.0LightFIeldViewer\SDK\lib\OpenCV\x64\vc15\bin

		ii.) Create environment variable "OPENCV_IO_MAX_IMAGE_PIXELS" and assign its value to: 1850000000000


3.) Then build the project and set startup file within solution to optixLightFieldViewer.

	Note: You can build in either debug or release without issue
	
	Note: You may need to restart visual studio after building the project

4.) Within bin folder created from building the project, go into the Debug/ Release folder and create a text file called config.txt

5.) Inside Config.txt you describe the lightfield you want to view

Example : {Light field file name/path within the bin folder} {width in hogels of lightfield} {height in hogels of lightfield} {Fov of the lightfield}

		inputData/Chess(600X600)FOV40_3.png   600    600     40

6.) Place lightfiled image into bin folder

7.) after which you can run program through the "optixLightFieldViewer.exe" found within the bin sub folders

Note: To run within visual studio (debug mode), changing of the working directory is needed, needs to be changed to the location of the config.txt file 
	Can be done through Properties within OptixLightFieldViewer, under the debug tab