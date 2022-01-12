To View lightfields, first build visual studio solution files (Visual Studio 2019 used) using CMake.

Then build the project and set startup file within solution to optixLightFieldViewer.

Now within bin folder create a text file coalled config.txt

Within this file you explain the lightfiled which you wish to use

	{Light field file name/path within the bin folder} {width in hogels of lightfield} {heightin hogels of lightfield} {Fov of the lightfield}

after which you place lightfield within the bin folder aswell

after which you can run program and view the light field