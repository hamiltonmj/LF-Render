from PIL import Image
import time
import configparser

timeStart = time.time()


config = configparser.ConfigParser()
config.read("config.cfg")

# The inputted Pixel to be represented 
pixelX = int(config["Input"]["pixelX"])
pixelY = int(config["Input"]["pixelY"])

# The inputted Capture Resolution of the light field 
capResX = int(config["LightField Parameters"]["lightFieldX"])
capResY = int(config["LightField Parameters"]["lightFieldY"])

# The inputted number of hogels per row/column
hogelRow = int(config["LightField Parameters"]["rowHogel"])
hogelColumn = int(config["LightField Parameters"]["columnHogel"])

#The resolution of the outputted image
outResX = int(config["Output"]["renderResolutionX"])
outResY = int(config["Output"]["renderResolutionY"])

outName = config["Output"]["fileName"]

inImage = Image.open(config["Input"]["fileName"])

#Prevents invalid input from the config file
if(capResX <= 0):
    capResX = int(inImage.width)
if(capResY <= 0):
    capResY = int(inImage.height)

#calculates the resolution per hogel based on inputted config 
resPerHogelX = int(capResX/ hogelRow)
resPerHogelY = int(capResY/ hogelColumn)

#Prevents invalid input from the config file
if(outResX <= 0):
    outResX = int(resPerHogelX)
if(outResY <= 0):
    outResY = int(resPerHogelY)

#Creates the image to represent a finished view
outImage = Image.new("RGBA",(hogelRow, hogelColumn) , (0, 0, 0, 0))

for i in range(0, hogelRow):
    for j in range(0, hogelColumn):
        #Sets the current pixel location to represent the current Hogel
        curCapResX = i * resPerHogelX + pixelX
        curCapResY = j * resPerHogelY + pixelY

        # Grabs the pixel information from the inputted image
        pixelData = inImage.getpixel((curCapResX,curCapResY))
        #Flipped hogel column so the output image is not flipped
        outImage.putpixel((i, hogelColumn - j - 1), pixelData)

#saves the output view with the given name 
outImage.save("Output Images/" + outName + ".png")
print("Time taken: %.4f sec" % (time.time() - timeStart))
