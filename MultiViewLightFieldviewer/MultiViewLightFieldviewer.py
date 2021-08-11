from PIL import Image
import time
import configparser

timeStart = time.time()

config = configparser.ConfigParser()
config.read("config.cfg")


# The inputted Pixel to be represented 
pixelX = int(config["Input"]["pixelX"])
pixelY = int(config["Input"]["pixelY"])

# The inputted Direction to move outputImage
viewDir = config["Input"]["viewDirection"].upper()

# The number of outputImages to generate
numOfViews = int(config["Input"]["numOfViews"])

# The inputted Capture Resolution of the light field 
capResX = int(config["LightField Parameters"]["lightFieldX"])
capResY = int(config["LightField Parameters"]["lightFieldY"])

# The inputted number of hogels per row/column
hogelRow = int(config["LightField Parameters"]["rowHogel"])
hogelColumn = int(config["LightField Parameters"]["columnHogel"])

#The resolution of the outputted image
outResX = int(config["Output"]["renderResolutionX"])
outResY = int(config["Output"]["renderResolutionY"])

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

prevTime = time.time()

for i in range(0, numOfViews):
    for u in range(0, hogelRow):
        for v in range(0, hogelColumn):
            #Sets the current pixel location to represent the current Hogel
            curCapPosX = u * resPerHogelX + pixelX
            curCapPosY = v * resPerHogelY + pixelY

            # Grabs the pixel information from the inputted image
            pixelData = inImage.getpixel((curCapPosX,curCapPosY))
            #Flipped hogel column so the output image is not flipped
            outImage.putpixel((u, hogelColumn - v - 1), pixelData)

    #saves the output view with the
    outImage.save("Output Images/view" + str(i) + ".png")

    #Adds pixel movement in direction specified 
    #unless the new pixel would be out of bounds of the current hogel
    if(viewDir == "U") and ((pixelY + 1) < resPerHogelX):
        pixelY = pixelY + 1

    elif(viewDir == "D") and ((pixelY - 1) >= 0):
        pixelY = pixelY - 1

    elif(viewDir == "L") and ((pixelX - 1) >= 0):
        pixelX = pixelX - 1

    elif((pixelX + 1) < resPerHogelX):
        pixelX = pixelX + 1

    #tracks the time taken to accomplish storing the current view
    print("view: " + str(i) + " Time taken: %.4f sec" % (time.time() - prevTime))
    prevTime = time.time()

print("Time taken: %.4f sec" % (time.time() - timeStart))
