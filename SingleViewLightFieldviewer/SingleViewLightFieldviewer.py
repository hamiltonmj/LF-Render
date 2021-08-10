from PIL import Image
import time
import configparser

timeStart = time.time()


config = configparser.ConfigParser()

config.read("config.cfg")

print(config.sections())


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
#outResX = int(config["Output"]["renderResolutionX"])

#outResY = int(config["Output"]["renderResolutionY"])

#int(config["Output"]["renderResolution_x"])

inImage = Image.open(config["Input"]["fileName"])
#inImage.show()

if(capResX <= 0):
    capResX = int(inImage.width)

if(capResY <= 0):
    capResY = int(inImage.height)

resPerHogelX = int(capResX/ hogelRow)
resPerHogelY = int(capResY/ hogelColumn)

#if(outResX <= 0):
    #outResX = int(resPerHogelX)

#if(outResY <= 0):
    #outResY = int(resPerHogelY)


outImage = Image.new("RGBA",(hogelRow, hogelColumn) , (0, 0, 0, 0))

for i in range(0, hogelRow):
    for j in range(0, hogelColumn):

        curCapResX = i * resPerHogelX + pixelX
        curCapResY = j * resPerHogelY + pixelY

        pixelData = inImage.getpixel((curCapResX,curCapResY))
        #Flipped hogel column so the output image is not flipped
        outImage.putpixel((i, hogelColumn - j - 1), pixelData)

outImage.save("testOutputImage.png")
outImage.show()
print("Time taken: %.4f sec" % (time.time() - timeStart))
