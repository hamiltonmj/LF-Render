import maya.cmds as cmds
import math
import time
import configparser


timeStart = time.time()
config = configparser.ConfigParser()
config.read("config.cfg")
x=int(config["Input"]["x"])
y=int(config["Input"]["y"])
start_x=int(config["Input"]["start_x"])
start_y=int(config["Input"]["start_y"])
spacingFactor=int(config["Input"]["spacingFactor"])
d=int(config["Input"]["d"])
rot_x=int(config["Input"]["rot_x"])
rot_y=int(config["Input"]["rot_y"])
rot_z=int(config["Input"]["rot_z"])




def makeSquare(x,y,cam_x,cam_y):
    cmds.curve(n='a',p=[(x,y,0),(x,y+cam_y,0)])
    cmds.curve(n='b',p=[(x,y,0),(x+cam_x,y,0)])
    cmds.curve(n='c',p=[(x+cam_x,y,0),(x+cam_x,y+cam_y,0)])
    cmds.curve(n='d',p=[(x+cam_x,y+cam_y,0),(x,y+cam_y,0)])
    
    cmds.group( 'a','b','c','d', name='box',w=1 )
    
   

def makeView(d,thetaX,thetaY,cam_x,cam_y,start_x,start_y):
       
    k=(math.tan(math.radians(thetaX)/2))*d
    l=(math.tan(math.radians(thetaY)/2))*d
   
    cmds.curve(n='e',p=[(start_x,start_y,0),(start_x-k,start_y-l,-d)])
    cmds.curve(n='f',p=[(start_x,start_y+cam_y,0),(start_x-k,start_y+l+cam_y,-d)])
    cmds.curve(n='g',p=[(start_x+cam_x,start_y,0),(start_x+(k+cam_x),start_y-l,-d)])
    cmds.curve(n='h',p=[(start_x+cam_x,start_y+cam_y,0),(start_x+(k+cam_x),start_y+l+cam_y,-d)])
    
    cmds.curve(n='i',p=[(start_x-k,start_y-l,-d),(start_x-k,start_y+l+cam_y,-d)])
    cmds.curve(n='j',p=[(start_x-k,start_y-l,-d),(start_x+k+cam_x,start_y-l,-d)])
    cmds.curve(n='k',p=[(start_x+k+cam_x,start_y+l+cam_y,-d),(start_x-k,start_y+l+cam_y,-d)])
    cmds.curve(n='l',p=[(start_x+k+cam_x,start_y-l,-d),(start_x+k+cam_x,start_y+l+cam_y,-d)])
    
    box=cmds.group( 'e','f','g','h','i','j','k','l', parent='box' )
    instanceA=cmds.instance('box',name='instanceA')
    cmds.rotate(180,0,0,instanceA)   
    

    
    
def makeGrid(x,y,start_x,start_y,spacingFactor):  
         
        import maya.cmds as cmds
        camList= cmds.ls('Cam*')
        cmds.delete(camList)
            
        result=cmds.camera()
        
        transformName=result[0]
        cameraGroup= cmds.group(empty= True, name='cameraGroup')
            
        for i in range(0,x):
            
            for j in range(0,y): 
             
             instanceResult=cmds.instance(transformName,name='cameraGroup')
             cmds.parent(instanceResult,cameraGroup)     
             cmds.move(i*spacingFactor+start_x,j*spacingFactor+start_y,0) 
             
        cmds.delete(result[0])   
   

def makeCamView(x,y,start_x,start_y,spacingFactor,d,thetaX,thetaY,rot_x,rot_y,rot_z):

   
       makeGrid(x,y,start_x,start_y,spacingFactor)
       cam_x=(x-1)*spacingFactor
       cam_y=(y-1)*spacingFactor
       makeSquare(start_x,start_y,cam_x,cam_y)
       makeView(d,thetaX,thetaY,cam_x,cam_y,start_x,start_y)
       compPack=cmds.group('box','instanceA','cameraGroup',name='compPack')
       cmds.rotate(rot_x,rot_y,rot_z,'compPack')
                    
makeCamView(x,y,start_x,start_y,spacingFactor,d,thetaX,thetaY,rot_x,rot_y,rot_z)
