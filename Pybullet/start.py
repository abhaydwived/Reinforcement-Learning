import pybullet as p
import pybullet_data
import time

physicsclient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally

planeId = p.loadURDF("plane.urdf")
p.setGravity(0,0,-9.8)
#husky
huskypos = [0,0,0]
huskyId = p.loadURDF("husky/husky.urdf", huskypos[0], huskypos[1], huskypos[2])


for i in range(10000):
    p.stepSimulation()
    time.sleep(1./240.)

p.disconnect()
