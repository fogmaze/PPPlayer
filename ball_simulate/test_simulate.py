import pybullet as p
import math
import random
import time


# 设置GUI模式，如果不需要可注释掉此行
p.connect(p.GUI)

# 创建平面
planeId = p.createCollisionShape(p.GEOM_PLANE)
plane = p.createMultiBody(0, planeId)

# 创建球
radius = 0.04
sphereId = p.createCollisionShape(p.GEOM_SPHERE, radius=radius)
startPos = [0, 0, 1]
startOrientation = p.getQuaternionFromEuler([0,0,0])
sphere = p.createMultiBody(27, sphereId, basePosition=startPos, baseOrientation=startOrientation)

# 随机设置球的速度
linearVelocity = [random.uniform(-2, 2), random.uniform(-2, 2), random.uniform(0, 5)]
angularVelocity = [0, 0, 0]
p.resetBaseVelocity(sphere, linearVelocity, angularVelocity)

# 添加空气阻力
linearDamping = 6 * math.pi * radius * 0.0185
angularDamping = 0.1
p.changeDynamics(sphere, -1, linearDamping=linearDamping, angularDamping=angularDamping)

# 添加彈性
restitution = 0.9 # 彈性係數
p.changeDynamics(sphere, -1, restitution=restitution)
p.changeDynamics(plane, -1, restitution=restitution)


# 模拟自由落体过程
p.setGravity(0, 0, -10)
while True:
    p.stepSimulation()
    time.sleep(1./240.) # 设置模拟速度，可以根据实际需求调整
    spherePos, sphereOrn = p.getBasePositionAndOrientation(sphere)
    if spherePos[2] < radius: # 如果球接触平面
        linearVelocity, angularVelocity = p.getBaseVelocity(sphere)

# 关闭模拟器
p.disconnect()
