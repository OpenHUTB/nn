import airsim, time
wps = [(0,0,-5),(5,0,-5),(5,5,-5),(0,5,-5),(0,0,-5)]
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)
client.takeoffAsync().join()
for x,y,z in wps:
    client.moveToPositionAsync(float(x),float(y),float(z),5).join()
    time.sleep(0.5)
client.moveToPositionAsync(0,0,-2,3).join()
client.landAsync().join()
client.armDisarm(False)
client.enableApiControl(False)