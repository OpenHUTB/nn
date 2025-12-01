import carla
import random
import time
import sys




print('Inside pedestrians........')

client = carla.Client('localhost', 2000) # connect to the CARLA server
client.set_timeout(10.0) # set a timeout for connecting to the server

world = client.get_world() # get the current world

spawn_point = carla.Transform(carla.Location(x=112.2033, y=310.975, z=5), carla.Rotation(yaw=180))


'''
to spawn the locations for pedestrians to walk across the road
'''
pedestrians_crossing = []


p1 = carla.Transform(carla.Location(x=100, y=310.975, z=5), carla.Rotation(yaw=180))
p2 = carla.Transform(carla.Location(x=120, y=310.975, z=5), carla.Rotation(yaw=180))
p3 = carla.Transform(carla.Location(x=142, y=310.975, z=5), carla.Rotation(yaw=180))
#p4 = carla.Transform(carla.Location(x=162, y=310.975, z=5), carla.Rotation(yaw=180))

pedestrians_crossing.append(p1)
pedestrians_crossing.append(p2)
pedestrians_crossing.append(p3)
#pedestrians_crossing.append(p4)



'''
locations for pedestrians to walk in the opposite direction of car on the sidewalk
'''
pedestrians_walking_oppo = []

p1 = carla.Transform(carla.Location(x=105, y=311.975, z=5), carla.Rotation(yaw=180))
p2 = carla.Transform(carla.Location(x=125, y=311.975, z=5), carla.Rotation(yaw=180))
p3 = carla.Transform(carla.Location(x=145, y=311.975, z=5), carla.Rotation(yaw=180))
#p4 = carla.Transform(carla.Location(x=165, y=311.975, z=5), carla.Rotation(yaw=180))

pedestrians_walking_oppo.append(p1)
pedestrians_walking_oppo.append(p2)
pedestrians_walking_oppo.append(p3)
#pedestrians_walking_oppo.append(p4)



'''
locations for pedestrains to walk in the direction of car on the sidewalk
'''
pedestrians_walking_away = []

p1 = carla.Transform(carla.Location(x=95, y=310, z=5), carla.Rotation(yaw=180))
p2 = carla.Transform(carla.Location(x=115, y=310, z=5), carla.Rotation(yaw=180))
p3 = carla.Transform(carla.Location(x=135, y=310, z=5), carla.Rotation(yaw=180))
#p4 = carla.Transform(carla.Location(x=155, y=310, z=5), carla.Rotation(yaw=180))

pedestrians_walking_away.append(p1)
pedestrians_walking_away.append(p2)
pedestrians_walking_away.append(p3)
#pedestrians_walking_away.append(p4)




print("Have taken the spawn point....")

#time.sleep(5)


for cross, opposite, away in zip(pedestrians_crossing, pedestrians_walking_oppo, pedestrians_walking_away):

    #spawn_point = random.choice(spawn_points) # select a random spawn point

    # select a random pedestrian blueprint
    pedestrian_bp = random.choice(world.get_blueprint_library().filter("walker.pedestrian.*"))


    # spawn the pedestrian at the selected spawn point 
    pedestrian_cross = world.try_spawn_actor(pedestrian_bp, cross)
    pedestrian_opposite = world.try_spawn_actor(pedestrian_bp, opposite)
    pedestrian_away = world.try_spawn_actor(pedestrian_bp, away)

    # actor_list.append(pedestrian_cross)
    # actor_list.append(pedestrian_opposite)
    # actor_list.append(pedestrians_walking_away)


    '''
    TO make the pedestrians move in a specified direction depending on their type
    '''
    # to cross
    if pedestrian_cross is not None:
        walker_control = carla.WalkerControl()
        walker_control.speed = 0.5
        #walker_control.direction = carla.Vector3D(x=random.uniform(-1, 1), y=random.uniform(-1, 1), z=0)
        walker_control.direction = carla.Vector3D(x=0, y=-1, z=0)
        pedestrian_cross.apply_control(walker_control)
        print("Spawned crossing pedestrian.....")
    

    # in opposite direction
    if pedestrian_opposite is not None:
        walker_control = carla.WalkerControl()
        walker_control.speed = 0.5
        #walker_control.direction = carla.Vector3D(x=random.uniform(-1, 1), y=random.uniform(-1, 1), z=0)
        walker_control.direction = carla.Vector3D(x=-1, y=0, z=0)
        pedestrian_opposite.apply_control(walker_control)
        print("Spawned pedestrian moving in opposite direction......")

    
    # in car's direction (away from the car)
    if pedestrian_away is not None:
        walker_control = carla.WalkerControl()
        walker_control.speed = 0.5
        #walker_control.direction = carla.Vector3D(x=random.uniform(-1, 1), y=random.uniform(-1, 1), z=0)
        walker_control.direction = carla.Vector3D(x=1, y=0, z=0)
        pedestrian_away.apply_control(walker_control)
        print("Spawned pedestrian moving away......")        



    time.sleep(8)



print("End of episode.....")
print("Destroying all the pedestrians.....")

# time.sleep(5)


actor_list = world.get_actors()
# for actor in actor_list_new:
#     if 'walker.pedestrian' in actor.type_id:
#         actor.destroy()
#         #print("PEDESTRIAN DESTROYED!!!!!")






