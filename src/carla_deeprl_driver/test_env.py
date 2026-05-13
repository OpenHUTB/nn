import source.carlaenv as carlaenv
import carla
env = carlaenv.CarlaEnv()
env.client.reload_world(False)
env.client.set_timeout(15)
n = 0
x = 0
while x < 3:
    start_image, _ = env.reset()
    print(f"episode {x + 1}")
    while n < 1000:
        observations, reward, done = env.step(carla.VehicleControl(1, 0, 0))
        n += 1
        print(f"n: {n}")
        print(f"actor num: {len(env.world.get_actors())}")
        print(f"reward is {reward}")
        if done:
            print("done")
            break
    env.client.set_timeout(5)

    x += 1

env._exit()
