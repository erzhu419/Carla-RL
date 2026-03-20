import carla

client = carla.Client('172.30.112.1', 2000)
client.set_timeout(20.0)
world = client.get_world()
map_name = world.get_map().name
print(map_name)
client.load_world('Town10HD')
map_name = world.get_map().name
print(map_name)