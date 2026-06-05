
SEMANTIC_CATEGORIES = {
    0: "Unlabeled",
    1: "Traffic Sign/Lights",
    2: "Roads",
    3: "Road Lines",
    4: "Sidewalk",
    5: "Ground",
    6: "Vehicles",
    7: "Pedestrians",
}

SEMANTIC_COLORS = {
    0: (0, 0, 0), # Unlabeled
    1: (220, 220, 0), # Traffic Sign/Lights
    2: (0, 255, 0),#(128, 64, 128), # Roads
    3: (157, 234, 50), # Road Lines
    4: (244, 35, 232), # Sidewalk
    5: (107, 142, 35), # Ground
    6: (0, 0, 255), # Vehicles
    7: (220, 20, 60), # Pedestrians
}


# By default, CARLA has a set number of categories, but depending on what
# version, a different number of labels. The default documentation
# https://carla.readthedocs.io/en/stable/cameras_and_sensors/#camera-semantic-segmentation
# lists 13 labels; however the latest build actually has an additional
# 10 for 23 labels.
# https://carla.readthedocs.io/en/latest/ref_sensors/#semantic-segmentation-camera
# Luckily, the overlapping label ID's are equivalent.
# We don't wish to cover all the categories of these labels, and want to
# consolidate it into our own. Thus SEMANTIC_CATEGORIES are our actual
# labels, and below is the CARLA=> our labels mapping.
# Thus irregardless if it's a dataset with 0-22 or 0-13, we can map it
# back to this
CARLA_SEMANTIC_CATEGORY_MAPPINGS = {
    0: 0, # Unlabeled
    1: 0, # Building
    2: 0, # Fence
    3: 0, # Other
    4: 7, # Pedestrian
    5: 0, # Pole
    6: 3, # RoadLine
    7: 2, # Road
    8: 4, # SideWalk
    9: 0, # Vegetation
    10: 6, # Vehicles
    11: 0, # Wall
    12: 1, # TrafficSign
    13: 0, # Sky
    14: 5, # Ground
    15: 0, # Bridge
    16: 0, # RailTrack
    17: 0, # GuardRail
    18: 1, # TrafficLight
    19: 0, # Static
    20: 0, # Dynamic
    21: 5, # Water
    22: 5, # Terrain
}

CARLA_SEMANTIC_COLORS = {
    0: (0, 0, 0),
    1: (70, 70, 70),
    2: (100, 40, 40),
    3: (55, 90, 80),
    4: (220, 20, 60),
    5: (153, 153, 153),
    6: (157, 234, 50),
    7: (128, 64, 128),
    8: (244, 35, 232),
    9: (107, 142, 35),
    10: (0, 0, 142),
    11: (102, 102, 156),
    12: (220, 220, 0),
    13: (70, 130, 180),
    14: (81, 0, 81),
    15: (150, 100, 100),
    16: (230, 150, 140),
    17: (180, 165, 180),
    18: (250, 170, 30),
    19: (110, 190, 160),
    20: (170, 120, 50),
    21: (45, 60, 150),
    22: (145, 170, 100),
}