import carla
import logging, random
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from enum import Enum
from gym_carla.multi_lane.settings import ROADS, STRAIGHT, CURVE, JUNCTION, DOUBLE_DIRECTION, DISTURB_ROADS
from gym_carla.multi_lane.util.misc import vector
#
class RoadOption(Enum):
    """
    RoadOption represents the possible topological configurations when moving from a segment of lane to other.

    """
    VOID = -1
    LEFT = 1
    RIGHT = 2
    STRAIGHT = 3
    LANEFOLLOW = 4
    CHANGELANELEFT = 5
    CHANGELANERIGHT = 6

def test_waypoint(wp):
    """Attention: the test_waypoint here should be different from test_waypoint function in misc.py
        or it could cause endless loop in GlobalPlanner._build_route()"""
    return wp.road_id in STRAIGHT or wp.road_id in CURVE or wp.road_id in JUNCTION

class GlobalPlanner:
    """
    class for generating chosen circuit's road topology,topology is saved with waypoints list
    vehicle always runs on the outer ring of chosen route

    temporarily used to get more spawnpoints
    """

    def __init__(self, map, sampling_resolution=1000.0) -> None:
        self._sampling_resolution = sampling_resolution
        self._wmap = map

        # code for simulation road generation
        self._route = []
        self._topology = []

        # generate circuit topology
        self._build_topology()
        # print(len(self._topology))
        # self._build_graph()
        # nx.draw(self._graph,with_labels=True,font_weight='bold')
        # plt.draw()
        # plt.show()

        # generate route waypoints list
        self._build_route()

    def get_route(self, ego_waypoint):
        return self._compute_next_waypoints(ego_waypoint, len(self._route))

    def get_spawn_points(self):
        """Vehicle can only be spawned on specific roads, return transforms"""
        spawn_points = []
        for wp in self._route:
            # print('wp.lane_id: ', wp.lane_id)
            if wp.road_id in STRAIGHT or wp.road_id in CURVE:
                # print(wp.lane_id)
                temp = carla.Transform(wp.transform.location, wp.transform.rotation)
                # Increase the z value a little bit to avoid collison upon initializing
                temp.location.z += 0.1
                spawn_points.append(temp)

        return spawn_points

    def split_spawn_points(self, spawn_points):
        pass

    def _build_route(self):
        begin_1 = self._topology[0]
        begin_2 = self._topology[1]
        begin_3 = self._topology[2]
        self._build(begin_1)
        self._build(begin_2)
        self._build(begin_3)

        # remove start
        # print(len(self._route))

    def _build(self,begin):
        self._route.append(begin['entry'])
        for wp in begin['path']:
            self._route.append(wp)
        # self._route.append(begin['exit'])
        indicator = begin['exit']
        iter = None
        for seg in self._topology:
            if seg['entry'].id == indicator.id:
                iter = seg
                break

        while (indicator.id != begin['entry'].id):
            self._route.append(iter['entry'])
            for wp in iter['path']:
                self._route.append(wp)
            # self._route.append(iter['exit'])
            indicator = iter['exit']
            for seg in self._topology:
                if seg['entry'].id == indicator.id:
                    iter = seg
                    break

    def _compute_next_waypoints(self, cur_wp, k=1):
        """
        Add new waypoints to the trajectory queue.

        :param cur_wp: current waypoint
        :param k: how many waypoints to compute
        :return: waypoint list
        """
        next_wps = []
        iter = None
        for i, wp in enumerate(self._route):
            if wp.id == cur_wp.id:
                iter = i
                break
            elif wp.transform.location.distance(cur_wp.transform.location) < self._sampling_resolution / 2:
                # can't find the exact waypoint, get an approximation
                iter = i
        if iter is None:
            logging.error("Current waypoint on route not found!")
        if iter + k < len(self._route):
            for i in range(k):
                next_wps.append(self._route[iter + i + 1])
        else:
            for i in range(len(self._route) - iter - 1):
                next_wps.append(self._route[iter + i + 1])
            for i in range(k - (len(self._route) - iter - 1)):
                next_wps.append(self._route[i])

        return next_wps

    def _build_topology(self):
        """
        This function retrieves topology from the server as a list of
        road segments as pairs of waypoint objects, and processes the
        topology into a list of dictionary objects with the following attributes

        - entry (carla.Waypoint): waypoint of entry point of road segment
        - entryxyz (tuple): (x,y,z) of entry point of road segment
        - exit (carla.Waypoint): waypoint of exit point of road segment
        - exitxyz (tuple): (x,y,z) of exit point of road segment
        - path (list of carla.Waypoint):  list of waypoints between entry to exit, separated by the resolution
        """
        # Retrieving waypoints to construct a detailed topology
        for segment in self._wmap.get_topology():
            wp1, wp2 = segment[0], segment[1]
            if test_waypoint(wp1) and test_waypoint(wp2):
                l1, l2 = wp1.transform.location, wp2.transform.location
                # Rounding off to avoid floating point imprecision
                x1, y1, z1, x2, y2, z2 = np.round([l1.x, l1.y, l1.z, l2.x, l2.y, l2.z], 0)
                wp1.transform.location, wp2.transform.location = l1, l2
                seg_dict = dict()
                seg_dict['entry'], seg_dict['exit'] = wp1, wp2
                seg_dict['entryxyz'], seg_dict['exitxyz'] = (x1, y1, z1), (x2, y2, z2)
                seg_dict['path'] = []
                endloc = wp2.transform.location
                if wp1.transform.location.distance(endloc) > self._sampling_resolution:
                    w = wp1.next(self._sampling_resolution)[0]
                    while w.transform.location.distance(endloc) > self._sampling_resolution:
                        if test_waypoint(w):
                            seg_dict['path'].append(w)
                        w = w.next(self._sampling_resolution)[0]
                    if test_waypoint(w):
                        seg_dict['path'].append(w)
                else:
                    next_wp = wp1.next(self._sampling_resolution)[0]
                    if test_waypoint(next_wp):
                        seg_dict['path'].append(next_wp)
                self._topology.append(seg_dict)

    def _build_graph(self):
        """
        This function builds a networkx graph representation of topology, creating several class attributes:
        - graph (networkx.DiGraph): networkx graph representing the world map, with:
            Node properties:
                vertex: (x,y,z) position in world map
            Edge properties:
                entry_vector: unit vector along tangent at entry point
                exit_vector: unit vector along tangent at exit point
                net_vector: unit vector of the chord from entry to exit
                intersection: boolean indicating if the edge belongs to an  intersection
        - id_map (dictionary): mapping from (x,y,z) to node id
        - road_id_to_edge (dictionary): map from road id to edge in the graph
        """

        self._graph = nx.DiGraph()
        self._id_map = dict()  # Map with structure {(x,y,z): id, ... }
        self._road_id_to_edge = dict()  # Map with structure {road_id: {lane_id: edge, ... }, ... }

        for segment in self._topology:
            entry_xyz, exit_xyz = segment['entryxyz'], segment['exitxyz']
            path = segment['path']
            entry_wp, exit_wp = segment['entry'], segment['exit']
            intersection = entry_wp.is_junction
            road_id, section_id, lane_id = entry_wp.road_id, entry_wp.section_id, entry_wp.lane_id

            for vertex in entry_xyz, exit_xyz:
                # Adding unique nodes and populating id_map
                if vertex not in self._id_map:
                    new_id = len(self._id_map)
                    self._id_map[vertex] = new_id
                    self._graph.add_node(new_id, vertex=vertex)
            n1 = self._id_map[entry_xyz]
            n2 = self._id_map[exit_xyz]
            if road_id not in self._road_id_to_edge:
                self._road_id_to_edge[road_id] = dict()
            if section_id not in self._road_id_to_edge[road_id]:
                self._road_id_to_edge[road_id][section_id] = dict()
            self._road_id_to_edge[road_id][section_id][lane_id] = (n1, n2)

            entry_carla_vector = entry_wp.transform.rotation.get_forward_vector()
            exit_carla_vector = exit_wp.transform.rotation.get_forward_vector()

            # Adding edge with attributes
            self._graph.add_edge(
                n1, n2,
                length=len(path) + 1, path=path,
                entry_waypoint=entry_wp, exit_waypoint=exit_wp,
                entry_vector=np.array(
                    [entry_carla_vector.x, entry_carla_vector.y, entry_carla_vector.z]),
                exit_vector=np.array(
                    [exit_carla_vector.x, exit_carla_vector.y, exit_carla_vector.z]),
                net_vector=vector(entry_wp.transform.location, exit_wp.transform.location),
                intersection=intersection, type=RoadOption.LANEFOLLOW)
