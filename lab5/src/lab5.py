#!/usr/bin/env python3

import rospy
import numpy as np
import math
import matplotlib.pyplot as mpl
from geometry_msgs.msg import Twist, Pose
from copy import deepcopy
import heapq as hq



ROBOT_SIZE = 0.1
G_MULTIPLIER = 0.2
MOVES = [(0.2, math.radians(0)), 	# move ahead
         (-0.2, math.radians(0)), 	# move backwards
         (0, math.radians(90)), 	# turn left
         (0, -math.radians(90))]  # turn right
TOLERANCE = 0.2

SIMILARITY_THRESHOLD = 0.1
SAFETY_OFFSET = 1

MAP_WIDTH = 42.0
MAP_HEIGHT = 42.0
LASER_MAX = 2.0


class Node:

    def __init__(self, x, y, theta=0.0, parent=None):
        self.x = x
        self.y = y
        self.theta = theta
        self.parent = parent
        # f(n) = h(n) + g(n)
        self.f = 0
        self.h = 0

    def euclidean_distance(self, goal):
        """
        Method to compute distance from current position to the goal
        @arg	goal 	Node object with x, y, theta
        @returns 	euclidean distance from current point to goal
        """
        return math.sqrt(math.pow((goal.x-self.x), 2) + math.pow((goal.y-self.y), 2))

    def apply_move(self, move):
        """
        Apply the given move to current position
        @arg 	move 	[length, dtheta]
        """
        theta_new = self.theta + move[1]
        x_new = self.x + math.cos(theta_new) * move[0]  # d.cos(theta)
        y_new = self.y + math.sin(theta_new) * move[0]  # d.sin(theta)
        return Node(x_new, y_new, theta_new)

    def is_move_valid(self, grid_map, move):
        """
        Return true if required move is legal
        """
        goal = self.apply_move(move)

        # convert goal coordinates to pixel coordinates before checking this
        goal_pixel = world_to_pixel((goal.x, goal.y), (42, 42))
        # check if too close to the walls
        if goal_pixel[0] >= SAFETY_OFFSET and not grid_map[int(goal_pixel[0]-SAFETY_OFFSET)][int(goal_pixel[1])]:
            return False
        if goal_pixel[1] >= SAFETY_OFFSET and not grid_map[int(goal_pixel[0])][int(goal_pixel[1]-SAFETY_OFFSET)]:
            return False
        if goal_pixel[0] >= SAFETY_OFFSET and goal_pixel[1] >= SAFETY_OFFSET and not grid_map[int(goal_pixel[0]-SAFETY_OFFSET)][int(goal_pixel[1]-SAFETY_OFFSET)]:
            return False
        if grid_map[goal_pixel[0]][goal_pixel[1]]:
            return True
        return False

    def is_similar(self, other):
        """
        Return true if other node is in similar position as current node
        """
        return self.euclidean_distance(other) <= SIMILARITY_THRESHOLD


def world_to_pixel(world_points, image_size):
    world_x, world_y = world_points
    img_h, img_w = image_size
    pixel_points = []
    pixel_points.append(int(max((world_x / MAP_WIDTH) * img_w, 0)))
    if pixel_points[0] > img_w - 1:
        pixel_points[0] = img_w - 1
    pixel_points.append(int(max((world_y / MAP_HEIGHT) * img_h, 0)))
    if pixel_points[1] > img_h - 1:
        pixel_points[1] = img_h
    pixel_points[1] = pixel_points[1]
    pixel_points[0] = img_w/2 + pixel_points[0]
    pixel_points[1] = img_h/2 - pixel_points[1]
    return pixel_points


def astar(grid_map, start, end):
    # start, end are in world coordinates and Node objects
    opened = []
    closed = []
    final = None
    hq.heappush(opened, (0.0, start))

    while (final == None) and opened:
        # q is a Node object with x, y, theta
        q = Node(0.0,0.0)
        tempVar= hq.heappop(opened)[1]
        for move in MOVES:		# move is in world coordinates
            q.x = tempVar[0]
            q.y = tempVar[1]
            if (q.is_move_valid(grid_map, move)):
                # Node is returned in world coordinates
                next_node = q.apply_move(move)
            else:
                next_node = None
            #print("next node is : ", next_node)
            if next_node != None:
                if next_node.euclidean_distance(end) < TOLERANCE:
                    next_node.parent = q
                    final = next_node
                    break
                # update heuristics h(n) and g(n)
                next_node.h = next_node.euclidean_distance(end)
                next_node.g = q.g + next_node.euclidean_distance(q)
                # f(n) = h(n) + g(n)
                next_node.f = G_MULTIPLIER * next_node.g + next_node.h
                next_node.parent = q

                # other candidate locations to put in the heap
                potential_open = any(other_f <= next_node.f and other_next.is_similar(
                    next_node) for other_f, other_next in opened)

                if not potential_open:
                    potential_closed = any(other_next.is_similar(
                        next_node) and other_next.f <= next_node.f for other_next in closed)
                    if not potential_closed:
                        hq.heappush(opened, (next_node.f, next_node))
        closed.append(q)
    return final


if __name__ == "__main__":
    while not rospy.is_shutdown():
        rospy.init_node('lab5',anonymous=True)
        global mapMatrix
        mapMatrix = [
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        [1,0,1,1,1,1,1,1,1,1,0,0,0,1,1,1,1,1,1,1,1,0,0,1,1,1,1,1,1,1,1,0,0,1,1,1,1,1,1,1,1,1],
        [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
        [1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1],
        [1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1],
        [1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1],
        [1,1,1,0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1],
        [1,1,1,0,0,0,0,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,1,1],
        [1,1,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,1,1],
        [1,1,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1],
        [1,1,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1],
        [1,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
        [1,1,0,0,0,0,0,1,1,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1],
        [1,1,0,0,0,0,0,1,1,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1],
        [1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1],
        [1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1],
        [1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1],
        [1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1],
        [1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1],
        [1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1],
        [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
        [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
        [1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1],
        [1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,1,1],
        [1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,1,1],
        [1,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,1,1],
        [1,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,1,1],
        [1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1],
        [1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1],
        [1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1],
        [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,1],
        [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,1],
        [1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1],
        [1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1],
        [1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1],
        [1,1,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1],
        [1,1,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1],
        [1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1],
        [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1],
        [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1],
        [1,1,1,1,1,1,1,1,1,0,0,1,1,1,1,1,1,1,1,0,0,1,1,1,1,1,1,1,1,0,0,1,1,1,1,1,1,1,1,0,0,1],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]]
        goalx = rospy.get_param('~goalx')
        goaly = rospy.get_param('~goaly')
        goalState = (goalx, goaly)
        startState = (4.0,6.0)
        vizMap = deepcopy(mapMatrix)
        figure = mpl.figure(figsize=(12, 12))
        vizMap[int(startState[0])][int(startState[1])] = 5
        vizMap[int(goalState[0])][int(goalState[1])] = 6
        mpl.imshow(vizMap, origin='upper', interpolation='none')
        mpl.pause(5)
        shortest_path_wc = []
        count = 0
        shortest_path = astar(mapMatrix,goalState,startState)
        for node in shortest_path:
            if(count == 0):
                vizMap[node[0]][node[1]] = 3
            elif count == len(shortest_path) -1:
                vizMap[node[0]][node[1]] = 7
            else:
                vizMap[node[0]][node[1]] = 10
            mpl.imshow(vizMap, origin='upper', interpolation='none')#, clim=COLOR_MAP)
            mpl.pause(2)
            count= count+1


