import time
import math

import inspect
from matplotlib import image
from pathlib import Path
from abc import ABC, abstractmethod
import matplotlib.lines as mlines
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import rps.utilities.misc as misc


# RobotariumABC: This is an interface for the Robotarium class that
# ensures the simulator and the robots match up properly.  

# THIS FILE SHOULD NEVER BE MODIFIED OR SUBMITTED!

class RobotariumABC(ABC):

    def __init__(self, number_of_robots=-1, show_figure=True, sim_in_real_time=True, initial_conditions=np.array([])):

        # Check user input types
        assert isinstance(number_of_robots,int), (
                "The number of robots used argument (number_of_robots) provided to create the Robotarium object must be an integer type. Recieved type %r." % type(
            number_of_robots).__name__)
        assert isinstance(initial_conditions,np.ndarray),\
            "The initial conditions array argument (initial_conditions) provided to create the Robotarium object must be a numpy ndarray. Recieved type %r." % type(
            initial_conditions).__name__
        assert isinstance(show_figure,bool), (
                "The display figure window argument (show_figure) provided to create the Robotarium object must be boolean type. Recieved type %r." % type(
            show_figure).__name__)
        assert isinstance(sim_in_real_time,bool), (
                "The simulation running at 0.033s per loop (sim_real_time) provided to create the Robotarium object must be boolean type. Recieved type %r." % type(
            show_figure).__name__)

        # Check user input ranges/sizes
        assert (
                    number_of_robots >= 0 and number_of_robots <= 50), "Requested %r robots to be used when creating the Robotarium object. The deployed number of robots must be between 0 and 50." % number_of_robots
        if (initial_conditions.size > 0):
            assert initial_conditions.shape == (3,
                                                number_of_robots), "Initial conditions provided when creating the Robotarium object must of size 3xN, where N is the number of robots used. Expected a 3 x %r array but recieved a %r x %r array." % (
            number_of_robots, initial_conditions.shape[0], initial_conditions.shape[1])

        self.number_of_robots = number_of_robots
        self.show_figure = show_figure
        self.initial_conditions = initial_conditions

        # Boundary stuff -> lower left point / width / height
        self.boundaries = [-1.6, -1, 3.2, 2]

        self.file_path = None
        self.current_file_size = 0

        # Constants
        self.time_step = 0.033
        self.robot_diameter = 0.11
        self.robot_diameter_hunter = 0.18
        self.wheel_radius = 0.016
        self.base_length = 0.105
        self.max_linear_velocity = 0.2
        self.max_linear_velocity_hunter = 0.23
        self.max_angular_velocity = 2 * (self.wheel_radius / self.robot_diameter) * (self.max_linear_velocity / self.wheel_radius)
        self.max_angular_velocity_hunter = 2 * (self.max_linear_velocity_hunter / self.robot_diameter_hunter)
        self.max_wheel_velocity = self.max_linear_velocity / self.wheel_radius # max12.5

        self.robot_radius = self.robot_diameter / 2
        self.robot_length = 0.095       # 机器人矩形的长度
        self.robot_width = 0.09         # 机器人矩形的宽

        self.collision_offset = 0.025  # May want to increase this
        self.collision_diameter = 0.135

        self.velocities = np.zeros((2, number_of_robots))
        self.poses = self.initial_conditions

        if self.initial_conditions.size == 0:
            self.poses = misc.generate_initial_conditions(self.number_of_robots, spacing=0.2, width=2.5, height=1.5)

        self.left_led_commands = []
        self.right_led_commands = []

        # Visualization
        self.figure = []                # 定义一个figure对象
        self.axes = []                  # 定义多个axes对象，方便为axes对象添加不同类型的图形元素
        self.left_led_patches = []      # 左侧led灯
        self.right_led_patches = []     # 右侧led灯
        self.chassis_patches = []       # 车身
        self.right_wheel_patches = []   # 右轮
        self.left_wheel_patches = []    # 左轮
        self.base_patches = []          #
        self.path=[]                    # 轨迹实体

        if (self.show_figure):
            # 这里只会在初始化的时候调用一次，后续的更新在robotarium中
            self.figure, self.axes = plt.subplots()

            self.axes.set_axis_off()
            for i in range(number_of_robots):
                # p = patches.RegularPolygon((self.poses[:2, i]), 4, math.sqrt(2)*self.robot_radius, self.poses[2,i]+math.pi/4, facecolor='#FFD700', edgecolor = 'k')
                if i == 0:
                    # 计算垂直方向上的单位向量
                    position=self.poses[:2, i] +self.robot_length / 2 * np.array(
                        (np.cos(self.poses[2, i] + math.pi / 2), np.sin(self.poses[2, i] + math.pi / 2)))+ 0.04 * np.array(
                        (-np.sin(self.poses[2, i] + math.pi / 2), np.cos(self.poses[2, i] + math.pi / 2)))

                    # 控制生成的矩形位置与质点中心的偏差
                    p = patches.Rectangle((self.poses[:2, i] +
                                           self.robot_length / 2 * np.array((np.cos(self.poses[2, i] + math.pi / 2), np.sin(self.poses[2, i] + math.pi / 2)))
                                            + 0.04 * np.array((-np.sin(self.poses[2, i] + math.pi / 2), np.cos(self.poses[2, i] + math.pi / 2)))),
                                          self.robot_length,
                                          self.robot_width,
                                          (self.poses[2, i] + math.pi / 4) * 180 / math.pi,  # 这里的pi/4，取决于机器人的外观和矩形的相似程度，越相似角度越小越逼真
                                          facecolor='#FFD700', edgecolor='k') # 颜色金黄，边框为黑色

                    pos=mlines.Line2D((position[0],position[0] ), (position[1] , position[1]), linewidth=1, linestyle='--', color='#FFD700')

                else:
                    position=self.poses[:2, i] +self.robot_length / 2 * np.array(
                        (np.cos(self.poses[2, i] + math.pi / 2), np.sin(self.poses[2, i] + math.pi / 2)))+ 0.04 * np.array(
                        (-np.sin(self.poses[2, i] + math.pi / 2), np.cos(self.poses[2, i] + math.pi / 2)))

                    p = patches.Rectangle((self.poses[:2, i] + self.robot_length / 2 * np.array(
                        (np.cos(self.poses[2, i] + math.pi / 2), np.sin(self.poses[2, i] + math.pi / 2))) + \
                                           0.04 * np.array(
                                (-np.sin(self.poses[2, i] + math.pi / 2), np.cos(self.poses[2, i] + math.pi / 2)))),
                                          self.robot_length, self.robot_width,
                                          (self.poses[2, i] + math.pi / 4) * 180 / math.pi, facecolor='#e32636', edgecolor='k') # 颜色红，边框为黑色

                    pos = mlines.Line2D((position[0],position[0] ), (position[1] , position[1]), linewidth = 1, linestyle = '--', color = '#e32636')


                rled = patches.Circle(self.poses[:2, i] + 0.75 * self.robot_length / 2 * np.array(
                    (np.cos(self.poses[2, i]), np.sin(self.poses[2, i])) + 0.04 * np.array(
                        (-np.sin(self.poses[2, i] + math.pi / 2), np.cos(self.poses[2, i] + math.pi / 2)))),
                                      self.robot_length / 2 / 5, fill=False)
                lled = patches.Circle(self.poses[:2, i] + 0.75 * self.robot_length / 2 * np.array(
                    (np.cos(self.poses[2, i]), np.sin(self.poses[2, i])) + \
                    0.015 * np.array(
                        (-np.sin(self.poses[2, i] + math.pi / 2), np.cos(self.poses[2, i] + math.pi / 2)))), \
                                      self.robot_length / 2 / 5, fill=False)

                rw = patches.Circle(self.poses[:2, i] + self.robot_length / 2 * np.array(
                    (np.cos(self.poses[2, i] + math.pi / 2), np.sin(self.poses[2, i] + math.pi / 2))) + \
                                    0.04 * np.array(
                    (-np.sin(self.poses[2, i] + math.pi / 2), np.cos(self.poses[2, i] + math.pi / 2))), \
                                    0.02, facecolor='k')
                lw = patches.Circle(self.poses[:2, i] + self.robot_length / 2 * np.array(
                    (np.cos(self.poses[2, i] - math.pi / 2), np.sin(self.poses[2, i] - math.pi / 2))) + \
                                    0.04 * np.array((-np.sin(self.poses[2, i] + math.pi / 2))), \
                                    0.02, facecolor='k')


                # base = patches.Circle(self.poses[:2, i], self.robot_radius/5, facecolor='r')

                # lw = patches.RegularPolygon(self.poses[:2, i]+self.robot_radius*np.array((np.cos(self.poses[2, i]-math.pi/2), np.sin(self.poses[2, i]-math.pi/2)))+\
                #                                0.035*np.array((-np.sin(self.poses[2, i]+math.pi/2), np.cos(self.poses[2, i]+math.pi/2))),\
                #                                4, math.sqrt(2)*0.02, self.poses[2,i]+math.pi/4, facecolor='k')

                self.chassis_patches.append(p)
                self.left_led_patches.append(lled)
                self.right_led_patches.append(rled)
                self.right_wheel_patches.append(rw)
                self.left_wheel_patches.append(lw)
                self.path.append(pos)
                # self.base_patches.append(base)

                self.axes.add_patch(rw)
                self.axes.add_patch(lw)
                self.axes.add_patch(p)
                self.axes.add_patch(lled)
                self.axes.add_patch(rled)
                self.axes.add_line(pos)
                # self.axes.add_patch(base)

            # Draw arena
            self.boundary_patch = self.axes.add_patch(
                patches.Rectangle(self.boundaries[:2], self.boundaries[2], self.boundaries[3], fill=False))

            self.axes.set_xlim(self.boundaries[0] - 0.1, self.boundaries[0] + self.boundaries[2] + 0.1)
            self.axes.set_ylim(self.boundaries[1] - 0.1, self.boundaries[1] + self.boundaries[3] + 0.1)

            plt.axis("equal")
            plt.ion()
            plt.show()

            # 调整子图之间的间距以及子图相对于图形边界的位置
            plt.subplots_adjust(left=-0.03, right=1.03, bottom=-0.03, top=1.03, wspace=0, hspace=0)

    def set_velocities(self, ids, velocities):

        if velocities.shape[1] == 2:
            velocities[0, 0] = velocities[0, 0] if np.abs(velocities[0, 0]) < self.max_linear_velocity else \
                self.max_linear_velocity * np.sign(velocities[0, 0])
            velocities[1, 0] = velocities[1, 0] if np.abs(velocities[1, 0]) < self.max_angular_velocity else \
                self.max_angular_velocity * np.sign(velocities[1, 0])
            velocities[0, 1] = velocities[0, 1] if np.abs(velocities[0, 1]) < self.max_linear_velocity_hunter else \
                self.max_linear_velocity_hunter * np.sign(velocities[0, 1])
            velocities[1, 1] = velocities[1, 1] if np.abs(velocities[1, 1]) < self.max_angular_velocity_hunter else \
                self.max_angular_velocity_hunter * np.sign(velocities[1, 1])

            # print(f"velocity: {velocities}")
            # print(f"max of hunter: {self.max_linear_velocity_hunter}, {self.max_angular_velocity_hunter}")
            # print(f"max of hunter: {self.max_linear_velocity}, {self.max_angular_velocity}")
        else:
            # Threshold linear velocities
            idxs = np.where(np.abs(velocities[0, :]) > self.max_linear_velocity)
            velocities[0, idxs] = self.max_linear_velocity * np.sign(velocities[0, idxs])

            # Threshold angular velocities
            idxs = np.where(np.abs(velocities[1, :]) > self.max_angular_velocity)
            velocities[1, idxs] = self.max_angular_velocity * np.sign(velocities[1, idxs])

        self.velocities = velocities


    @abstractmethod
    def get_poses(self):
        raise NotImplementedError()

    @abstractmethod
    def step(self):
        raise NotImplementedError()

    # Protected Functions
    def _threshold(self, dxu):
        dxdd = self._uni_to_diff(dxu)

        to_thresh = np.absolute(dxdd) > self.max_wheel_velocity
        dxdd[to_thresh] = self.max_wheel_velocity * np.sign(dxdd[to_thresh])

        dxu = self._diff_to_uni(dxdd)



    def _uni_to_diff(self, dxu):
        r = self.wheel_radius
        l = self.base_length
        dxdd = np.vstack((1 / (2 * r) * (2 * dxu[0, :] - l * dxu[1, :]), 1 / (2 * r) * (2 * dxu[0, :] + l * dxu[1, :])))

        return dxdd

    def _diff_to_uni(self, dxdd):
        r = self.wheel_radius
        l = self.base_length
        dxu = np.vstack((r / (2) * (dxdd[0, :] + dxdd[1, :]), r / l * (dxdd[1, :] - dxdd[0, :])))

        return dxu



    def _validate(self, errors={}):
        # This is meant to be called on every iteration of step.
        # Checks to make sure robots are operating within the bounds of reality.

        p = self.poses
        b = self.boundaries
        N = self.number_of_robots

        for i in range(N):
            x = p[0, i]
            y = p[1, i]

            if (x < b[0] or x > (b[0] + b[2]) or y < b[1] or y > (b[1] + b[3])):
                if "boundary" in errors:
                    errors["boundary"] += 1
                else:
                    errors["boundary"] = 1
                    errors["boundary_string"] = "iteration(s) robots were outside the boundaries."

        for j in range(N - 1):
            for k in range(j + 1, N):
                first_position = p[:2, j] + self.collision_offset * np.array([np.cos(p[2, j]), np.sin(p[2, j])])
                second_position = p[:2, k] + self.collision_offset * np.array([np.cos(p[2, k]), np.sin(p[2, k])])
                if (np.linalg.norm(first_position - second_position) <= (self.collision_diameter)):
                    # if (np.linalg.norm(p[:2,j]-p[:2,k]) <= self.robot_diameter):
                    if "collision" in errors:
                        errors["collision"] += 1
                    else:
                        errors["collision"] = 1
                        errors["collision_string"] = "iteration(s) where robots collided."

        dxdd = self._uni_to_diff(self.velocities)
        exceeding = np.absolute(dxdd) > self.max_wheel_velocity
        if (np.any(exceeding)):
            if "actuator" in errors:
                errors["actuator"] += 1
            else:
                errors["actuator"] = 1
                errors["actuator_string"] = "iteration(s) where the actuator limits were exceeded."

        return errors