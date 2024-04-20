import numpy as np
from matplotlib import patches
from rps.robotarium import Robotarium
from rps.utilities import barrier_certificates
from rvo import rvo_vel,compute_V_des

class Simulator(Robotarium):
    """ HITSZ ML Lab simulator """

    def __init__(self, number_of_robots=1, *args, **kwd):
        super(Simulator, self).__init__(number_of_robots=number_of_robots, *args, **kwd)
        self.init_environment()
        self.terminate = 0

    def render(self, show=0):
        if show:
            self.show_figure = True

    def close(self):
        self.terminate = 1

    def init_environment(self):
        # 初始化矩形边界
        self.boundaries = [-3.1, -3.1, 6.2, 6.2]  # 坐标区域左下[-3.1,-3.1]  右上[3.1，3.1]
        if self.show_figure:
            self.boundary_patch.remove()  # 删除之前的边界框
        padding = 1

        if self.show_figure:
            # NOTE: boundaries = [x_min, y_min, x_max - x_min, y_max - y_min] ?
            self.axes.set_xlim(self.boundaries[0] - padding, self.boundaries[0] + self.boundaries[2] + padding)  # 设置x边界
            self.axes.set_ylim(self.boundaries[1] - padding, self.boundaries[1] + self.boundaries[3] + padding)  # 设置y边界

            patch = patches.Rectangle(self.boundaries[:2], *self.boundaries[2:4], fill=False, linewidth=2)  # 创建一个矩形边界框
            self.boundary_patch = self.axes.add_patch(patch)

        # 设置障碍物
        self.barrier_centers = [(1, 1), (-1, 1), (0, -1),(2,0),(-2,0)]
        self.radius = 0.1
        if self.show_figure:
            self.barrier_patches = [
                patches.Circle(self.barrier_centers[0], radius=self.radius),
                patches.Circle(self.barrier_centers[1], radius=self.radius),
                patches.Circle(self.barrier_centers[2], radius=self.radius),
                patches.Circle(self.barrier_centers[3], radius = self.radius),
                patches.Circle(self.barrier_centers[4], radius = self.radius),
                # patches.Rectangle((-0.25, 2.45), 0.55, 0.55),
                # patches.Rectangle((-0.25, -2.55), 0.55, 0.55)
            ]

            for patch in self.barrier_patches:
                patch.set(fill=True, color="#000") # 对于每个障碍物对象，调用 set 方法来设置障碍物的填充属性和颜色
                self.axes.add_patch(patch)         # 将设置了填充和颜色的障碍物对象添加到图形界面中

            # TODO: barries certs
            self.barrier_certs = []

            # set goals areas
            self.goal_patches = [
                # patches.Circle((4, 4), radius=0.24),
                # patches.Circle((-4, 4), radius=0.24),
                # patches.Circle((4, -4), radius=0.24),
                patches.Circle((-2.5, -2.5), radius=0.2),
            ]

            for patch in self.goal_patches:
                patch.set(fill=False, color='#5af')
                self.axes.add_patch(patch)

    def set_velocities(self, velocities):
        """
        velocites is a (N, 2) np.array contains (ω, v) of agents
        """
        self._velocities = velocities

    def step(self, action):
        # 计算追捕者的动作
        """
        get robot pose 3x2
        first column is prey
        second column is hunter
        """
        poses = self.get_poses()  # 返回np.array([x,y,theta]) x N,这里N=2，第一列是猎物，第二列是追捕者
        # print('poses before', poses)

        # get hunter's velocity
        hunter_states=np.concatenate((poses[:, 1].reshape(-1, 1), poses[:, 2].reshape(-1, 1),poses[:, 3].reshape(-1, 1)),axis=1)

        dxu_hunter = self.hunter_policy(hunter_states, poses[:2, 0].reshape(-1, 1))
        # 将猎人和猎物的动作合并成一个数组，用于设置机器人的速度
        dxu = np.concatenate([action.reshape(-1, 1), dxu_hunter], axis=1)
        terminate = 0
        reward = 0

        # make a step
        self.set_velocities(dxu)
        self._step()
        # print('poses after', poses)
        # 碰撞检测
        for robot in range(4):
            # collision with boundaries
            padding = 0.15 # 设置了一个填充值，用于在检测到碰撞时将机器人从边界处移开一小段距离，以防止它们粘在边界上
            self.poses[0, robot] = self.poses[0, robot] if self.poses[0, robot] > self.boundaries[0] + padding else \
                self.boundaries[0] + padding
            self.poses[0, robot] = self.poses[0, robot] if self.poses[0, robot] < self.boundaries[0] + self.boundaries[
                2] - padding else self.boundaries[0] + self.boundaries[2] - padding
            self.poses[1, robot] = self.poses[1, robot] if self.poses[1, robot] > self.boundaries[1] + padding else \
                self.boundaries[1] + padding
            self.poses[1, robot] = self.poses[1, robot] if self.poses[1, robot] < self.boundaries[0] + self.boundaries[
                3] - padding else self.boundaries[1] + self.boundaries[3] - padding

            # collision with barriers
            for barrier in self.barrier_centers:
                tempA = self.poses[:2, robot] - np.array(barrier) # 计算机器人当前位置与障碍物中心的向量
                dist = np.linalg.norm(tempA)
                # 如果机器人与障碍物的距离小于机器人的半径加上一个填充值，则说明机器人与障碍物发生了碰撞
                if dist < self.radius + padding:
                    tempA = tempA / dist * (self.radius + padding)
                    self.poses[:2, robot] = tempA + np.array(barrier)
                    if robot == 0:
                        reward -= 50*dist

        # collision with prey,逃跑者与追捕者之间的避碰
        # 如果将要发生冲突，那这一时刻的动作由rvo模块来生成
        # for robot in range(4):
        #     tempB=self.poses[:2,]
        # for robot in range(3):
        #     tempB = self.poses[:2, robot+1] - self.poses[:2, 0]
        #     dist_temp = np.linalg.norm(tempB)
        #     if dist_temp < self.radius:
        #         tempB = tempB / dist_temp * (self.radius)
        #         self.poses[:2, 1] = tempB + np.array(self.poses[:2, 0])
        #         # self.terminate = 1
        #         reward -= 10
        for i in range(3):
            tempB = poses[:2, i+1] - poses[:2, 0]
            dist_temp = np.linalg.norm(tempB)
            if dist_temp < 2*self.radius+0.2:
                tempB = tempB / dist_temp * (2*self.radius+0.2)
                poses[:2, i+1] = tempB + np.array(poses[:2, 0])
                reward -= dist_temp*50

        for i in range(3):
            for j in range(3):
                if j==i:
                    continue
                tempB = poses[:2, i+1] - poses[:2, j+1]
                dist_temp = np.linalg.norm(tempB)
                if dist_temp < 2*self.radius+0.1:
                    tempB = tempB / dist_temp * (2*self.radius+0.1)
                    poses[:2, i+1] = tempB + np.array(poses[:2, j+1])
                    reward -= dist_temp*100

        # whether reach goal area
        tempC = self.poses[:2, 0] - np.array([-2.5, -2.5])
        dist_C = np.linalg.norm(tempC)
        # print(dist_C)
        if dist_C < 0.2:
            self.terminate = 1
            reward += 5000
        elif dist_C < 1:
            reward += 2000
        elif dist_C < 2:
            reward += 1000
        elif dist_C < 3:
            reward += 500
        elif dist_C < 4:
            reward += 200
        elif dist_C < 5:
            reward += 50
        else:
            pass

        # compute the reward
        reward = reward + self.get_reward(poses[:, 0], action) + 10.0 / dist_C
        hunter_states = np.concatenate(
            (self.poses[:, 1].reshape(-1, 1), self.poses[:, 2].reshape(-1, 1), self.poses[:, 3].reshape(-1, 1)), axis = 1)
        eva=self.poses[:, 0][:, np.newaxis]
        state = np.hstack((eva,hunter_states))
        state = np.ravel(state)
        # state = np.append(state, dist_C)
        info = None
        return state, reward, self.terminate, info

    def _step(self, *args, **kwd):
        dxu = self._velocities
        # print('_step_dxu', dxu)
        """
        the first column is the velocity of prey
        the second column is the velocity of hunter
        """
        if self.show_figure:
            for cert in self.barrier_certs:
                dxu = cert(dxu, self.poses)

        super(Simulator, self).set_velocities(range(self.number_of_robots), dxu)
        super(Simulator, self).step(*args, **kwd)
        # print("self.poses", self.poses)

    def evaluate(self, action):
        poses = self.get_poses()
        # print('poses before', poses)

        # get hunter's velocity
        dxu_hunter = self.hunter_policy(poses[:, 1].reshape(-1, 1), poses[:2, 0].reshape(-1, 1))
        velocities = np.concatenate([action.reshape(-1, 1), dxu_hunter], axis=1)

        # 限制猎人与猎物的线速度与角速度不超过最大角速度
        velocities[0, 0] = velocities[0, 0] if np.abs(velocities[0, 0]) < self.max_linear_velocity else \
            self.max_linear_velocity * np.sign(velocities[0, 0])
        velocities[1, 0] = velocities[1, 0] if np.abs(velocities[1, 0]) < self.max_angular_velocity else \
            self.max_angular_velocity * np.sign(velocities[1, 0])
        velocities[0, 1] = velocities[0, 1] if np.abs(velocities[0, 1]) < self.max_linear_velocity_hunter else \
            self.max_linear_velocity_hunter * np.sign(velocities[0, 1])
        velocities[1, 1] = velocities[1, 1] if np.abs(velocities[1, 1]) < self.max_angular_velocity_hunter else \
            self.max_angular_velocity_hunter * np.sign(velocities[1, 1])

        # Update dynamics of agents
        # velocities[0, i] 表示第 i 个机器人的线速度，而 velocities[1, i] 表示第 i 个机器人的角速度
        poses[0, :] = poses[0, :] + self.time_step * np.cos(poses[2, :]) * velocities[0, :]
        poses[1, :] = poses[1, :] + self.time_step * np.sin(poses[2, :]) * velocities[0, :]
        poses[2, :] = poses[2, :] + self.time_step * velocities[1, :]

        # Ensure angles are wrapped 确保机器人的方向角在合适的范围内，通常是 [-π, π]
        poses[2, :] = np.arctan2(np.sin(poses[2, :]), np.cos(poses[2, :]))

        # collision detect
        for robot in range(4):
            # collision with boundaries
            padding = 0.1
            poses[0, robot] = poses[0, robot] if poses[0, robot] > self.boundaries[0] + padding else \
                self.boundaries[0] + padding
            poses[0, robot] = poses[0, robot] if poses[0, robot] < self.boundaries[0] + self.boundaries[
                2] - padding else self.boundaries[0] + self.boundaries[2] - padding
            poses[1, robot] = poses[1, robot] if poses[1, robot] > self.boundaries[1] + padding else \
                self.boundaries[1] + padding
            poses[1, robot] = poses[1, robot] if poses[1, robot] < self.boundaries[0] + self.boundaries[
                3] - padding else self.boundaries[1] + self.boundaries[3] - padding

            # collision with barriers
            for barrier in self.barrier_centers:
                tempA = poses[:2, robot] - np.array(barrier)
                dist = np.linalg.norm(tempA)

                if dist < self.radius + padding:
                    tempA = tempA / dist * (self.radius + padding)
                    poses[:2, robot] = tempA + np.array(barrier)

        # collision with prey
        for i in range(3):
            tempB = poses[:2, i+1] - poses[:2, 0]
            dist_temp = np.linalg.norm(tempB)
            if dist_temp < 2*self.radius+0.2:
                tempB = tempB / dist_temp * (2*self.radius+0.2)
                poses[:2, i+1] = tempB + np.array(poses[:2, 0])

        for i in range(3):
            for j in range(3):
                if j==i:
                    continue
                tempB = poses[:2, i+1] - poses[:2, j+1]
                dist_temp = np.linalg.norm(tempB)
                if dist_temp < 2*self.radius+0.1:
                    tempB = tempB / dist_temp * (2*self.radius+0.1)
                    poses[:2, i+1] = tempB + np.array(poses[:2, j+1])

        # whether reach goal area
        tempC = poses[:2, 0] - np.array([-2.5, -2.5])
        dist_C = np.linalg.norm(tempC)
        if dist_C < 0.2:
            terminate = 1

        # compute the reward
        reward = self.get_reward(poses[:, 0], action)
        return reward

    def reset(self, initial_conditions=np.array([[2], [2], [0]])):
        assert initial_conditions.shape[1] > 0, "the initial conditions must not be empty"
        assert initial_conditions.shape[1] < 5, "More than 4 robot's initial conditions receive"
        if initial_conditions.shape[1] == 1: # 如果只有1列，在第二列增加一个全0向量
            random_array = np.random.uniform(-3.1, 3.1, size = (3, 3))
            self.poses = np.concatenate([initial_conditions.reshape(-1, 1), random_array], axis=1)
        elif initial_conditions.shape[1] == 4:
            self.poses = initial_conditions
        # temp = np.array([2, 2]) - np.array([-2.5, -2.5])
        # dist = np.linalg.norm(temp)
        state1 = np.append(self.poses[:, 0], self.poses[:, 1]) # 状态是把所有智能体的观测拼接起来的
        state2 = np.append(self.poses[:, 2], self.poses[:, 3])
        state = np.append(state1,state2)
        self.terminate = 0
        return state

    # def hunter_policy(self, hunter_states, prey_positions): # 追捕者的动作策略
    #     _, N = np.shape(hunter_states)  # 获取机器人状态的维度
    #     dxu = np.zeros((2, N))          # 储存追捕者的速度
    #     # print('states', hunter_states)
    #     # print('positions', hunter_states)
    #     pos_error = prey_positions - hunter_states[:2][:] # 计算猎人与猎物的相对位置
    #     rot_error = np.arctan2(pos_error[1][:], pos_error[0][:]) # 计算猎人需要旋转的角度
    #     dist = np.linalg.norm(pos_error, axis=0)                 # 计算相对距离
    #
    #     # 计算了猎人沿着目标方向移动的速度，移动系数0.8 * (dist + 0.2)
    #     dxu[0][:] = 0.8 * (dist + 0.2) * np.cos(rot_error - hunter_states[2][:])
    #     # 计算了猎人沿着目标方向旋转的速度，距离目标越远，旋转速度越快，旋转系数3 * dist
    #     dxu[1][:] = 3 * dist * np.sin(rot_error - hunter_states[2][:])
    #
    #     return dxu
    def hunter_policy(self, hunter_states, prey_positions):
        _, N_hunters = np.shape(hunter_states)
        _, N_prey = np.shape(prey_positions)
        # assert N_hunters == N_prey, "The number of hunters must be equal to the number of prey positions."

        dxu = np.zeros((2, N_hunters))
        v_omni_list=[]


        for i in range(N_hunters):
            pos_error = prey_positions[:, 0] - hunter_states[:2, i]
            rot_error = np.arctan2(pos_error[1], pos_error[0])
            dist = np.linalg.norm(pos_error)


            dxu[0, i] = 0.8 * (dist + 0.2) * np.cos(rot_error - hunter_states[2, i])
            dxu[1, i] = 3 * dist * np.sin(rot_error - hunter_states[2, i])

        # for i in range(N_hunters):
        #     tmp=np.array(dxu[0,i])
        #     v_omni=self.diff2omni(tmp,hunter_states[:,i])
        #     v_omni_list.append(v_omni)
        #
        # prey_pos=np.tile(prey_positions.flatten(),3)[:6]
        # prey_pos=prey_pos.reshape((3,2))
        # states = hunter_states[:2, ].T
        # v_des = compute_V_des(states,prey_pos, 0.2)
        #
        # v_update = rvo_vel(states, v_des, v_omni_list)
        # v_diff_list=[]
        # for i in range(N_hunters):
        #     v_diff = self.omni2diff(v_update[i])
        #     v_diff_list.append(v_diff)
        #
        # v_diff_list = [list(i) for i in zip(*v_diff_list)]

        # for i in range(N_hunters):
        #     for j in range(N_hunters):
        #         if j==i:
        #             continue
        #         dis=np.linalg.norm(hunter_states[:2,i]-hunter_states[:2,j])
        #         if dis<self.radius*2:
        #             dxu[0,i]=v_diff_list[0][i]
        #             dxu[1,i]=v_diff_list[1][i]
        #         else:
        #             dxu[0,i]=dxu[0,i]
        #             dxu[1,i]=dxu[1,i]

        return dxu

    ############### Add Your Code Here ##############################
    def get_reward(self, prey_state, action):
        # add you own reward function here
        hunter_state1 = self.poses[:, 1]
        hunter_state2 = self.poses[:,2]
        hunter_state3 = self.poses[:,3]
        reward1 = np.linalg.norm(hunter_state1[:2] - prey_state[:2])
        reward2 = np.linalg.norm(hunter_state2[:2] - prey_state[:2])
        reward3 = np.linalg.norm(hunter_state3[:2] - prey_state[:2])
        reward=reward2+reward3+reward1
        return reward
