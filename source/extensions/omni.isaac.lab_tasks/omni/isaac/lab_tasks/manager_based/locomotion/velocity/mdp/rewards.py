# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to define rewards for the learning environment.

The functions can be passed to the :class:`omni.isaac.lab.managers.RewardTermCfg` object to
specify the reward function and its parameters.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.sensors import ContactSensor
from omni.isaac.lab.utils.math import quat_rotate_inverse, yaw_quat

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv


def feet_air_time(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:
    """Reward long steps taken by the feet using L2-kernel.

    This function rewards the agent for taking steps that are longer than a threshold. This helps ensure
    that the robot lifts its feet off the ground and takes steps. The reward is computed as the sum of
    the time for which the feet are in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
   
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    #print(f"body_names: {sensor_cfg.body_names}")
    # print(f"first_contact:{first_contact}\n")
    # print(f"last_air_time:{last_air_time}\n")
    # print(f"current time : {contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]}")
    #print(f"body_ids:{sensor_cfg.body_ids}\n")
    # index = [4,8,14,18]
    # print(f"currenct contact time:{contact_sensor.data.current_air_time} + {contact_sensor.data.current_air_time[:,index]}")
    return reward


def feet_air_time_positive_biped(env, command_name: str, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward long steps taken by the feet for bipeds.

    This function rewards the agent for taking steps up to a specified threshold and also keep one foot at
    a time in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


def feet_slide(env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize feet sliding.

    This function penalizes the agent for sliding its feet on the ground. The reward is computed as the
    norm of the linear velocity of the feet multiplied by a binary contact sensor. This ensures that the
    agent is penalized only when the feet are in contact with the ground.
    """
    # Penalize feet sliding
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset = env.scene[asset_cfg.name]
    body_vel = asset.data.body_lin_vel_w[:, sensor_cfg.body_ids, :2]
    reward = torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)
    return reward


def track_lin_vel_xy_yaw_frame_exp(
    env, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) in the gravity aligned robot frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    vel_yaw = quat_rotate_inverse(yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3])
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - vel_yaw[:, :2]), dim=1
    )
    return torch.exp(-lin_vel_error / std**2)


def track_ang_vel_z_world_exp(
    env, command_name: str, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) in world frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_w[:, 2])
    # print(f"sudo {asset.data.root_ang_vel_w[:, 2]}")
    # print(f"xy vel: {env.command_manager.get_command(command_name)[:, 2]}")

    return torch.exp(-ang_vel_error / std**2)

def height_penalty(env, target_height: float = 0.25, weight: float = 1.0, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:

    # 获取机器人的基座高度
    base_height = env.scene[asset_cfg.name].data.root_pos_w[:, 2]
    # 计算基座高度与目标高度之间的偏差
    penalty = torch.abs(base_height - target_height) * weight

    # joint_velocity = env.scene[asset_cfg.name].data.joint_vel
    # print(f"关节速度是：{joint_velocity} ")
    # print(f"joint name: {env.scene[asset_cfg.name].data.joint_names}")
    # print(f"top4: {joint_velocity[:,:4]}")

    # print(f"Weight value used for height penalty: {weight}")
    # print(f"penalty: {penalty}")
    return penalty

def height_penalty_toolow(env, min_height: float = 0.55, weight: float = -2.0, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalty for base height being too low, to prevent the robot from collapsing on the ground."""
    # 获取机器人的基座高度
    base_height = env.scene[asset_cfg.name].data.root_pos_w[:, 2]
    # 计算低于最小高度的部分
    height_diff = torch.clamp(min_height - base_height, min=0.0)
    # 计算惩罚值
    penalty = height_diff * weight
    return penalty

def diagonal_gait_reward(env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:
    """
    Reward for encouraging diagonal gait.

    This function rewards the agent for keeping diagonal feet (e.g., left front and right hind, 
    right front and left hind) in contact with the ground or in the air simultaneously, which encourages
    a diagonal gait.

    Args:
        env: The reinforcement learning environment.
        command_name: The name of the command used to determine if the agent should move.
        sensor_cfg: The configuration for the contact sensors of the feet.
        weight: The weight of the reward.

    Returns:
        A tensor representing the reward for diagonal gait.
    """
    # 获取接触传感器的数据
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    first_contact = contact_sensor.data.current_contact_time[:,sensor_cfg.body_ids]

    # 假设有四条腿，分别编号为 0: 左前腿, 1: 右前腿, 2: 左后腿, 3: 右后腿
    left_front = first_contact[:, 0] > 0.0
    right_front = first_contact[:, 1] > 0.0
    left_hind = first_contact[:, 2] > 0.0
    right_hind = first_contact[:, 3] > 0.0

    # 鼓励对角脚同时接触地面
    diagonal_contact_1 = left_front * right_hind
    diagonal_contact_2 = right_front * left_hind
    
    # 计算奖励值
    reward = (diagonal_contact_1 + diagonal_contact_2)

    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.05
    return reward

def hip_movement_penalty(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), weight: float = -0.5) -> torch.Tensor:

    """
    Penalty for hip joint movement during straight walking to avoid unnatural gait.

    This function penalizes the agent for excessive movement of the hip joints during straight walking,
    which helps in preventing unnatural gaits like the "duck walk" or excessive external rotation.

    Args:
        env: The reinforcement learning environment.
        sensor_cfg: The configuration for the joints of the feet.
        weight: The weight of the penalty.

    Returns:
        A tensor representing the penalty for hip joint movement.
    """
    # root_lin_vel_w
    velocity_xy = env.scene[asset_cfg.name].data.root_lin_vel_w[:,:2]
    hip_joint_velocity = env.scene[asset_cfg.name].data.joint_vel[:,:4]
    # root_lin_vel_x = command_velocity[0]
    # root_lin_vel_y = command_velocity[1]
    # print(f"关节速度是：{joint_velocity} ")
    # print(f"joint name: {env.scene[asset_cfg.name].data.joint_names}")
    # 计算惩罚值
    
    movement_condition = torch.abs(velocity_xy[:, 0]) / (torch.abs(velocity_xy[:, 1]) + 1e-6) > 1.3
    # excessive_hip_movement = torch.abs(hip_joint_velocity) > 0.1  # 髋关节角速度阈值
    # hip_movement_penalty = torch.sum(excessive_hip_movement.float() * torch.abs(hip_joint_velocity), dim=1) * movement_condition.float()

    
    hip_movement_penalty = torch.sum(torch.abs(hip_joint_velocity), dim=1) * movement_condition.float()

    # 返回惩罚值
    penalty = hip_movement_penalty * weight
    # hip_movement_penalty = torch.sum(torch.abs(hip_joint_velocity), dim=1) * movement_condition.float()

    # # 返回惩罚值
    # penalty = hip_movement_penalty * weight

    return penalty