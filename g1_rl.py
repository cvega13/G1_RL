import copy
from typing import Any, Dict, Union

import numpy as np
import sapien
import torch
import os

from mani_skill.agents.robots.unitree_g1.g1 import UnitreeG1

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.types import SimConfig
from mani_skill.utils.scene_builder.kitchen_counter import KitchenCounterSceneBuilder
from mani_skill.utils.structs.types import GPUMemoryConfig, SceneConfig


class HumanoidPickPlaceEnv(BaseEnv):
    SUPPORTED_REWARD_MODES = ["sparse", "none"]
    """sets up a basic scene with an item to pick up and a fridge to place it in"""
    kitchen_scene_scale = 1.0

    def __init__(self, *args, robot_init_qpos_noise=0.02, **kwargs):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        super().__init__(*args, **kwargs)

    @property
    def _default_sim_config(self):
        return SimConfig(
            gpu_memory_config=GPUMemoryConfig(
                max_rigid_contact_count=2**22,
            )
        )

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(eye=[0.3, 0, 0.6], target=[-0.1, 0, 0.1])
        return [
            CameraConfig("base_camera", pose=pose, width=128, height=128, fov=np.pi / 2)
        ]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([0.6, 0.7, 0.6], [0.0, 0.0, 0.35])
        return CameraConfig("render_camera", pose=pose, width=512, height=512, fov=1)

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[0, 0, 1]))

    def _load_scene(self, options: dict):
        self.scene_builder = KitchenCounterSceneBuilder(self)
        self.kitchen_scene = self.scene_builder.build(scale=self.kitchen_scene_scale)

        # Refrigerator
        loader = self.scene.create_urdf_loader()
        self.scene.set_timestep(1 / 100.0)
        loader.fix_root_link = True
        current_file_path = os.path.abspath(__file__)
        current_directory = os.path.dirname(current_file_path)
        urdf_path = "/assets/11211/mobility.urdf"
        self.fridge = loader.load(current_directory + urdf_path)
        assert self.fridge, "failed to load URDF."


    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        self.scene_builder.initialize(env_idx)

    def evaluate(self):
        return {
            "success": torch.zeros(self.num_envs, device=self.device, dtype=bool),
            "fail": torch.zeros(self.num_envs, device=self.device, dtype=bool),
        }

    def _get_obs_extra(self, info: Dict):
        return dict()



@register_env("G1RL-v1", max_episode_steps=200)
class UnitreeG1PutInFridge(HumanoidPickPlaceEnv):
    """
    **Task Description:**
    Control the humanoid unitree G1 robot to grab an item with its right arm and place it inside a fridge

    **Randomizations:**
    - the items's xy position is randomized on top of a table in the region [0.025, 0.025] x [-0.025, -0.025]. It is placed flat on the table
    - the item's z-axis rotation is randomized to a random angle

    **Success Conditions:**
    - the item position is within the bounds of the fridge's shelf.
    - the fridge door is closed

    **Goal Specification:**
    - The fridges's 3D position
    """ 
    SUPPORTED_ROBOTS = ["UnitreeG1"]
    SUPPORTED_REWARD_MODES = ["normalized_dense", "dense", "sparse", "none"]
    agent: Union[UnitreeG1]

    def __init__(self, *args, robot_uids="unitree_g1", **kwargs):
        self.init_robot_pose = copy.deepcopy(
            UnitreeG1.keyframes["standing"].pose
        )
        self.init_robot_pose.p = [-0.6, 0, 0.755]
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sim_config(self):
        return SimConfig(
            gpu_memory_config=GPUMemoryConfig(
                max_rigid_contact_count=2**22, max_rigid_patch_count=2**21
            ),
            # TODO (stao): G1 robot may need some custom collision disabling as the dextrous fingers may often be close to each other
            # and slow down simulation. A temporary fix is to reduce contact_offset value down so that we don't check so many possible
            # collisions
            scene_config=SceneConfig(contact_offset=0.01),
        )

    @property
    def _default_sensor_configs(self):
        pose = sapien.Pose([0.279123, 0.303438, 1.34794], [0.252428, 0.396735, 0.114442, -0.875091])
        return CameraConfig(
            uid="base_camera", pose=pose, width=128, height=128, fov=np.pi / 2, near=0.01, far=100,
        )

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien.Pose([0.279123, 0.303438, 1.34794], [0.252428, 0.396735, 0.114442, -0.875091])
        return CameraConfig(
            uid="render_camera", pose=pose, width=512, height=512, fov=np.pi / 2, near=0.01, far=100,
        )

    def _load_scene(self, options: Dict):
        super()._load_scene(options)

        # Position Fridge
        mesh = self.fridge.get_first_visual_mesh()
        z_min = mesh.bounding_box.bounds[0][2]
        self.fridge.initial_pose = sapien.Pose(p=[0.15, -1.9, -z_min])  # Place it visibly



    def _initialize_episode(self, env_idx: torch.Tensor, options: Dict):
        super()._initialize_episode(env_idx, options)
        with torch.device(self.device):
            b = len(env_idx)
            
            # Position G1RL
            self.agent.robot.set_qpos(self.agent.keyframes["standing"].qpos)
            self.agent.robot.set_pose(self.init_robot_pose)

    def evaluate(self):
        return {
            "success": torch.zeros(self.num_envs, device=self.device, dtype=bool),
            "fail": torch.zeros(self.num_envs, device=self.device, dtype=bool),
        }

    def _get_obs_extra(self, info: Dict):
        return dict()

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        return torch.zeros(self.num_envs, device=self.device)

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        max_reward = 1.0
        return self.compute_dense_reward(obs=obs, action=action, info=info) / max_reward
    