import copy
from typing import Any, Dict, Union

import numpy as np
import sapien
import torch
import os
from transforms3d.euler import euler2quat

from mani_skill.agents.robots.unitree_g1.g1_upper_body import UnitreeG1UpperBodyWithHeadCamera

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.types import SimConfig
from mani_skill.utils.scene_builder.kitchen_counter import KitchenCounterSceneBuilder
from mani_skill.utils.structs.types import GPUMemoryConfig, SceneConfig
from mani_skill.envs.utils.randomization import random_quaternions
from mani_skill.utils.structs.pose import Pose

# YCB Asset Dataset
from mani_skill.utils.building.actors import ycb

FRIDGE_BOTTOM_Z_OFFSET = -0.9700819821870343     # Offset value of fridge to be on the ground

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
        pose = sapien.Pose(p=[-0.6, -1.3, 0.755])
        super()._load_agent(options, pose)

    def _load_scene(self, options: dict):
        self.scene_builder = KitchenCounterSceneBuilder(self)
        self.kitchen_scene = self.scene_builder.build(scale=self.kitchen_scene_scale)

        current_dir = os.path.dirname(__file__)
        model_dir = current_dir + "/assets"
        scale = self.kitchen_scene_scale

        # Refrigerator
        loader = self.scene.create_urdf_loader()
        urdf_path = "/11211/mobility.urdf"
        articulation_builders = loader.parse(model_dir + urdf_path)["articulation_builders"]
        builder = articulation_builders[0]
        builder.initial_pose = sapien.Pose(p=[0.2, -1.9, -FRIDGE_BOTTOM_Z_OFFSET])
        self.fridge = builder.build(name="fridge")

        # Bowl
        builder = self.scene.create_actor_builder()
        fix_rotation_pose = sapien.Pose(q=euler2quat(np.pi / 2, 0, 0))
        builder.add_nonconvex_collision_from_file(
            filename=os.path.join(model_dir, "frl_apartment_bowl_07.ply"),
            pose=fix_rotation_pose,
            scale=[scale] * 3,
        )
        builder.add_visual_from_file(
            filename=os.path.join(model_dir, "frl_apartment_bowl_07.glb"),
            pose=fix_rotation_pose,
            scale=[scale] * 3,
        )
        builder.set_initial_pose(sapien.Pose(p=[0.05, -1.3, 1.07], q=[0, 0, 0, 1]))
        self.bowl = builder.build_kinematic(name="bowl")

        # Apple
        model_id = "013_apple"
        builder = ycb.get_ycb_builder(
            scene=self.scene,
            id=model_id
        )
        builder.set_initial_pose(sapien.Pose(p=[0.2,-1.0, 1.0], q=[0, 0, 0, 1]))
        self.apple = builder.build(name=model_id)


    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        b = len(env_idx)
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
    Control the humanoid unitree G1 robot to grab an item and place it inside a fridge

    **Randomizations:**
    - the items's xy position is randomized on top of a table in the region [0.025, 0.025] x [-0.025, -0.025]. It is placed inside a bowl on the table
    - the item's z-axis rotation is randomized to a random angle

    **Success Conditions:**
    - the item position is within the bounds of the fridge's shelf.
    - the fridge door is closed

    **Goal Specification:**
    - The fridges's 3D position
    """ 
    SUPPORTED_ROBOTS = ["unitree_g1_simplified_upper_body_with_head_camera"]
    SUPPORTED_REWARD_MODES = ["normalized_dense", "dense", "sparse", "none"]
    agent: UnitreeG1UpperBodyWithHeadCamera

    def __init__(self, *args, robot_uids="unitree_g1_simplified_upper_body_with_head_camera", **kwargs):
        self.init_robot_pose = copy.deepcopy(
            UnitreeG1UpperBodyWithHeadCamera.keyframes["standing"].pose
        )
        self.init_robot_pose.p = [-0.4, -1.3, 0.755]
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
        #pose = sapien.Pose([-1.2, -1.5, 1.7], [1.0, 0.09, 0.3, -0.2])
        pose = sapien.Pose([0.279123, -0.503438, 1.54794], [0.252428, 0.396735, 0.114442, -0.875091])
        return CameraConfig(
            uid="base_camera", pose=pose, width=128, height=128, fov=np.pi / 2, near=0.01, far=100,
        )

    @property
    def _default_human_render_camera_configs(self):
        #pose = sapien.Pose([-1.2, -1.5, 1.7], [1.0, 0.09, 0.3, -0.2])
        pose = sapien.Pose([0.279123, -0.503438, 1.54794], [0.252428, 0.396735, 0.114442, -0.875091])
        return CameraConfig(
            uid="render_camera", pose=pose, width=512, height=512, fov=np.pi / 2, near=0.01, far=100,
        )

    def _load_scene(self, options: Dict):
        super()._load_scene(options)


    def _initialize_episode(self, env_idx: torch.Tensor, options: Dict):
        super()._initialize_episode(env_idx, options)
        with torch.device(self.device):
            b = len(env_idx)
        
            # Position G1RL 
            self.agent.robot.set_qpos(self.agent.keyframes["standing"].qpos)
            self.agent.robot.set_pose(self.init_robot_pose)

            # Randomizes position of the object
            base = torch.tensor([0.05, -1.3, 1.09])
            p = base.repeat(b, 1)
            offset = torch.rand((b, 2)) * 0.2 - 0.1
            p[:, :2] += offset
            qs = random_quaternions(b, lock_x=True, lock_y=True)
            obj_pose = Pose.create_from_pq(p=p, q=qs)
            self.apple.set_pose(obj_pose)

            # Position bowl
            p[:, 2] -= 0.2
            obj_pose = Pose.create_from_pq(p=p, q=qs)
            self.bowl.set_pose(obj_pose)

            # Position Fridge
            self.fridge.set_pose(sapien.Pose(p=[0.2, -1.9, -FRIDGE_BOTTOM_Z_OFFSET], q=[1, 0, 0, 0]))



    def evaluate(self):
        return {
            "success": torch.zeros(self.num_envs, device=self.device, dtype=bool),
            "fail": torch.zeros(self.num_envs, device=self.device, dtype=bool),
        }

    def _get_obs_extra(self, info: Dict):
        obs = dict(

        )
        return dict()

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        left_hand_to_obj_dist = torch.linalg.norm(
            self.apple.pose.p - self.agent.left_tcp.pose.p, axis=1
        )
        reaching_apple_reward = 1 - torch.tanh(5 * left_hand_to_obj_dist)
        reward = reaching_apple_reward

        return reward

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        max_reward = 1.0
        return self.compute_dense_reward(obs=obs, action=action, info=info) / max_reward
    
"""
CUDA_VISIBLE_DEVICES=0 python ppo.py --env_id="G1RL-v1" --no-capture-video \
    --num_envs=2048 --update_epochs=8 --num_minibatches=32 \
    --total_timesteps=4_000_000 --eval_freq=10 --num-steps=20
    
CUDA_VISIBLE_DEVICES=0 python ppo.py --env_id="G1RL-v1" --capture-video \
    --evaluate --checkpoint=runs/G1RL-v1__ppo__1__1752696234/final_ckpt.pt \
    --num_eval_envs=6 --num-eval-steps=1500 

watch -n 1 nvidia-smi
"""
