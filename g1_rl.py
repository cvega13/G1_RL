import copy
from typing import Any, Dict, List, Optional, Union

import numpy as np
import sapien
import torch
import os
import trimesh
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
from mani_skill.utils.structs import Articulation, Link, Pose
from mani_skill.utils.geometry.geometry import transform_points

# YCB Asset Dataset
from mani_skill.utils.building.actors import ycb
from mani_skill.utils.building import actors

FRIDGE_BOTTOM_Z_OFFSET = -0.9700819821870343        # Offset value of fridge to be on the ground
door_open_qpos = [-0.651, -0.651, 0.276, 0.276]     # Rotation fridge door when open
door_close_qpos = [-0.5, -0.5, 0.5, 0.5]            # Rotation of fridge door when closed

class HumanoidPickPlaceEnv(BaseEnv):
    SUPPORTED_REWARD_MODES = ["sparse", "none"]
    """sets up a basic scene with an item to pick up and a fridge to place it in"""
    kitchen_scene_scale = 1.0
    joint_types = ["revolute", "revolute_unwrapped"]


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
        pose = sapien.Pose(p=[-0.5, -1.34, 0.755])
        super()._load_agent(options, pose)

        self.right_arm_joints = [
            "right_shoulder_pitch_joint",
            "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint",
            "right_elbow_pitch_joint",
            "right_elbow_roll_joint",
            #"right_zero_joint",
            #"right_three_joint",
            #"right_five_joint",
            #"right_one_joint",
            #"right_four_joint",
            #"right_six_joint",
            #"right_two_joint",
        ]
        self.right_arm_joint_indexes = [
            self.agent.robot.active_joints_map[joint].active_index[0].item()
            for joint in self.right_arm_joints
        ]

        self.left_arm_joints = [
            "left_shoulder_pitch_joint",
            "left_shoulder_roll_joint",
            "left_shoulder_yaw_joint",
            "left_elbow_pitch_joint",
            "left_elbow_roll_joint",
            #"left_zero_joint",
            #"left_three_joint",
            #"left_five_joint",
            #"left_one_joint",
            #"left_four_joint",
            #"left_six_joint",
            #"left_two_joint",
        ]
        self.left_arm_joint_indexes = [
            self.agent.robot.active_joints_map[joint].active_index[0].item()
            for joint in self.left_arm_joints
        ]

    def _load_scene(self, options: dict):
        self.scene_builder = KitchenCounterSceneBuilder(self)
        self.kitchen_scene = self.scene_builder.build(scale=self.kitchen_scene_scale)

        current_dir = os.path.dirname(__file__)
        model_dir = current_dir + "/assets"
        scale = self.kitchen_scene_scale

        # Refrigerator
        urdf_path = "/11211/mobility.urdf"
        self.fridge = self._load_fridge(model_dir + urdf_path)

        # Bowl
        # builder = self.scene.create_actor_builder()
        # fix_rotation_pose = sapien.Pose(q=euler2quat(np.pi / 2, 0, 0))
        # builder.add_nonconvex_collision_from_file(
        #     filename=os.path.join(model_dir, "frl_apartment_bowl_07.ply"),
        #     pose=fix_rotation_pose,
        #     scale=[scale] * 3,
        # )
        # builder.add_visual_from_file(
        #     filename=os.path.join(model_dir, "frl_apartment_bowl_07.glb"),
        #     pose=fix_rotation_pose,
        #     scale=[scale] * 3,
        # )
        # builder.set_initial_pose(sapien.Pose(p=[0.05, -1.3, 1.07], q=[0, 0, 0, 1]))
        # self.bowl = builder.build_kinematic(name="bowl")

        # Apple
        model_id = "013_apple"
        builder = ycb.get_ycb_builder(
            scene=self.scene,
            id=model_id
        )
        builder.set_initial_pose(sapien.Pose(p=[0.2,-1.0, 1.0], q=[0, 0, 0, 1]))
        self.apple = builder.build(name=model_id)


    def _load_fridge(self, file_path):
        loader = self.scene.create_urdf_loader()
        articulation_builders = loader.parse(file_path)["articulation_builders"]
        builder = articulation_builders[0]
        builder.initial_pose = sapien.Pose(p=[0.2, -1.9, -FRIDGE_BOTTOM_Z_OFFSET])
        fridge = builder.build(name="fridge")

        handle_links: List[Link] = []
        handle_link_meshes: List[trimesh.Trimesh] = []

        # Find link labeled as a handle
        for link, joint in zip(fridge.links, fridge.joints):
            if joint.type[0] in self.joint_types:
                handle_links.append(link)
                handle_link_meshes.append(
                    link.generate_mesh(
                        filter=lambda _, render_shape: "handle"
                        in render_shape.name,
                        mesh_name="handle"
                    )[0]
                )

        self.handle_link = handle_links[0]
        handle_pos_np = np.array(handle_link_meshes[0].bounding_box.center_mass)
        self.handle_link_pos = common.to_tensor(np.tile(handle_pos_np, (self.num_envs, 1)), device=self.device)
        # Sphere representing the handle's center of mass
        self.handle_link_goal = actors.build_sphere(
            self.scene,
            radius=0.02,
            color=[0.5, 0, 1, 0.8],
            name="handle_link_goal",
            body_type="kinematic",
            add_collision=False,
            initial_pose=sapien.Pose(p=[0, 0, 0], q=[1, 0, 0, 0]),
        )

        return fridge

    # Apply's fridge door's transformatin matrix to handle's center of mass position
    def handle_link_positions(self, env_idx: Optional[torch.Tensor] = None):
        if env_idx is None:
            return transform_points(
                self.handle_link.pose.to_transformation_matrix().clone(),
                common.to_tensor(self.handle_link_pos, device=self.device),
            )
        return transform_points(
            self.handle_link.pose[env_idx].to_transformation_matrix().clone(),
            common.to_tensor(self.handle_link_pos[env_idx], device=self.device),
        )

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
        self.init_robot_pose.p = [-0.5, -1.3, 0.755]
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sim_config(self):
        return SimConfig(
            gpu_memory_config=GPUMemoryConfig(
                max_rigid_contact_count=2**22, max_rigid_patch_count=2**21,
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
            base = torch.tensor([-0.05, -1.3, 1.0])
            p = base.repeat(b, 1)
            offset = torch.rand((b, 2)) * 0.2 - 0.1
            p[:, :2] += offset
            qs = random_quaternions(b, lock_x=True, lock_y=True)
            obj_pose = Pose.create_from_pq(p=p, q=qs)
            self.apple.set_pose(obj_pose)

            # Position bowl
            # p[:, 2] -= 0.2
            # obj_pose = Pose.create_from_pq(p=p, q=qs)
            # self.bowl.set_pose(obj_pose)

            # Position Fridge
            self.fridge.set_pose(sapien.Pose(p=[0.2, -1.9, -FRIDGE_BOTTOM_Z_OFFSET], q=[1, 0, 0, 0]))

            # Position Handle
            self.handle_link_goal.set_pose(
                Pose.create_from_pq(p=self.handle_link_positions(env_idx))
            )


    def _after_control_step(self):
        # after each control step, we update the goal position of the handle link
        # for GPU sim we need to update the kinematics data to get latest pose information for up to date link poses
        # and fetch it, followed by an apply call to ensure the GPU sim is up to date
        if self.gpu_sim_enabled:
            self.scene.px.gpu_update_articulation_kinematics()
            self.scene._gpu_fetch_all()
        self.handle_link_goal.set_pose(
            Pose.create_from_pq(p=self.handle_link_positions())
        )
        if self.gpu_sim_enabled:
            self.scene._gpu_apply_all()


    def evaluate(self):
        is_grasped = self.agent.left_hand_is_grasping(self.apple, max_angle=85)
        door_is_closed = torch.linalg.norm(
            self.fridge.find_link_by_name("link_0").pose.q
            - torch.tensor(door_close_qpos, device=self.device), axis=1
        ) < 0.01

        door_is_open = torch.linalg.norm(
            self.fridge.find_link_by_name("link_0").pose.q
            - torch.tensor(door_open_qpos, device=self.device), axis=1
        ) < 0.05

        apple_fell = self.apple.pose.p[..., 2] <= 0.5
        handle_link_pos = self.handle_link_positions()

        return {
            "success": door_is_open,      # Temporary Success Condition
            "fail": apple_fell,
            "is_grasped": is_grasped,
            "door_is_closed": door_is_closed,
            "door_is_open": door_is_open,
            "handle_link_pos": handle_link_pos,
        }


    def _get_obs_extra(self, info: Dict):
        obs = dict(
            left_hand_tcp_pos = self.agent.left_tcp.pose.p,
            right_hand_tcp_pos = self.agent.right_tcp.pose.p,
            is_grasped=info["is_grasped"]
        )
        if "state" in self.obs_mode:
            obs.update(
                handle_link_pos = info["handle_link_pos"],
                fridge_door_qpos = self.fridge.find_link_by_name("link_0").pose.q,
                apple_pos = self.apple.pose.p
            )
        return obs


    def joint_velocity(self, joint_indices):
        return torch.mean(
            torch.abs(self.agent.robot.qvel[:, joint_indices]), dim=1
        )


    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        # Reach for apple
        left_tcp_to_obj_dist = torch.linalg.norm(
            self.apple.pose.p - self.agent.left_tcp.pose.p, axis=1
        )
        reaching_apple_reward = 1 - torch.tanh(5 * left_tcp_to_obj_dist)
        reward = reaching_apple_reward

        # Grasp apple
        is_grasped = info["is_grasped"]
        reward += is_grasped


        right_arm_velocity = self.joint_velocity(self.right_arm_joint_indexes)
        right_arm_static_reward = 1 - torch.tanh(5 * right_arm_velocity)
        reward[~is_grasped] += right_arm_static_reward[~is_grasped]

        left_arm_velocity = self.joint_velocity(self.left_arm_joint_indexes)
        left_arm_static_reward = 1 - torch.tanh(5 * left_arm_velocity)
        reward[is_grasped] += left_arm_static_reward[is_grasped]

        ## CONSIDER 
        # Reward for grasping apple stable
        # Add function to detect grasping with handle using get_pairwise_contact_forces

        right_tcp_to_handle_dist = torch.linalg.norm(
            info["handle_link_pos"] - self.agent.right_tcp.pose.p, axis=1 
        )
        reaching_handle_reward = 1 - torch.tanh(5 * right_tcp_to_handle_dist)
        reward[is_grasped] += reaching_handle_reward[is_grasped]

        # Reward to open the door
        open_fridge_diff = torch.linalg.norm(
            self.fridge.find_link_by_name("link_0").pose.q
            - torch.tensor(door_open_qpos, device=self.device), axis=1
        )
        open_door_reward = 1 - torch.tanh(5 * open_fridge_diff)
        reward[is_grasped] += open_door_reward[is_grasped]

        reward[info["fail"]] = 0.0
        reward[info["success"]] = 5.0

        return reward

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        max_reward = 5.0
        return self.compute_dense_reward(obs=obs, action=action, info=info) / max_reward
    
"""
CUDA_VISIBLE_DEVICES=0 python ppo.py --env_id="G1RL-v1" --no-capture-video \
    --num_envs=2048 --update_epochs=8 --num_minibatches=32 \
    --total_timesteps=8_000_000 --eval_freq=10 --num-steps=20
    
CUDA_VISIBLE_DEVICES=0 python ppo.py --env_id="G1RL-v1" --capture-video \
    --evaluate --checkpoint=runs/G1RL-v1__ppo__1__1753382916/final_ckpt.pt \
    --num_eval_envs=6 --num-eval-steps=1500 

watch -n 1 nvidia-smi
"""
