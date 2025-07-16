import gymnasium as gym
import mani_skill.envs
import g1_rl
import fix_g1

env = gym.make(
    #"FIX-v1",
    "G1RL-v1",
    #"UnitreeG1PlaceAppleInBowl-v1",
    num_envs=1,
    obs_mode="state",
    control_mode="pd_joint_delta_pos",
    render_mode="human"
)
print("Observation Space", env.observation_space)
print("Action Space", env.action_space)

obs, _ = env.reset(seed=0)
done = False
while True:
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    env.render()
