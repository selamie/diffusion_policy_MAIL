from gym.envs.registration import register
import diffusion_policy.env.panda

for reward_type in ["sparse", "dense"]:
    for control_type in ["ee", "joints"]:
        reward_suffix = "Dense" if reward_type == "dense" else ""
        control_suffix = "Joints" if control_type == "joints" else ""
        kwargs = {"reward_type": reward_type, "control_type": control_type}

register(
    id="PandaDiff{}{}-v0".format(control_suffix,reward_suffix),
    entry_point='envs.panda.panda_env:PandaReachDiffEnv',
    kwargs=kwargs,
    max_episode_steps=200
)


# entry_point='envs.pusht.pusht_keypoints_env:PushTKeypointsEnv',

# entry_point="panda_gym.envs:PandaReachDiffEnv",
# kwargs=kwargs,
# max_episode_steps=50,