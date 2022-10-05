import numpy as np
from PIL import Image
import numpy as np
import habitat_sim
from habitat_sim.utils.common import d3_40_colors_rgb
import cv2
import os


# This is the scene we are going to load.
# support a variety of mesh formats, such as .glb, .gltf, .obj, .ply
### put your scene path ###
test_scene = "apartment_0/habitat/mesh_semantic.ply"

sim_settings = {
    "scene": test_scene,  # Scene path
    "default_agent": 0,  # Index of the default agent
    "sensor_height": 1,  # Height of sensors in meters, relative to the agent
    "width": 512,  # Spatial resolution of the observations
    "height": 512,
    "sensor_pitch": 0,  # sensor pitch (x rotation in rads)
    "bev_height": 2.5,  # bev sensor height
    "bev_pitch": -(np.pi/2)  # bev sensor pitch
}

# This function generates a config for the simulator.
# It contains two parts:
# one for the simulator backend
# one for the agent, where you can attach a bunch of sensors

def transform_rgb_bgr(image):
    return image[:, :, [2, 1, 0]]

def transform_depth(image):
    depth_img = (image / 10 * 255).astype(np.uint8)
    return depth_img

def transform_semantic(semantic_obs):
    semantic_img = Image.new("P", (semantic_obs.shape[1], semantic_obs.shape[0]))
    semantic_img.putpalette(d3_40_colors_rgb.flatten())
    semantic_img.putdata((semantic_obs.flatten() % 40).astype(np.uint8))
    semantic_img = semantic_img.convert("RGB")
    semantic_img = cv2.cvtColor(np.asarray(semantic_img), cv2.COLOR_RGB2BGR)
    return semantic_img

def make_simple_cfg(settings):
    # simulator backend
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = settings["scene"]
    # agent
    agent_cfg = habitat_sim.agent.AgentConfiguration()

    # In the 1st example, we attach only one sensor,
    # a RGB visual sensor, to the agent
    rgb_sensor_spec = habitat_sim.CameraSensorSpec()
    rgb_sensor_spec.uuid = "color_sensor"
    rgb_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
    rgb_sensor_spec.resolution = [settings["height"], settings["width"]]
    rgb_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
    rgb_sensor_spec.orientation = [
        settings["sensor_pitch"],
        0.0,
        0.0,
    ]
    rgb_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE

    # bev sensor
    bev_sensor_spec = habitat_sim.CameraSensorSpec()
    bev_sensor_spec.uuid = "bev_sensor"
    bev_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
    bev_sensor_spec.resolution = [settings["height"], settings["width"]]
    bev_sensor_spec.position = [0.0, settings["bev_height"], 0.0]
    bev_sensor_spec.orientation = [
        settings["bev_pitch"],
        0.0,
        0.0,
    ]
    bev_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE

    #depth snesor
    depth_sensor_spec = habitat_sim.CameraSensorSpec()
    depth_sensor_spec.uuid = "depth_sensor"
    depth_sensor_spec.sensor_type = habitat_sim.SensorType.DEPTH
    depth_sensor_spec.resolution = [settings["height"], settings["width"]]
    depth_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
    depth_sensor_spec.orientation = [
        settings["sensor_pitch"],
        0.0,
        0.0,
    ]
    depth_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE

    agent_cfg.sensor_specifications = [rgb_sensor_spec, bev_sensor_spec, depth_sensor_spec]

    return habitat_sim.Configuration(sim_cfg, [agent_cfg])


cfg = make_simple_cfg(sim_settings)
sim = habitat_sim.Simulator(cfg)


# initialize an agent
agent = sim.initialize_agent(sim_settings["default_agent"])

# Set agent state
agent_state = habitat_sim.AgentState()
agent_state.position = np.array([0.0, 0.0, 0.0])  # agent in world space
agent.set_state(agent_state)

# obtain the default, discrete actions that an agent can perform
# default action space contains 3 actions: move_forward, turn_left, and turn_right
action_names = list(cfg.agents[sim_settings["default_agent"]].action_space.keys())
print("Discrete action space: ", action_names)


FORWARD_KEY="w"
LEFT_KEY="a"
RIGHT_KEY="d"
FINISH="f"
SAVE="s"
print("#############################")
print("use keyboard to control the agent")
print(" w for go forward  ")
print(" a for turn left  ")
print(" d for trun right  ")
print(" s for save data  ")
print(" f for finish and quit the program")
print("#############################")

if not os.path.exists("data/"):
    os.makedirs("data/")

if not os.path.exists("data/task1"):
    os.makedirs("data/task1")

last_observations = None
count = 0
def navigateAndSee(action=""):
    global count, last_observations
    if action == "save_data":
        print("save data")
        cv2.imwrite(f'data/task1/front_view_{count}.png', transform_rgb_bgr(last_observations["color_sensor"]))
        cv2.imwrite(f'data/task1/bev_view_{count}.png', transform_rgb_bgr(last_observations["bev_sensor"]))
        count += 1
        return count

    observations = sim.step(action)
    #print("action: ", action)

    cv2.imshow("RGB", transform_rgb_bgr(observations["color_sensor"]))
    cv2.imshow("BEV", transform_rgb_bgr(observations["bev_sensor"]))
    print(np.max(observations['depth_sensor']))
    
    agent_state = agent.get_state()
    sensor_state = agent_state.sensor_states['color_sensor']
    print("camera pose: x y z rw rx ry rz")
    print(sensor_state.position[0],sensor_state.position[1],sensor_state.position[2],  sensor_state.rotation.w, sensor_state.rotation.x, sensor_state.rotation.y, sensor_state.rotation.z)

    last_observations = observations

action = "move_forward"
navigateAndSee(action)

while True:
    keystroke = cv2.waitKey(0)
    if keystroke == ord(FORWARD_KEY):
        action = "move_forward"
        navigateAndSee(action)
        print("action: FORWARD")
    elif keystroke == ord(LEFT_KEY):
        action = "turn_left"
        navigateAndSee(action)
        print("action: LEFT")
    elif keystroke == ord(RIGHT_KEY):
        action = "turn_right"
        navigateAndSee(action)
        print("action: RIGHT")
    elif keystroke == ord(SAVE):
        print("action: SAVE")
        action = "save_data"
        navigateAndSee(action)
    elif keystroke == ord(FINISH):
        print("action: FINISH")
        break
    else:
        print("INVALID KEY")
        continue