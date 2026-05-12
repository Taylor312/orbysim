import os
import sys
import torch
import numpy as np 
import time

# --- 1. CONFIGURATION ---
try:
    venv_root = os.path.dirname(os.path.dirname(sys.executable))
    torch_lib = os.path.join(venv_root, "Lib", "site-packages", "torch", "lib")
    if os.path.exists(torch_lib):
        os.add_dll_directory(torch_lib)
except Exception:
    pass

from isaacsim import SimulationApp

launch_config = {
    "headless": False,
    "width": 1280, "height": 720,
    "renderer": "RayTracedLighting",
    "exts": [
        "omni.physx.fabric", 
        "omni.kit.xr.core",             
        "omni.kit.xr.profile.vr",       
        "omni.kit.xr.system.openxr",
        "omni.kit.viewport.window",
        "omni.kit.viewport.utility"
    ], 
    "extra_args": [
        "--/renderer/activeGpu=0",
        "--/renderer/multiGpu/enabled=false",
        "--/physics/cudaDevice=0", 
        "--/rtx/divider/enabled=false",
        "--/rtx/optixDenoiser/enabled=false", 
        "--/rtx/denoiser/enabled=false"
    ]
}

print(f" [INFO] Booting Swervy Ultimate (5090 Native)...")
simulation_app = SimulationApp(launch_config)

# --- 2. IMPORTS ---
from omni.kit.viewport.utility import get_active_viewport_window, create_viewport_window, get_viewport_from_window_name
from isaacsim.core.utils.extensions import enable_extension
import omni.appwindow
import carb.input
import carb.settings
import omni.kit.commands
from isaacsim.core.api import World
from isaacsim.core.api.objects import GroundPlane
from isaacsim.core.prims import SingleArticulation as Articulation
from isaacsim.core.utils.stage import add_reference_to_stage
from pxr import UsdLux, Sdf, UsdGeom, Gf

# --- 3. PRE-LOAD EXTENSIONS ---
enable_extension("omni.kit.viewport.window")
enable_extension("omni.kit.xr.profile.vr")
enable_extension("omni.kit.xr.core")
for _ in range(5): simulation_app.update()

# --- 4. SCENE SETUP ---
world = World(backend="torch", device="cuda")
scene_path = "/physicsScene"
if not world.stage.GetPrimAtPath(scene_path):
    omni.kit.commands.execute("AddPhysicsScene", stage=world.stage, path=scene_path)
scene_prim = world.stage.GetPrimAtPath(scene_path)
scene_prim.CreateAttribute("physxScene:enableGPUDynamics", Sdf.ValueTypeNames.Bool).Set(True)
scene_prim.CreateAttribute("physxScene:broadphaseType", Sdf.ValueTypeNames.Token).Set("GPU")

light = UsdLux.DomeLight.Define(world.stage, Sdf.Path("/World/Sky"))
light.CreateIntensityAttr(1000)
world.scene.add(GroundPlane(prim_path="/World/Ground", z_position=0))

USD_OUTPUT_PATH = "C:/Users/taylo/Documents/robotexports/finalrobotone_description/urdf/finalrobotone.usd"
add_reference_to_stage(usd_path=USD_OUTPUT_PATH, prim_path="/World/finalrobotoneBot")

# --- 5. ROBOT SETUP ---
world.reset()
robot = world.scene.add(Articulation(prim_path="/World/finalrobotoneBot", name="Swervy"))
robot.set_enabled_self_collisions(False)
world.reset()
# Spawn 2m forward
robot.set_world_pose(position=torch.tensor([2.0, 0.0, 0.5], device="cuda"))

# --- 6. WARMUP ---
print(" [INFO] Warming up Simulation...")
for _ in range(60): world.step(render=True)

# --- 7. INPUT HANDLING ---
keys = {
    "W": False, "A": False, "S": False, "D": False,
    "UP": False, "DOWN": False, "LEFT": False, "RIGHT": False,
    "Q": False, "E": False 
}

settings = carb.settings.get_settings()
app_window = omni.appwindow.get_default_app_window()
input_interface = carb.input.acquire_input_interface()
keyboard = app_window.get_keyboard()

def on_key_event(e, *args):
    is_down = (e.type == carb.input.KeyboardEventType.KEY_PRESS)
    # WASD
    if e.input == carb.input.KeyboardInput.W: keys["W"] = is_down
    if e.input == carb.input.KeyboardInput.S: keys["S"] = is_down
    if e.input == carb.input.KeyboardInput.A: keys["A"] = is_down
    if e.input == carb.input.KeyboardInput.D: keys["D"] = is_down
    # Arrows (Flight)
    if e.input == carb.input.KeyboardInput.UP:    keys["UP"] = is_down
    if e.input == carb.input.KeyboardInput.DOWN:  keys["DOWN"] = is_down
    if e.input == carb.input.KeyboardInput.LEFT:  keys["LEFT"] = is_down
    if e.input == carb.input.KeyboardInput.RIGHT: keys["RIGHT"] = is_down
    if e.input == carb.input.KeyboardInput.Q:     keys["Q"] = is_down 
    if e.input == carb.input.KeyboardInput.E:     keys["E"] = is_down

sub_id = input_interface.subscribe_to_keyboard_events(keyboard, on_key_event)

# --- 8. TUNING & FLIPS ---
drive_names = ["front_left_w", "front_right_w", "back_left_w", "back_right_w"]
steer_names = ["front_left_mod", "front_right_mod", "back_left_mod", "back_right_mod"]

# FLIPPED: Front-Left and Back-Right are now multiplied by -1 relative to previous settings
DIRECTIONS = {
    # Drive Wheels
    "front_left_w": 1.0,   # Flipped from -1.0
    "front_right_w": -1.0, 
    "back_left_w": -1.0, 
    "back_right_w": 1.0,   # Flipped from -1.0
    
    # Steer Modules
    "front_left_mod": -1.0,  # Flipped from 1.0
    "front_right_mod": 1.0, 
    "back_left_mod": 1.0, 
    "back_right_mod": -1.0   # Flipped from 1.0
}

drive_indices = [robot.get_dof_index(n) for n in drive_names]
steer_indices = [robot.get_dof_index(n) for n in steer_names]
num_dof = robot.num_dof
physics_view = robot._articulation_view

# Turbo Power: KD=10000 for grip
physics_view.set_gains(
    kps=torch.zeros(num_dof, device="cuda"), 
    kds=torch.full((num_dof,), 10000.0, device="cuda") 
)

print(f" [READY] Controls:")
print(f"   WASD   -> Drive Robot")
print(f"   Arrows -> Fly Camera (Offset)")

# --- 9. MAIN LOOP ---
while simulation_app.is_running():
    
    # --- A. HYBRID CAMERA (Offset Mode) ---
    viewport_api = get_viewport_from_window_name("Viewport")
    if viewport_api:
        active_cam_path = viewport_api.get_active_camera()
        if active_cam_path:
            stage = world.stage
            prim = stage.GetPrimAtPath(active_cam_path)
            if prim:
                # We check for a PARENT Xform to move the 'Rig' instead of the camera itself
                # This lets the camera move freely within the rig (Head tracking)
                xform = UsdGeom.Xformable(prim)
                
                # If we modify the camera directly, we fight the VR headset.
                # Ideally, we move the camera's translation op by a tiny amount
                # respecting its current state.
                translate_op = None
                for op in xform.GetOrderedXformOps():
                    if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                        translate_op = op
                        break
                if not translate_op: translate_op = xform.AddTranslateOp()
                
                current_vec = translate_op.Get()
                if not current_vec: current_vec = Gf.Vec3d(0,0,0)
                
                speed = 0.05 
                # Move the camera base relative to world
                if keys["UP"]:    current_vec[0] -= speed 
                if keys["DOWN"]:  current_vec[0] += speed
                if keys["LEFT"]:  current_vec[1] += speed
                if keys["RIGHT"]: current_vec[1] -= speed
                if keys["E"]:     current_vec[2] += speed 
                if keys["Q"]:     current_vec[2] -= speed
                
                translate_op.Set(current_vec)

    # --- B. ROBOT DRIVE ---
    fwd = 0.0
    turn = 0.0
    if keys["W"]: fwd = 15.0
    if keys["S"]: fwd = -15.0
    if keys["A"]: turn = 5.0
    if keys["D"]: turn = -5.0
    
    velocities = torch.zeros(num_dof, device="cuda")
    for name, idx in zip(drive_names, drive_indices):
        velocities[idx] = fwd * DIRECTIONS[name]
    for name, idx in zip(steer_names, steer_indices):
        velocities[idx] = turn * DIRECTIONS[name]

    # --- C. CRASH PROTECTION ---
    try:
        physics_view.set_joint_velocity_targets(velocities)
    except Exception:
        print(" [INFO] VR Session Ended. Exiting Cleanly.")
        break
    
    simulation_app.update()
    world.step(render=True)

simulation_app.close()