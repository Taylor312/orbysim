import os
import sys
import numpy as np
import multiprocessing
import time

# --- 1. DLL ANCHOR ---
try:
    v_root = os.path.dirname(os.path.dirname(sys.executable))
    t_lib = os.path.join(v_root, "Lib", "site-packages", "torch", "lib")
    if os.path.exists(t_lib):
        os.add_dll_directory(t_lib)
    import torch
except Exception:
    pass

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.constants import OrbitronConfig
from utils.telemetry import run_telemetry

def main(telemetry_queue):
    from isaacsim import SimulationApp
    launch_config = {
        "headless": False,
        "width": 1280, "height": 720,
        "renderer": "RayTracedLighting",
        "display_options": 3094,
        "extra_args": [
            f"--/renderer/activeGpu={OrbitronConfig.TARGET_GPU_UUID}",
            "--/renderer/multiGpu/enabled=false",
            "--/physics/cudaDevice=0", 
        ]
    }

    print(f" [INFO] Orbitron Architecture: Phase 1 Vector Edition")
    simulation_app = SimulationApp(launch_config)

    from isaacsim.core.api import World
    from isaacsim.core.api.objects import GroundPlane
    from isaacsim.core.prims import Articulation
    from omni.isaac.core.prims import RigidPrimView 
    from isaacsim.core.utils.stage import add_reference_to_stage
    from pxr import UsdLux, Sdf, UsdPhysics, PhysxSchema
    import carb.input
    import omni.appwindow
    import pygame 

    from utils.urdf_compiler import compile_and_import
    from utils.chassis_surgeon import setup_orbitron_physics
    from utils.motor_sim import Apex3Motor
    from utils.simple_force_solver import SimpleForceSolver

    world = World(backend="torch", device="cuda:0")
    stage = world.stage

    scene_prim = stage.GetPrimAtPath("/physicsScene")
    if not scene_prim:
        scene_prim = UsdPhysics.Scene.Define(stage, Sdf.Path("/physicsScene")).GetPrim()
    PhysxSchema.PhysxSceneAPI.Apply(scene_prim).CreateSolverTypeAttr("TGS")

    light = UsdLux.DomeLight.Define(stage, Sdf.Path("/World/Sky"))
    light.CreateIntensityAttr(1000)
    ground = world.scene.add(GroundPlane(prim_path="/World/Ground", z_position=0))

    compile_and_import(OrbitronConfig.URDF_PATH, OrbitronConfig.USD_OUTPUT_PATH)
    add_reference_to_stage(usd_path=OrbitronConfig.USD_OUTPUT_PATH, prim_path=OrbitronConfig.ROBOT_PRIM_PATH)
    setup_orbitron_physics(stage, OrbitronConfig.ROBOT_PRIM_PATH)

    orbitron = Articulation(prim_paths_expr=OrbitronConfig.ROBOT_PRIM_PATH, name="Orbitron")
    world.scene.add(orbitron)

    chassis_view = RigidPrimView(prim_paths_expr=f"{OrbitronConfig.ROBOT_PRIM_PATH}/base_link", name="chassis_view")
    world.scene.add(chassis_view)
    world.reset()

    dof_names = orbitron.dof_names
    def get_dof_idx(keywords):
        for i, name in enumerate(dof_names):
            if all(k in name.lower() for k in keywords):
                return i
        return 0 

    fl_idx = get_dof_idx(["revolute_4"]) 
    fr_idx = get_dof_idx(["revolute_1"]) 
    bl_idx = get_dof_idx(["revolute_3"]) 
    br_idx = get_dof_idx(["revolute_2"]) 

    input_interface = carb.input.acquire_input_interface()
    appwindow = omni.appwindow.get_default_app_window()
    keyboard = appwindow.get_keyboard()

    pygame.init()
    pygame.joystick.init()
    use_gamepad = False
    joystick = None

    for i in range(pygame.joystick.get_count()):
        temp_joy = pygame.joystick.Joystick(i)
        name = temp_joy.get_name().lower()
        if "xbox" in name or "controller" in name or "xinput" in name:
            joystick = pygame.joystick.Joystick(i)
            joystick.init()
            use_gamepad = True
            print(f" [SUCCESS] Locked onto Gamepad: {joystick.get_name()}")
            break

    motor_fl = Apex3Motor()
    motor_fr = Apex3Motor()
    motor_bl = Apex3Motor()
    motor_br = Apex3Motor()
    force_solver = SimpleForceSolver()

    print(f" [INFO] Vector Drive System Active. Global Kinematics Inverted.")
    frame_count = 0

    while simulation_app.is_running():
        frame_count += 1
        accel = 0.0
        steer = 0.0
        
        if frame_count > 60:
            if use_gamepad:
                pygame.event.pump()
                accel = -joystick.get_axis(1)
                steer = joystick.get_axis(2) 
                if abs(accel) < 0.1: accel = 0.0
                if abs(steer) < 0.1: steer = 0.0
            else:
                up = input_interface.get_keyboard_value(keyboard, carb.input.KeyboardInput.UP)
                down = input_interface.get_keyboard_value(keyboard, carb.input.KeyboardInput.DOWN)
                left = input_interface.get_keyboard_value(keyboard, carb.input.KeyboardInput.LEFT)
                right = input_interface.get_keyboard_value(keyboard, carb.input.KeyboardInput.RIGHT)
                accel = float(up - down)       
                steer = float((right - left))    

        left_mix = accel + steer
        right_mix = accel - steer
        max_mag = max(1.0, abs(left_mix), abs(right_mix))
        
        throttle_fl = (left_mix / max_mag) * OrbitronConfig.INV_FL
        throttle_bl = (left_mix / max_mag) * OrbitronConfig.INV_BL
        throttle_fr = (right_mix / max_mag) * OrbitronConfig.INV_FR
        throttle_br = (right_mix / max_mag) * OrbitronConfig.INV_BR

        if orbitron.num_dof >= 4:
            current_rads = orbitron.get_joint_velocities()[0]
            c_rpm_fl = current_rads[fl_idx].item() * (30.0 / np.pi)
            c_rpm_fr = current_rads[fr_idx].item() * (30.0 / np.pi)
            c_rpm_bl = current_rads[bl_idx].item() * (30.0 / np.pi)
            c_rpm_br = current_rads[br_idx].item() * (30.0 / np.pi)

            torque_fl = motor_fl.compute_torque(throttle_fl, c_rpm_fl)
            torque_fr = motor_fr.compute_torque(throttle_fr, c_rpm_fr)
            torque_bl = motor_bl.compute_torque(throttle_bl, c_rpm_bl)
            torque_br = motor_br.compute_torque(throttle_br, c_rpm_br)

            f_thrust, t_yaw = force_solver.compute_chassis_forces(torque_fl, torque_fr, torque_bl, torque_br)

            action = torch.zeros(orbitron.num_dof, device="cuda:0")
            action[OrbitronConfig.IDX_FL] = throttle_fl * OrbitronConfig.MAX_TELEOP_RPM * (np.pi / 30.0)
            action[OrbitronConfig.IDX_FR] = throttle_fr * OrbitronConfig.MAX_TELEOP_RPM * (np.pi / 30.0)
            action[OrbitronConfig.IDX_BL] = throttle_bl * OrbitronConfig.MAX_TELEOP_RPM * (np.pi / 30.0)
            action[OrbitronConfig.IDX_BR] = throttle_br * OrbitronConfig.MAX_TELEOP_RPM * (np.pi / 30.0)
            orbitron.set_joint_velocity_targets(action)

            f_vec = torch.tensor([[f_thrust, 0.0, 0.0]], dtype=torch.float32, device="cuda:0")
            t_vec = torch.tensor([[0.0, 0.0, t_yaw]], dtype=torch.float32, device="cuda:0")
            chassis_view.apply_forces_and_torques_at_pos(forces=f_vec, torques=t_vec, is_global=False)

            try:
                telemetry_queue.put_nowait({
                    "accel": accel, "steer": steer,
                    "t_rpm_fl": throttle_fl * OrbitronConfig.MAX_TELEOP_RPM, "c_rpm_fl": c_rpm_fl,
                    "f_thrust": float(f_thrust), "t_yaw": float(t_yaw)
                })
            except:
                pass 

        world.step(render=True)

    simulation_app.close()

if __name__ == '__main__':
    tele_queue = multiprocessing.Queue()
    tele_process = multiprocessing.Process(target=run_telemetry, args=(tele_queue,))
    tele_process.start()

    try:
        main(tele_queue)
    finally:
        tele_process.terminate()