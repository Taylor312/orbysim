import os
import sys
import numpy as np
import multiprocessing
import time

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

def main(telemetry_queue, speed_limit_val):
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

    print(f" [INFO] Orbitron Architecture: Phase 1 Vector Edition (2026 Build)")
    simulation_app = SimulationApp(launch_config)

    from isaacsim.core.api import World
    from isaacsim.core.api.objects import GroundPlane
    from isaacsim.core.prims import Articulation
    from isaacsim.core.prims import RigidPrim                 
    from isaacsim.core.utils.stage import add_reference_to_stage
    from isaacsim.util.debug_draw import _debug_draw          
    from pxr import UsdLux, Sdf, UsdPhysics, PhysxSchema

    import carb.input
    import omni.appwindow
    import pygame 

    from old.urdf_compiler import compile_and_import
    from old.chassis_surgeon import setup_orbitron_physics
    from utils.motor_sim import Apex3Motor
    from old.simple_force_solver import SimpleForceSolver

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

    chassis_view = RigidPrim(prim_paths_expr=f"{OrbitronConfig.ROBOT_PRIM_PATH}/base_link", name="chassis_view")
    world.scene.add(chassis_view)
    world.reset()
    
    draw = _debug_draw.acquire_debug_draw_interface()

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
    
    dt = OrbitronConfig.PHYSICS_DT

    while simulation_app.is_running():
        frame_count += 1
        
        if frame_count < 15:
            world.step(render=True)
            continue

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

        # Turning authority dampener
        steer = steer * 0.10

        left_mix = accel + steer
        right_mix = accel - steer
        max_mag = max(1.0, abs(left_mix), abs(right_mix))
        
        if accel == 0.0 and steer == 0.0:
            motor_fl.integral_error = 0.0
            motor_fr.integral_error = 0.0
            motor_bl.integral_error = 0.0
            motor_br.integral_error = 0.0
        
        throttle_fl = (left_mix / max_mag) * OrbitronConfig.INV_FL
        throttle_bl = (left_mix / max_mag) * OrbitronConfig.INV_BL
        throttle_fr = (right_mix / max_mag) * OrbitronConfig.INV_FR
        throttle_br = (right_mix / max_mag) * OrbitronConfig.INV_BR

        if chassis_view.get_masses() is not None and orbitron.num_dof >= 4:
            real_mass = chassis_view.get_masses()[0].item()
            required_downforce_n = (real_mass * 9.81) * 4.5

            current_rads = orbitron.get_joint_velocities()[0]
            c_rpm_fl = current_rads[fl_idx].item() * (30.0 / np.pi)
            c_rpm_fr = current_rads[fr_idx].item() * (30.0 / np.pi)
            c_rpm_bl = current_rads[bl_idx].item() * (30.0 / np.pi)
            c_rpm_br = current_rads[br_idx].item() * (30.0 / np.pi)

            # Extract world vectors cleanly
            poses, quats = chassis_view.get_world_poses()
            poses_clone = poses.clone()
            c_pos = poses[0].cpu().numpy()
            c_quat = quats[0].cpu().numpy()
            w, x, y, z = c_quat
            
            rot_matrix = np.array([
                [1 - 2*(y**2 + z**2), 2*(x*y - z*w), 2*(x*z + y*w)],
                [2*(x*y + z*w), 1 - 2*(x**2 + z**2), 2*(y*z - x*w)],
                [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x**2 + y**2)]
            ])

            linear_velocities = chassis_view.get_linear_velocities()
            lin_vel = linear_velocities[0].cpu().numpy()

            # Heading dot-products
            world_forward_vec = rot_matrix[:, 1]
            world_lateral_vec = rot_matrix[:, 0]
            
            forward_speed_ms = -np.dot(lin_vel, world_forward_vec) 
            
            if abs(forward_speed_ms) < 0.005: 
                forward_speed_ms = 0.0

            wheel_rpm = (forward_speed_ms / OrbitronConfig.WHEEL_RADIUS_M) * (30.0 / np.pi)
            motor_c_rpm = wheel_rpm * OrbitronConfig.GEAR_RATIO

            dynamic_max_rpm = OrbitronConfig.MAX_TELEOP_RPM * speed_limit_val.value

            torque_fl, _ = motor_fl.compute_torque(throttle_fl * dynamic_max_rpm, motor_c_rpm, dt)
            torque_fr, _ = motor_fr.compute_torque(throttle_fr * dynamic_max_rpm, motor_c_rpm, dt)
            torque_bl, _ = motor_bl.compute_torque(throttle_bl * dynamic_max_rpm, motor_c_rpm, dt)
            torque_br, _ = motor_br.compute_torque(throttle_br * dynamic_max_rpm, motor_c_rpm, dt)

            wheel_torque_fl = torque_fl * OrbitronConfig.GEAR_RATIO * OrbitronConfig.INV_FL
            wheel_torque_fr = torque_fr * OrbitronConfig.GEAR_RATIO * OrbitronConfig.INV_FR
            wheel_torque_bl = torque_bl * OrbitronConfig.GEAR_RATIO * OrbitronConfig.INV_BL
            wheel_torque_br = torque_br * OrbitronConfig.GEAR_RATIO * OrbitronConfig.INV_BR

            net_force, net_torque, slips, raw_forces = force_solver.compute_4_corner_forces(
                wheel_torque_fl, wheel_torque_fr, wheel_torque_bl, wheel_torque_br,
                required_downforce_n=required_downforce_n, mu=1.0
            )

            # Visual mesh tracking
            slip_fl, slip_fr, slip_bl, slip_br = slips
            slip_scalar = 5.0 
            actual_wheel_rads = wheel_rpm * (np.pi / 30.0)
            
            action = torch.zeros(orbitron.num_dof, device="cuda:0")
            action[fl_idx] = (actual_wheel_rads + (slip_fl * slip_scalar)) * OrbitronConfig.INV_FL
            action[fr_idx] = (actual_wheel_rads + (slip_fr * slip_scalar)) * OrbitronConfig.INV_FR
            action[bl_idx] = (actual_wheel_rads + (slip_bl * slip_scalar)) * OrbitronConfig.INV_BL
            action[br_idx] = (actual_wheel_rads + (slip_br * slip_scalar)) * OrbitronConfig.INV_BR
            orbitron.set_joint_velocity_targets(action)

            # Flat projection transformations
            flat_x = rot_matrix[:, 0]
            flat_y = rot_matrix[:, 1]
            flat_x[2] = 0.0
            flat_y[2] = 0.0
            if np.linalg.norm(flat_x) > 0.001: flat_x = flat_x / np.linalg.norm(flat_x)
            if np.linalg.norm(flat_y) > 0.001: flat_y = flat_y / np.linalg.norm(flat_y)
            flat_rot_matrix = np.column_stack((flat_x, flat_y, np.array([0.0, 0.0, 1.0])))
            
            global_force = np.dot(flat_rot_matrix, net_force)
            global_torque = np.dot(rot_matrix, net_torque)

            if np.isnan(global_force).any() or np.isnan(global_torque).any():
                global_force = np.zeros(3)
                global_torque = np.zeros(3)

            f_vec_2d = torch.as_tensor(global_force, dtype=torch.float32, device="cuda:0").unsqueeze(0)
            t_vec_2d = torch.as_tensor(global_torque, dtype=torch.float32, device="cuda:0").unsqueeze(0)
            chassis_view.apply_forces_and_torques_at_pos(forces=f_vec_2d, torques=t_vec_2d, positions=poses_clone, is_global=True)

            # --- THE KINEMATIC VELOCITY ALIGNMENT FILTER ---
            # Extract fresh, post-force physics state
            fresh_lin_vel = chassis_view.get_linear_velocities()[0].cpu().numpy()
            local_lin_vel = np.dot(rot_matrix.T, fresh_lin_vel)
            
            # Attenuate the local lateral sliding speed by 15% every single frame.
            # This implicitly acts as an infinite-stability lateral rubber grip model!
            local_lin_vel[0] *= 0.85 
            
            clean_global_lin_vel = np.dot(rot_matrix, local_lin_vel)
            chassis_view.set_linear_velocities(torch.as_tensor(clean_global_lin_vel, dtype=torch.float32, device="cuda:0").unsqueeze(0))

            # Handle active yaw stabilizing when steering input is neutral
            fresh_ang_vel = chassis_view.get_angular_velocities()[0].cpu().numpy()
            if steer == 0.0:
                fresh_ang_vel[2] *= 0.90
            chassis_view.set_angular_velocities(torch.as_tensor(fresh_ang_vel, dtype=torch.float32, device="cuda:0").unsqueeze(0))
            # -----------------------------------------------------------------

            draw.clear_lines()
            if np.linalg.norm(net_force) > 1.0 or np.linalg.norm(net_torque) > 1.0:
                pos_list = [force_solver.pos_fl, force_solver.pos_fr, force_solver.pos_bl, force_solver.pos_br]
                start_pts, end_pts, colors, sizes = [], [], [], []

                for i in range(4):
                    local_joint = pos_list[i]
                    global_joint_pos = c_pos + np.dot(rot_matrix, local_joint)
                    
                    x_shift = 0.02 if local_joint[0] > 0 else -0.02
                    local_shift = np.array([x_shift, 0.0, 0.0])
                    contact_patch = global_joint_pos + np.dot(rot_matrix, local_shift)
                    
                    world_force_vec = np.dot(flat_rot_matrix, raw_forces[i] * 0.001) 
                    end_pt = contact_patch + world_force_vec
                    
                    start_pts.append(contact_patch.tolist())
                    end_pts.append(end_pt.tolist())
                    
                    force_magnitude = raw_forces[i][1] 
                    colors.append((1.0, 0.0, 0.0, 1.0) if force_magnitude < 0 else (0.0, 0.0, 1.0, 1.0)) 
                    sizes.append(5.0) 
                
                draw.draw_lines(start_pts, end_pts, colors, sizes)

            try:
                f_thrust_graph = -net_force[1]
                t_yaw_graph = net_torque[2]
                
                telemetry_queue.put_nowait({
                    "accel": accel, "steer": steer,
                    "t_rpm_fl": throttle_fl * dynamic_max_rpm, "c_rpm_fl": motor_c_rpm,
                    "f_thrust": float(f_thrust_graph), "t_yaw": float(t_yaw_graph)
                })
            except:
                pass 

        world.step(render=True)

    simulation_app.close()

if __name__ == '__main__':
    tele_queue = multiprocessing.Queue()
    speed_limit_val = multiprocessing.Value('d', 1.0)
    
    tele_process = multiprocessing.Process(target=run_telemetry, args=(tele_queue, speed_limit_val))
    tele_process.start()

    try:
        main(tele_queue, speed_limit_val)
    finally:
        tele_process.terminate()