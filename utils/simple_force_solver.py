import numpy as np
from utils.constants import OrbitronConfig

class SimpleForceSolver:
    def __init__(self):
        self.r = OrbitronConfig.WHEEL_RADIUS_M
        self.half_track = OrbitronConfig.TRACK_WIDTH_M / 2.0
        self.half_base = OrbitronConfig.WHEELBASE_M / 2.0
        self.z_drop = -(OrbitronConfig.CG_HEIGHT_M + self.r)

        # Contact Patch local coordinates relative to CoM [X, Y, Z]
        self.pos_fl = np.array([self.half_base, self.half_track, self.z_drop])
        self.pos_fr = np.array([self.half_base, -self.half_track, self.z_drop])
        self.pos_bl = np.array([-self.half_base, self.half_track, self.z_drop])
        self.pos_br = np.array([-self.half_base, -self.half_track, self.z_drop])

    def compute_4_corner_forces(self, torque_fl, torque_fr, torque_bl, torque_br, total_downforce_n, mu=1.0):
        """
        Calculates individual contact patch forces, evaluating grip limits PER WHEEL.
        Crucial for future Traction Control integration.
        """
        # 1. Calculate Normal Force per wheel (Assuming 50/50 static weight distribution for now)
        # In the future, we can add dynamic pitch/roll weight transfer here!
        total_z_force = (OrbitronConfig.CHASSIS_MASS_KG * 9.81) + total_downforce_n
        normal_f_per_wheel = total_z_force / 4.0
        
        # Max Grip = Mu * Normal Force
        grip_limit = mu * normal_f_per_wheel

        # 2. Convert Motor Torque to Commanded Linear Tire Force
        req_f_fl = torque_fl / self.r
        req_f_fr = torque_fr / self.r
        req_f_bl = torque_bl / self.r
        req_f_br = torque_br / self.r

        # 3. INDEPENDENT SLIP CLAMPING
        # If the motor asks for more force than the tire has grip, it slips. 
        # The chassis only receives the grip limit.
        f_fl = np.clip(req_f_fl, -grip_limit, grip_limit)
        f_fr = np.clip(req_f_fr, -grip_limit, grip_limit)
        f_bl = np.clip(req_f_bl, -grip_limit, grip_limit)
        f_br = np.clip(req_f_br, -grip_limit, grip_limit)

        # Calculate how much force was "lost" to slipping (for visual wheel speed feedback)
        slip_fl = req_f_fl - f_fl
        slip_fr = req_f_fr - f_fr
        slip_bl = req_f_bl - f_bl
        slip_br = req_f_br - f_br

        # 4. Create explicit 3D force vectors for each contact patch (Pushing in local X)
        vec_f_fl = np.array([f_fl, 0.0, 0.0])
        vec_f_fr = np.array([f_fr, 0.0, 0.0])
        vec_f_bl = np.array([f_bl, 0.0, 0.0])
        vec_f_br = np.array([f_br, 0.0, 0.0])

        # 5. Physics Engine Aggregation 
        # (Rigid bodies require 1 net force and 1 net torque at the CoM)
        net_force = vec_f_fl + vec_f_fr + vec_f_bl + vec_f_br
        net_torque = (
            np.cross(self.pos_fl, vec_f_fl) + 
            np.cross(self.pos_fr, vec_f_fr) + 
            np.cross(self.pos_bl, vec_f_bl) + 
            np.cross(self.pos_br, vec_f_br)
        )

        return net_force[0], net_torque[1], net_torque[2], (slip_fl, slip_fr, slip_bl, slip_br)