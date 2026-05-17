import numpy as np
from utils.constants import OrbitronConfig

class SimpleForceSolver:
    def __init__(self):
        self.r = OrbitronConfig.WHEEL_RADIUS_M
        
        # Exact Local Coordinates [X, Y, Z] from your CAD base_link
        # Coordinate System: X = Lateral (Left/Right), Y = Forward/Backward, Z = Up/Down
        self.pos_fl = np.array([0.09, -0.04481, -0.000205 - self.r])   
        self.pos_fr = np.array([-0.09, -0.04481, -0.000205 - self.r])  
        self.pos_bl = np.array([0.09, 0.04481, -0.000205 - self.r])    
        self.pos_br = np.array([-0.09, 0.04481, -0.000205 - self.r])   

    def compute_4_corner_forces(self, torque_fl, torque_fr, torque_bl, torque_br, total_downforce_n, mu=1.0):
        """
        Calculates independent longitudinal motor propulsion forces clamped by traction limits.
        """
        # 1. Total vertical force tracking
        normal_f_per_wheel = total_downforce_n / 4.0
        grip_limit = mu * normal_f_per_wheel

        # 2. Convert axle torques to longitudinal forces
        req_f_fl = torque_fl / self.r
        req_f_fr = torque_fr / self.r
        req_f_bl = torque_bl / self.r
        req_f_br = torque_br / self.r

        # 3. Clamp by individual traction ceilings
        f_fl = np.clip(req_f_fl, -grip_limit, grip_limit)
        f_fr = np.clip(req_f_fr, -grip_limit, grip_limit)
        f_bl = np.clip(req_f_bl, -grip_limit, grip_limit)
        f_br = np.clip(req_f_br, -grip_limit, grip_limit)

        slips = (req_f_fl - f_fl, req_f_fr - f_fr, req_f_bl - f_bl, req_f_br - f_br)

        # 4. Construct local vectors (Forward in your CAD is Negative Y)
        vec_f_fl = np.array([0.0, -f_fl, 0.0])
        vec_f_fr = np.array([0.0, -f_fr, 0.0])
        vec_f_bl = np.array([0.0, -f_bl, 0.0])
        vec_f_br = np.array([0.0, -f_br, 0.0])

        # 5. Sum up propulsion force and wheelie moments
        net_force = vec_f_fl + vec_f_fr + vec_f_bl + vec_f_br
        net_torque = (
            np.cross(self.pos_fl, vec_f_fl) + 
            np.cross(self.pos_fr, vec_f_fr) + 
            np.cross(self.pos_bl, vec_f_bl) + 
            np.cross(self.pos_br, vec_f_br)
        )

        return net_force, net_torque, slips, (vec_f_fl, vec_f_fr, vec_f_bl, vec_f_br)