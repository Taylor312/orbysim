from utils.constants import OrbitronConfig

class SimpleForceSolver:
    def __init__(self):
        self.r = OrbitronConfig.WHEEL_RADIUS_M
        self.track_width = OrbitronConfig.TRACK_WIDTH_M

    def compute_chassis_forces(self, torque_fl, torque_fr, torque_bl, torque_br):
        """ 
        Converts 4 independent wheel torques into a central push.
        """
        f_fl = torque_fl / self.r
        f_fr = torque_fr / self.r
        f_bl = torque_bl / self.r
        f_br = torque_br / self.r
        
        total_left_force = f_fl + f_bl
        total_right_force = f_fr + f_br
        
        forward_thrust = total_left_force + total_right_force
        yaw_torque = (total_right_force - total_left_force) * (self.track_width / 2.0)
        
        return forward_thrust, yaw_torque