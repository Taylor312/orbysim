import numpy as np
import torch

class Apex3Motor:
    def __init__(self, voltage=24.0, kv=80.0, internal_resistance=0.05):
        self.V_bus = voltage
        self.Kv = kv
        self.R = internal_resistance
        # Max theoretical torque at 0 RPM (stall)
        self.stall_torque = (self.V_bus / self.R) / self.Kv 

    def get_available_torque(self, commanded_throttle, current_rpm):
        """
        Simulates Back-EMF. Torque drops linearly as RPM increases.
        commanded_throttle: -1.0 to 1.0
        """
        # Convert RPM to rad/s for back-emf math
        rads_per_sec = current_rpm * (np.pi / 30.0)
        back_emf = rads_per_sec / self.Kv
        
        # Effective voltage pushing through the coils
        effective_voltage = (self.V_bus * abs(commanded_throttle)) - back_emf
        
        if effective_voltage <= 0:
            return 0.0 # Motor has maxed out its speed
            
        current = effective_voltage / self.R
        torque = current / self.Kv
        
        return torque * np.sign(commanded_throttle)

class OrbitronTractionAllocator:
    def __init__(self):
        # Orbitron Physical Constants (Converted to metric for Physics Engine)
        self.mass = 113.4 # 250 lbs in kg
        self.downforce = 5782.0 # 1300 lbs in Newtons
        self.h_cg = 0.106 # 4.2 inches in meters
        self.wheelbase = 0.343 # 13.5 inches in meters
        self.track_width = 0.508 # ~20 inches in meters (estimate)
        self.wheel_radius = 0.0635 # 2.5 inches in meters
        self.mu = 0.9 # Coefficient of Friction
        
        # State Machine
        self.trench_mode = False

    def calculate_dynamic_normal_forces(self, a_x_gs, a_y_gs):
        """ The Feedforward Observer: Calculates dynamic weight transfer """
        static_weight = (self.mass * 9.81) + self.downforce
        w_per_wheel = static_weight / 4.0
        
        # Weight transfer (F = ma)
        delta_w_x = (self.mass * (a_x_gs * 9.81) * self.h_cg) / self.wheelbase
        delta_w_y = (self.mass * (a_y_gs * 9.81) * self.h_cg) / self.track_width
        
        # [FR, BR, BL, FL] matching your HAL indices
        F_N = torch.zeros(4, device="cuda:0")
        F_N[0] = max(0.0, w_per_wheel - (delta_w_x / 2) + (delta_w_y / 2)) # FR
        F_N[1] = max(0.0, w_per_wheel + (delta_w_x / 2) + (delta_w_y / 2)) # BR
        F_N[2] = max(0.0, w_per_wheel + (delta_w_x / 2) - (delta_w_y / 2)) # BL
        F_N[3] = max(0.0, w_per_wheel - (delta_w_x / 2) - (delta_w_y / 2)) # FL
        
        return F_N

    def update_physics(self, throttle_cmds, current_rpms, v_actual, a_x, a_y):
        """
        The Master Loop: Run this every physics step.
        throttle_cmds: Array of [-1 to 1] from HAL
        current_rpms: Actual joint velocities
        v_actual: True chassis speed from overhead camera/sim root
        """
        # 1. Check Shoving Match (Trench Mode)
        avg_rpm = torch.mean(torch.abs(current_rpms))
        if avg_rpm > 100.0 and abs(v_actual) < 0.5:
            self.trench_mode = True
        elif abs(v_actual) > 0.5:
            self.trench_mode = False

        # 2. Get Dynamic Grip Limits
        F_N = self.calculate_dynamic_normal_forces(a_x, a_y)
        max_grip_force = F_N * self.mu 

        # 3. Calculate Allowed Forces
        applied_forces = torch.zeros(4, device="cuda:0")
        target_joint_vels = torch.zeros(4, device="cuda:0")
        
        for i in range(4):
            # Motor Model: How much torque does the motor WANT to produce?
            motor = Apex3Motor()
            requested_torque = motor.get_available_torque(throttle_cmds[i], current_rpms[i])
            requested_force = requested_torque / self.wheel_radius
            
            if self.trench_mode:
                # Lock to continuous thermal limit to win shoving match
                allowed_force = min(requested_force, 200.0) # Approx 40A equivalent
            else:
                # Friction Ellipse Clamp
                allowed_force = torch.clamp(torch.tensor(requested_force), 
                                            -max_grip_force[i], 
                                            max_grip_force[i])
                
            applied_forces[i] = allowed_force
            
            # The Spoof: Let the joint spin freely based on motor curve, 
            # while the allowed_force is what actually moves the chassis
            target_joint_vels[i] = current_rpms[i] + (requested_torque * 0.1) # Simulate rotational inertia spool-up

        return target_joint_vels, applied_forces