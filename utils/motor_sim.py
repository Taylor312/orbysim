import numpy as np
from utils.constants import OrbitronConfig

class Apex3Motor:
    def __init__(self):
        # Hardware Specs
        self.V_bus = OrbitronConfig.V_BUS_MAX
        self.Kv = OrbitronConfig.MOTOR_KV
        self.R = OrbitronConfig.MOTOR_RESISTANCE
        self.Kt = 60.0 / (2.0 * np.pi * self.Kv)
        
        # VESC Configuration
        self.current_limit = OrbitronConfig.ESC_CURRENT_LIMIT
        self.kp = OrbitronConfig.VESC_KP
        self.ki = OrbitronConfig.VESC_KI
        self.integral_error = 0.0

        # Stateful Limiter to Crush Chattering
        self.prev_allowed_current = 0.0

        # Stateful Thermal Nodes
        self.T_ambient = OrbitronConfig.T_AMBIENT
        self.T_copper = OrbitronConfig.T_AMBIENT
        self.T_bulk = OrbitronConfig.T_AMBIENT
        self.C_copper = OrbitronConfig.C_TH_COPPER
        self.C_bulk = OrbitronConfig.C_TH_BULK
        self.R_internal = OrbitronConfig.R_TH_INTERNAL
        self.R_ambient = OrbitronConfig.R_TH_AMBIENT

    def compute_torque(self, target_rpm, current_rpm, dt, battery_obj=None):
        """
        VESC RPM Controller + Slew Rate Filter + Asymmetric Regen Circuit Solver
        """
        if battery_obj is not None:
            self.V_bus = battery_obj.v_bus

        # 1. PI Velocity Controller
        error = target_rpm - current_rpm
        
        prelim_current = (self.kp * error) + (self.ki * self.integral_error)
        if abs(prelim_current) < self.current_limit:
            self.integral_error += error * dt
            
        requested_current = (self.kp * error) + (self.ki * self.integral_error)
        esc_allowed_current = np.clip(requested_current, -self.current_limit, self.current_limit)

        # THE FIX: Slew-Rate Limiter (Limits delta-I to 2000 A/s max to stop discrete oscillations)
        max_current_step = 2000.0 * dt
        esc_allowed_current = np.clip(
            esc_allowed_current, 
            self.prev_allowed_current - max_current_step, 
            self.prev_allowed_current + max_current_step
        )
        self.prev_allowed_current = esc_allowed_current

        # 2. Asymmetric Physics Circuit Solver
        v_emf = current_rpm / self.Kv
        i_max_motoring = (self.V_bus - v_emf) / self.R
        i_max_braking = (-self.V_bus - v_emf) / self.R
        
        i_lower_bound = min(i_max_motoring, i_max_braking)
        i_upper_bound = max(i_max_motoring, i_max_braking)
        
        actual_current = np.clip(esc_allowed_current, i_lower_bound, i_upper_bound)

        # 3. Electrical Power Extraction Math for Battery
        v_applied = (actual_current * self.R) + v_emf
        p_bus = v_applied * actual_current  
        bus_current = p_bus / self.V_bus if self.V_bus > 0 else 0.0

        # 4. Thermal Node Logic Update
        P_copper_loss = (actual_current ** 2) * self.R
        P_iron_loss = (OrbitronConfig.K_HYSTERESIS * abs(current_rpm)) + (OrbitronConfig.K_EDDY_CURRENT * (current_rpm ** 2))
        
        Q_internal = (self.T_copper - self.T_bulk) / self.R_internal
        Q_ambient = (self.T_bulk - self.T_ambient) / self.R_ambient
        
        dT_copper = (P_copper_loss - Q_internal) / self.C_copper
        dT_bulk = (Q_internal + P_iron_loss - Q_ambient) / self.C_bulk
        
        self.T_copper += dT_copper * dt
        self.T_bulk += dT_bulk * dt

        # 5. Mechanical Torque Translation
        torque = actual_current * self.Kt
        
        return torque, bus_current