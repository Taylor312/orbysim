import numpy as np
from utils.constants import OrbitronConfig

class Apex3Motor:
    def __init__(self):
        self.V_bus = OrbitronConfig.V_BUS_MAX
        self.Kv = OrbitronConfig.MOTOR_KV
        self.R = OrbitronConfig.MOTOR_RESISTANCE
        self.Kt = 60.0 / (2.0 * np.pi * self.Kv)
        
        self.current_limit = OrbitronConfig.ESC_CURRENT_LIMIT
        self.kp = OrbitronConfig.VESC_KP
        self.ki = OrbitronConfig.VESC_KI
        self.integral_error = 0.0
        self.prev_allowed_current = 0.0

        self.T_ambient = OrbitronConfig.T_AMBIENT
        self.T_copper = OrbitronConfig.T_AMBIENT
        self.T_bulk = OrbitronConfig.T_AMBIENT
        self.C_copper = OrbitronConfig.C_TH_COPPER
        self.C_bulk = OrbitronConfig.C_TH_BULK
        self.R_internal = OrbitronConfig.R_TH_INTERNAL
        self.R_ambient = OrbitronConfig.R_TH_AMBIENT

    def compute_torque(self, target_rpm, current_rpm, dt, battery_obj=None):
        """
        VESC RPM Controller + Implicit Asymmetric Circuit Solver + Thermal Model
        """
        # 1. PI Velocity Controller
        error = target_rpm - current_rpm
        
        prelim_current = (self.kp * error) + (self.ki * self.integral_error)
        if abs(prelim_current) < self.current_limit:
            self.integral_error += error * dt
            
        requested_current = (self.kp * error) + (self.ki * self.integral_error)
        esc_allowed_current = np.clip(requested_current, -self.current_limit, self.current_limit)

        max_current_step = 5000.0 * dt
        esc_allowed_current = np.clip(
            esc_allowed_current, 
            self.prev_allowed_current - max_current_step, 
            self.prev_allowed_current + max_current_step
        )
        self.prev_allowed_current = esc_allowed_current

        # 2. IMPLICIT ASYMMETRIC CIRCUIT SOLVER (The Fix!)
        v_emf = current_rpm / self.Kv
        
        if battery_obj is not None:
            # Combine resistances to kill the discrete feedback loop
            R_total = self.R + battery_obj.R_internal
            V_source = battery_obj.v_oc
        else:
            R_total = self.R
            V_source = self.V_bus

        i_max_motoring = (V_source - v_emf) / R_total
        i_max_braking = (-V_source - v_emf) / R_total
        
        i_lower_bound = min(i_max_motoring, i_max_braking)
        i_upper_bound = max(i_max_motoring, i_max_braking)
        
        actual_current = np.clip(esc_allowed_current, i_lower_bound, i_upper_bound)

        # 3. Electrical Power to Battery
        v_applied = (actual_current * self.R) + v_emf
        
        if battery_obj is not None:
            # Calculate true terminal voltage for accurate power extraction
            V_terminal = battery_obj.v_oc - (actual_current * battery_obj.R_internal)
        else:
            V_terminal = self.V_bus
            
        p_bus = v_applied * actual_current  
        bus_current = p_bus / V_terminal if V_terminal > 0 else 0.0

        # 4. Thermal Update
        P_copper_loss = (actual_current ** 2) * self.R
        P_iron_loss = (OrbitronConfig.K_HYSTERESIS * abs(current_rpm)) + (OrbitronConfig.K_EDDY_CURRENT * (current_rpm ** 2))
        
        Q_internal = (self.T_copper - self.T_bulk) / self.R_internal
        Q_ambient = (self.T_bulk - self.T_ambient) / self.R_ambient
        
        dT_copper = (P_copper_loss - Q_internal) / self.C_copper
        dT_bulk = (Q_internal + P_iron_loss - Q_ambient) / self.C_bulk
        
        self.T_copper += dT_copper * dt
        self.T_bulk += dT_bulk * dt

        torque = actual_current * self.Kt
        
        return torque, bus_current