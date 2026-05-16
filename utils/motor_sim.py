import numpy as np

class Apex3Motor:
    def __init__(self, voltage=24.0, kv=80.0, internal_resistance=0.05):
        self.V_bus = voltage
        self.Kv = kv
        self.R = internal_resistance
        self.Kt = 60.0 / (2.0 * np.pi * self.Kv)

    def compute_torque(self, throttle_cmd, current_rpm):
        """ 
        throttle_cmd: -1.0 to 1.0 
        """
        # Hard deadzone to absolutely kill phantom idle jitter
        if abs(throttle_cmd) < 0.01:
            return 0.0

        rads_per_sec = current_rpm * (np.pi / 30.0)
        back_emf = abs(rads_per_sec / self.Kv)
        
        # Effective voltage pushing through coils
        effective_voltage = (self.V_bus * abs(throttle_cmd)) - back_emf
        
        if effective_voltage <= 0:
            return 0.0 # Motor has hit its top physical speed for this throttle
            
        current = effective_voltage / self.R
        torque = current * self.Kt
        
        return torque * np.sign(throttle_cmd)