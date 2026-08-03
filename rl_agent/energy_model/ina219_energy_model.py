class INA219EnergyModel:
    """
    Simulated INA219 Voltage/Current Sensor Model for Energy-Aware Navigation.
    Formula:
    P = V * I (Watts)
    Energy = P * dt (Joules)
    """
    def __init__(self, voltage_nom=12.0):
        # Battery Nominal Voltage (12.0 V for 3S LiPo)
        self.voltage = voltage_nom
        
        # Current Profiles in Amperes (A)
        self.I_idle = 0.2       # Baseline electronics / Jetson idle current
        self.I_move_flat = 1.0  # Normal movement current
        self.I_move_rough = 2.5 # High friction / rough terrain movement current

    def get_energy_consumption(self, action_type="flat", step_duration_sec=1.0):
        """
        Calculates energy spent in Joules for a given step.
        
        Parameters:
            action_type (str): "idle", "flat", or "rough"
            step_duration_sec (float): Time taken for one step in seconds (default = 1.0s)
            
        Returns:
            energy_joules (float): Total energy consumed in Joules.
        """
        if action_type == "idle":
            current = self.I_idle
        elif action_type == "rough":
            current = self.I_move_rough
        else:
            current = self.I_move_flat

        # 1. Total Power P = V * I
        power_watts = self.voltage * current
        
        # 2. Energy E = P * dt (Joules)
        energy_joules = power_watts * step_duration_sec
        
        return energy_joules