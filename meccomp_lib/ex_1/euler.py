from .base_method import *

class Euler(BaseMethod):    
    def __init__(self, system: System, simulation_time: float = 10):
        """
        Initialize the Euler method with the given system and simulation time.
        
        Parameters:
        system (System): The system to be simulated.
        simulation_time (float): Total time for simulation. Default is 10.
        """
        super().__init__(system=system, simulation_time=simulation_time)
        
    def simul(self):
        """
        Run the simulation using the Euler method.
        
        This method iterates over the number of time steps, updating the system's state 
        at each step based on the current state and the system's dynamics.
        """
        self.system.reset()
        for time_step in range(self.n_steps):
            current_state = self.system.get_current_state()
            
            # Calculate next state for theta_1
            theta_1 = current_state['theta_1']
            theta_1_dot = current_state['theta_1_dot']
            next_theta_1 = theta_1 + self.step * theta_1_dot
            next_theta_1_dot_dot = self.system.theta_1_dot_dot(**current_state)
            next_theta_1_dot = theta_1_dot + self.step * next_theta_1_dot_dot
            
            # Calculate next state for theta_2
            theta_2 = current_state['theta_2']
            theta_2_dot = current_state['theta_2_dot']
            next_theta_2 = theta_2 + self.step * theta_2_dot
            next_theta_2_dot_dot = self.system.theta_2_dot_dot(**current_state)
            next_theta_2_dot = theta_2_dot + self.step * next_theta_2_dot_dot
            
            # Create next state and insert it into the system
            next_state = [next_theta_1, next_theta_1_dot, next_theta_1_dot_dot, 
                          next_theta_2, next_theta_2_dot, next_theta_2_dot_dot]
            self.system.insert_state(next_state)
