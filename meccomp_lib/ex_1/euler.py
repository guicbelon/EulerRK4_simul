from .base_method import *

class Euler(BaseMethod):
    def __init__(self, system:System, simulation_time:float = 10):        
        super().__init__(system=system, simulation_time=simulation_time)
        
    def simul(self):
        self.system.reset()
        for time_step in range(self.n_steps):
            current_state = self.system.get_current_state()
            theta_1 = current_state['theta_1']
            theta_1_dot = current_state['theta_1_dot']
            next_theta_1 = theta_1 + self.step*theta_1_dot
            next_theta_1_dot_dot = self.system.theta_1_dot_dot(**current_state)
            next_theta_1_dot = theta_1_dot + self.step*next_theta_1_dot_dot
            
            theta_2 = current_state['theta_2']
            theta_2_dot = current_state['theta_2_dot']
            next_theta_2 = theta_2 + self.step*theta_2_dot
            next_theta_2_dot_dot = self.system.theta_2_dot_dot(**current_state)
            next_theta_2_dot = theta_2_dot + self.step*next_theta_2_dot_dot
            
            next_state =[next_theta_1, next_theta_1_dot, next_theta_1_dot_dot, 
                         next_theta_2, next_theta_2_dot, next_theta_2_dot_dot]
            self.system.insert_state(next_state)
            