from .base_method import *

class RK4(BaseMethod):    
    def __init__(self, system: System, simulation_time: float = 10):
        """
        Initialize the RK4 method with the given system and simulation time.

        Parameters:
        system (System): The system to be simulated.
        simulation_time (float): Total time for simulation. Default is 10.
        """
        super().__init__(system=system, simulation_time=simulation_time)
       
    def apply_f(self, state):
        """
        Compute the derivatives of the state variables.

        Parameters:
        state (list): A list containing the current state variables [theta_1, theta_1_dot, theta_2, theta_2_dot].

        Returns:
        list: The derivatives [theta_1_dot, theta_1_dot_dot, theta_2_dot, theta_2_dot_dot].
        """
        theta_1 = state[0]
        theta_1_dot = state[1]
        theta_2 = state[2]
        theta_2_dot = state[3]
        state_kwargs = {
            'theta_1': theta_1,
            'theta_1_dot': theta_1_dot,
            'theta_2': theta_2,
            'theta_2_dot': theta_2_dot, 
        }

        theta_1_dot_dot = self.system.theta_1_dot_dot(**state_kwargs)
        theta_2_dot_dot = self.system.theta_2_dot_dot(**state_kwargs)
        
        result = [theta_1_dot, theta_1_dot_dot, theta_2_dot, theta_2_dot_dot]
        return result 

    def simul(self):
        """
        Run the simulation using the RK4 method.

        This method iterates over the number of time steps, updating the system's state 
        at each step using the RK4 integration scheme.
        """
        self.system.reset()
        current_state = self.system.get_current_state()
        current_state = np.array([current_state['theta_1'],
                                  current_state['theta_1_dot'],
                                  current_state['theta_2'], 
                                  current_state['theta_2_dot']])
        
        for time_step in range(self.n_steps): 
            k1 = self.apply_f(current_state)
            k1 = np.array(k1)
            
            k2_state = current_state + (self.step / 2) * k1
            k2 = self.apply_f(k2_state)
            k2 = np.array(k2)
            
            k3_state = current_state + (self.step / 2) * k2
            k3 = self.apply_f(k3_state)
            k3 = np.array(k3)
            
            k4_state = current_state + self.step * k3
            k4 = self.apply_f(k4_state)
            
            next_state = current_state + (self.step / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
            
            next_theta_1 = next_state[0]
            next_theta_1_dot = next_state[1]
            next_theta_1_dot_dot = k1[1]
            next_theta_2 = next_state[2]
            next_theta_2_dot = next_state[3]
            next_theta_2_dot_dot = k1[3]
            
            system_state = [next_theta_1, 
                            next_theta_1_dot, 
                            next_theta_1_dot_dot, 
                            next_theta_2, 
                            next_theta_2_dot, 
                            next_theta_2_dot_dot]
            
            current_state = next_state
            self.system.insert_state(system_state)
