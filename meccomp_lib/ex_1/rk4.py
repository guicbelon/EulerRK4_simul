from .base_method import *

class RK4(BaseMethod):
    def __init__(self, step:float=0.01, simulation_time:float = 10):
        super().__init__(step=step, simulation_time=simulation_time)
       
    def apply_f(self, state):
        theta_1 = state[0]
        theta_1_dot = state[1]
        theta_2 = state[3]
        theta_2_dot = state[4]
        state_kwargs ={
            'theta_1' : theta_1,
            'theta_1_dot' : theta_1_dot,
            'theta_2' : theta_2,
            'theta_2_dot' : theta_2_dot, 
        }
        
        next_theta_1 = theta_1 + self.step*theta_1_dot
        next_theta_1_dot_dot = self.system.theta_1_dot_dot(**state_kwargs)
        next_theta_1_dot = theta_1_dot + self.step*next_theta_1_dot_dot
        
        next_theta_2 = theta_2 + self.step*theta_2_dot
        next_theta_2_dot_dot = self.system.theta_2_dot_dot(**state_kwargs)
        next_theta_2_dot = theta_2_dot + self.step*next_theta_2_dot_dot
        
        next_state =[next_theta_1, next_theta_1_dot, next_theta_1_dot_dot, 
                        next_theta_2, next_theta_2_dot, next_theta_2_dot_dot]
        
        #next_state = [0 if np.isnan(x) else x for x in next_state]
        #next_state = [0 if np.isinf(x) else x for x in next_state]
        
        #next_state = [round(value,4) for value in next_state]

        return next_state
        
        
    def simul(self):
        self.system.reset()
        current_state = self.system.get_current_state()
        current_state = np.array(list(current_state.values()))
        for time_step in range(self.n_steps): 
            k1 = self.apply_f(current_state)
            k1 = np.array(k1)
            
            k2_state = current_state + (self.step/2)*k1
            k2 = self.apply_f(k2_state)
            k2 = np.array(k2)
            
            k3_state = current_state + (self.step/2)*k2
            k3 = self.apply_f(k3_state)
            k3 = np.array(k3)
            
            k4_state = current_state + (self.step)*k3
            k4 = self.apply_f(k4_state)
            
            next_state = current_state + (self.step/6)*(k1 + 2*k2 + 2*k3 + k4)
            next_state[2] = k1[2]
            next_state[5] = k1[5]
            current_state = next_state
            self.system.insert_state(list(next_state))
            
            
