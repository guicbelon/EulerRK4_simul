import numpy as np

class System:
    def __init__(self,
        m1:float=0.5,
        m2:float=1.0,
        l1:float=1.0,
        l2:float=1.5,
        theta_1_initial:float=0,
        theta_2_initial:float=0,
        theta_1_dot_initial:float=0.05,
        theta_2_dot_initial:float=0.1,
        theta_1_dot_dot_initial:float=np.nan,
        theta_2_dot_dot_initial:float=np.nan,
        step:float=0.1):
        self.m1 = m1
        self.m2 = m2
        self.l1 = l1
        self.l2 = l2
        self.theta_1_initial = theta_1_initial
        self.theta_2_initial = theta_2_initial
        self.theta_1_dot_initial = theta_1_dot_initial
        self.theta_2_dot_initial = theta_2_dot_initial
        self.theta_1_dot_dot_initial = theta_1_dot_dot_initial
        self.theta_2_dot_dot_initial = theta_2_dot_dot_initial
        self.step = step
        self.g = 9.8
        self.reset()
    
    def reset(self):
        self.current_step = 0
        self.state_memory = [[self.theta_1_initial, self.theta_1_dot_initial, self.theta_1_dot_dot_initial, 
                              self.theta_2_initial, self.theta_2_dot_initial, self.theta_2_dot_dot_initial]]
        
    
    def theta_1_dot_dot(self, **kwargs):
        theta_1 = kwargs.get('theta_1')
        theta_1_dot = kwargs.get('theta_1_dot')
        theta_2 = kwargs.get('theta_2')
        theta_2_dot = kwargs.get('theta_2_dot')
        
        num = (-np.sin(theta_1-theta_2)*(self.m2*self.l2*(theta_2_dot**2) 
                +self.m2*self.l1*(theta_1_dot**2)*np.cos(theta_1-theta_2)) 
            -self.g*((self.m1 +self.m2)*np.sin(theta_1) - self.m2*(np.sin(theta_2)*np.cos(theta_1-theta_2))))
        den = self.l1*(self.m1+self.m2*((np.sin(theta_1-theta_2))**2))
        return num/den
    
    def theta_2_dot_dot(self, **kwargs):
        theta_1 = kwargs.get('theta_1')
        theta_1_dot = kwargs.get('theta_1_dot')
        theta_2 = kwargs.get('theta_2')
        theta_2_dot = kwargs.get('theta_2_dot')
        num = (np.sin(theta_1-theta_2)*((self.m1+self.m2)*self.l1*(theta_1_dot**2) 
                +self.m2*self.l2*(theta_2_dot**2)*np.cos(theta_1-theta_2)) 
            +self.g*((self.m1 +self.m2)*(np.sin(theta_1)*np.cos(theta_1-theta_2)-np.sin(theta_2))))
        den = self.l2*(self.m1+self.m2*((np.sin(theta_1-theta_2))**2))
        return num/den
    
    def get_current_state(self):
        current_state = self.state_memory[self.current_step]
        state = {
            'theta_1': current_state[0],
            'theta_1_dot': current_state[1],
            'theta_1_dot_dot': current_state[2],
            'theta_2': current_state[3],
            'theta_2_dot': current_state[4],
            'theta_2_dot_dot': current_state[5]
        }
        return state
        
    def insert_state(self, state):
        self.state_memory.append(state)
        self.current_step += 1    
    
    
    def retrieve_states(self):
        return self.state_memory
    
    def retrieve_scalar_states(self):
        scalar_states = []
        for state in self.state_memory:
            theta_1 = state[0]
            theta_1_dot = state[1]
            theta_1_dot_dot = state[2]
            theta_2 = state[3]
            theta_2_dot = state[4]
            theta_2_dot_dot = state[5]            
            x_1 = self.l1*np.sin(theta_1)
            x_1_dot = self.l1*np.cos(theta_1)*theta_1_dot
            x_1_dot_dot = self.l1*np.cos(theta_1)*theta_1_dot_dot - self.l1*np.sin(theta_1)*(theta_1_dot**2)
            y_1 = self.l1*np.cos(theta_1)
            y_1_dot = -self.l1*np.sin(theta_1)*theta_1_dot
            y_1_dot_dot = -self.l1*np.cos(theta_1)*(theta_1_dot**2) - self.l1*np.sin(theta_1)*theta_1_dot_dot    
             
            x_2 = self.l2*np.sin(theta_2) + x_1
            x_2_dot = self.l2*np.cos(theta_2)*theta_2_dot + x_1_dot
            x_2_dot_dot = self.l2*np.cos(theta_2)*theta_2_dot_dot - self.l2*np.sin(theta_2)*(theta_2_dot**2)  + x_1_dot_dot
            y_2 = self.l2*np.cos(theta_2) + y_1
            y_2_dot = -self.l2*np.sin(theta_2)*theta_2_dot + y_1_dot
            y_2_dot_dot = -self.l2*np.cos(theta_2)*(theta_2_dot**2) - self.l2*np.sin(theta_2)*theta_2_dot_dot + y_1_dot_dot
            scalar_state = [[x_1, y_1],[x_1_dot, y_1_dot], [x_1_dot_dot, y_1_dot_dot],
                            [x_2, y_2],[x_2_dot, y_2_dot], [x_2_dot_dot, y_2_dot_dot]]
            scalar_states.append(scalar_state)
        return scalar_states
    