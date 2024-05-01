import numpy as np
import matplotlib.pyplot as plt
from .leg_system import *

class RK4:
    def __init__(self, step:float=0.01, simulation_time:float = 10):
        self.step = step
        self.n_steps = int(simulation_time/step)
        self.system = System(step=step)
    
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
        for time_step in range(self.n_steps): 
            current_state = self.system.get_current_state()
            current_state = np.array(list(current_state.values()))
            
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
            
            self.system.insert_state(list(next_state))
            

    def plot_scalar_results(self):        
        time_vector = np.arange(0, self.step*(self.n_steps+1), self.step)
        scalar_states = self.system.retrieve_scalar_states()
        scalar_states = np.array(scalar_states).reshape(self.n_steps+1,12)
        x_1_pos = scalar_states[:,0]
        y_1_pos = scalar_states[:,1]
        x_1_dot = scalar_states[:,2]
        y_1_dot = scalar_states[:,3]
        x_1_dot_dot = scalar_states[:,4]
        y_1_dot_dot = scalar_states[:,5]
        x_2_pos = scalar_states[:,6]
        y_2_pos = scalar_states[:,7]
        x_2_dot = scalar_states[:,8]
        y_2_dot = scalar_states[:,9]
        x_2_dot_dot = scalar_states[:,10]
        y_2_dot_dot = scalar_states[:,11]
        
        fig, axs = plt.subplots(6, 2, figsize=(10, 20))
        
        axs[0, 0].plot(time_vector, x_1_pos)
        axs[0, 0].set_title(f'Position in x of mass 1')
        axs[0, 1].plot(time_vector, y_1_pos)
        axs[0, 1].set_title(f'Position in y of mass 1')
        axs[1, 0].plot(time_vector, x_1_dot)
        axs[1, 0].set_title(f'Velocity in x of mass 1')
        axs[1, 1].plot(time_vector, y_1_dot)
        axs[1, 1].set_title(f'Velocity in y of mass 1')
        axs[2, 0].plot(time_vector, x_1_dot_dot)
        axs[2, 0].set_title(f'Aceleration in x of mass 1')
        axs[2, 1].plot(time_vector, y_1_dot_dot)
        axs[2, 1].set_title(f'Acceleration in y of mass 1')
        axs[3, 0].plot(time_vector, x_2_pos)
        axs[3, 0].set_title(f'Position in x of mass 2')
        axs[3, 1].plot(time_vector, y_2_pos)
        axs[3, 1].set_title(f'Position in y of mass 2')
        axs[4, 0].plot(time_vector, x_2_dot)
        axs[4, 0].set_title(f'Velocity in x of mass 2')
        axs[4, 1].plot(time_vector, y_2_dot)
        axs[4, 1].set_title(f'Velocity in y of mass 2')
        axs[5, 0].plot(time_vector, x_2_dot_dot)
        axs[5, 0].set_title(f'Aceleration in x of mass 2')
        axs[5, 1].plot(time_vector, y_2_dot_dot)
        axs[5, 1].set_title(f'Acceleration in y of mass 2')
        plt.tight_layout()
        plt.show()
        
    def plot_results(self):        
        time_vector = np.arange(0, self.step*(self.n_steps+1), self.step)
        scalar_states = self.system.retrieve_states()
        
        scalar_states = np.array(scalar_states).reshape(self.n_steps+1,6)
        theta_1 = scalar_states[:,0]
        theta_1_dot = scalar_states[:,1]
        theta_1_dot_dot = scalar_states[:,2]
        theta_2 = scalar_states[:,3]
        theta_2_dot = scalar_states[:,4]
        theta_2_dot_dot = scalar_states[:,5]
        
        fig, axs = plt.subplots(3, 2, figsize=(10, 10))
        
        axs[0, 0].plot(time_vector, theta_1, color='blue')
        axs[0, 0].set_title(f'Position of mass 1')
        axs[0, 1].plot(time_vector, theta_2,color='red')
        axs[0, 1].set_title(f'Position of mass 2')
        axs[1, 0].plot(time_vector, theta_1_dot, color='blue')
        axs[1, 0].set_title(f'Velocity of mass 1')
        axs[1, 1].plot(time_vector, theta_2_dot, color='red')
        axs[1, 1].set_title(f'Velocity of mass 2')
        axs[2, 0].plot(time_vector, theta_1_dot_dot, color='blue')
        axs[2, 0].set_title(f'Aceleration of mass 1')
        axs[2, 1].plot(time_vector, theta_2_dot_dot, color='red')
        axs[2, 1].set_title(f'Acceleration of mass 2')
        plt.tight_layout()
        plt.show()

        