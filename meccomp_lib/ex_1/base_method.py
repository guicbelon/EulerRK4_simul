import numpy as np
import matplotlib.pyplot as plt
from .leg_system import *

class BaseMethod:
    def __init__(self, system:System, simulation_time:float = 10):        
        self.system = system
        self.step = self.system.step
        self.n_steps = int(simulation_time/self.step)
        
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
        axs[0, 0].set_title(f'Posição da massa 1 - Discretização: {self.system.step}')
        axs[0, 1].plot(time_vector, theta_2,color='red')
        axs[0, 1].set_title(f'Posição da massa 2 - Discretização: {self.system.step}')
        axs[1, 0].plot(time_vector, theta_1_dot, color='blue')
        axs[1, 0].set_title(f'Velocidade da massa 1 - Discretização: {self.system.step}')
        axs[1, 1].plot(time_vector, theta_2_dot, color='red')
        axs[1, 1].set_title(f'Velocidade da massa 2 - Discretização: {self.system.step}')
        axs[2, 0].plot(time_vector, theta_1_dot_dot, color='blue')
        axs[2, 0].set_title(f'Aceleração da massa 1 - Discretização: {self.system.step}')
        axs[2, 1].plot(time_vector, theta_2_dot_dot, color='red')
        axs[2, 1].set_title(f'Aceleração da massa 2 - Discretização: {self.system.step}')
        plt.tight_layout()
        plt.show()