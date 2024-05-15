import numpy as np

class Point:
    def __init__(self, 
                 x_pos:float=None, 
                 y_pos:float=None, 
                 is_border:bool=False, 
                 is_airfoil:bool=False, 
                 U_infinity:float=30):
        self.x_pos = x_pos
        self.y_pos = y_pos
        self.is_border = is_border
        self.is_airfoil = is_airfoil
        self.U_infinity = U_infinity
        self._preprocess()
    
    def _preprocess(self):
        self.streamline = 0
        self.del_streamline_del_x_value = 0
        self.del_streamline_del_y_value = 0
        self.pressure = 0
        self.absolute_velocity = 0
        self.g = 9.8
        self.density = 1
        self.bernoulli_result = 10E5/(self.density*self.g)
        
    def set_streamline(self, streamline):
        if self.is_airfoil:
            streamline = 0
        self.streamline = streamline
        
    def get_del_streamline_del_x(self):
        if self.is_border or self.is_airfoil:
            return 0
        return self.del_streamline_del_x_value
    
    def set_del_streamline_del_x(self, stream_line):
        if (not self.is_airfoil) and (not self.is_border):
            self.del_streamline_del_x_value = stream_line
    
    def get_del_streamline_del_y(self):
        if self.is_border:
            return self.U_infinity
        elif self.is_airfoil:
            return 0
        return self.del_streamline_del_y_value
    
    def set_del_streamline_del_y(self, stream_line):
        if (not self.is_airfoil) and (not self.is_border):
            self.del_streamline_del_y_value = stream_line
            
    def calculate_absolute_velocity(self):
        del_in_x = self.get_del_streamline_del_x()
        del_in_y = self.get_del_streamline_del_y()
        self.absolute_velocity = np.sqrt(del_in_x**2 + del_in_y**2)

    def calculate_pressure(self):
        if self.is_airfoil:
            return
        self.pressure = self.density*self.g*(
            self.bernoulli_result - (self.absolute_velocity**2)/2*self.g)
        
