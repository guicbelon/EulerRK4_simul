import numpy as np

class Point:
    def __init__(self, 
                 x_pos: float = None, 
                 y_pos: float = None, 
                 is_border: bool = False, 
                 is_airfoil: bool = False, 
                 U_infinity: float = 30):
        """
        Initialize a Point instance.
        
        Parameters:
        x_pos (float): The x-coordinate of the point.
        y_pos (float): The y-coordinate of the point.
        is_border (bool): Indicates if the point is on the border.
        is_airfoil (bool): Indicates if the point is part of an airfoil.
        U_infinity (float): The freestream velocity.
        """
        self.x_pos = x_pos
        self.y_pos = y_pos
        self.is_border = is_border
        self.is_airfoil = is_airfoil
        self.U_infinity = U_infinity
        self._preprocess()
    
    def _preprocess(self):
        """
        Initialize default values for various attributes of the flow.
        """
        self.streamline = 0
        self.del_streamline_del_x_value = 0
        self.del_streamline_del_y_value = 0
        self.pressure = 0
        self.absolute_velocity = 0
        self.is_point_A = False
        self.is_point_B = False
        self.is_in_airfoil_surface = False
        self.normal_direction = None
        self.g = 9.8
        self.density = 1
        self.bernoulli_result = 1E5 / (self.density * self.g)
        
    def set_streamline(self, streamline):
        """
        Set the streamline value, with a special condition if the point is part of an airfoil.
        
        Parameters:
        streamline (float): The streamline value to set.
        """
        if self.is_airfoil:
            streamline = 0
        self.streamline = streamline
        
    def get_del_streamline_del_x(self):
        """
        Get the x-component of the streamline gradient.
        
        Returns:
        float: The x-component of the streamline gradient, 0 if the point is on the border or part of an airfoil.
        """
        if self.is_border or self.is_airfoil:
            return 0
        return self.del_streamline_del_x_value
    
    def set_del_streamline_del_x(self, stream_line):
        """
        Set the x-component of the streamline gradient.
        
        Parameters:
        stream_line (float): The x-component of the streamline gradient to set.
        """
        if not self.is_airfoil and not self.is_border:
            self.del_streamline_del_x_value = stream_line
    
    def get_del_streamline_del_y(self):
        """
        Get the y-component of the streamline gradient.
        
        Returns:
        float: The y-component of the streamline gradient, U_infinity if the point is on the border, 
               0 if the point is part of an airfoil.
        """
        if self.is_border:
            return self.U_infinity
        elif self.is_airfoil:
            return 0
        return self.del_streamline_del_y_value
    
    def set_del_streamline_del_y(self, stream_line):
        """
        Set the y-component of the streamline gradient.
        
        Parameters:
        stream_line (float): The y-component of the streamline gradient to set.
        """
        if not self.is_airfoil and not self.is_border:
            self.del_streamline_del_y_value = stream_line
            
    def calculate_absolute_velocity(self):
        """
        Calculate the absolute velocity based on the streamline gradients.
        """
        del_in_x = self.get_del_streamline_del_x()
        del_in_y = self.get_del_streamline_del_y()
        self.absolute_velocity = np.sqrt(del_in_x**2 + del_in_y**2)

    def calculate_pressure(self):
        """
        Calculate the pressure at the point using Bernoulli's equation.
        """
        if self.is_airfoil:
            return
        self.pressure = self.density * self.g * (
            self.bernoulli_result - (self.absolute_velocity**2)/(2 * self.g))
