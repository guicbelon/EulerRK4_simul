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
        self.streamline = 0
        
    def set_streamline(self, streamline):
        if self.is_airfoil:
            streamline = 0
        self.streamline = streamline