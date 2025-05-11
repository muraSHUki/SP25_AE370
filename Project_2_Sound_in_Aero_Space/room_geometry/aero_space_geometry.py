###### IMPORTS ######################################################################################
import numpy as np                                                                                  #
from shapely.geometry import Point, Polygon                                                         #
#####################################################################################################



### Room Geometry ###################################################################################
def get_room_polygon():                                                                             
    """                                                                                             
    Returns a Shapely Polygon representing the outer boundary of the room.                          
    The room has an irregular shape with a cutout and non-rectangular structure.                    
    """                                                                                             
    return Polygon([                                                                                 
        (1, 0), (15, 0), (15, 1), (14.5, 2), (14.5, 3),                                               
        (15, 4), (15, 5), (0, 5), (0, 3), (1, 3)                                                      
    ])                                                                                               
#####################################################################################################



### Pillar Configuration ############################################################################
def get_pillars():                                                                                  
    """                                                                                             
    Returns a list of pillars, where each pillar is represented as a tuple:                         
    (center_x, center_y, side_length).                                                              
    These define square obstacles to be excluded from the wave propagation domain.                  
    """                                                                                             
    return [                                                                                         
        (3, 2.5, 0.3), (8, 2.5, 0.3), (12, 1, 0.3), (12, 4, 0.3),                                     
        (3, 4.8, 0.3), (6, 4.8, 0.3), (9, 4.8, 0.3), (12, 4.8, 0.3),                                  
        (14.8, 4.8, 0.3), (14.8, 0.2, 0.3), (7.5, 0, 0.4), (2, 0.2, 0.3)                              
    ]                                                                                                
#####################################################################################################



### Domain Mask Generation ##########################################################################
def generate_domain_mask_fast(X, Y):                                                                
    """                                                                                             
    Generates a boolean mask array matching the shape of X and Y coordinate grids.                  
    Returns True for grid points inside the room and not inside any pillar.                         
    """                                                                                             
    room = get_room_polygon()                                                                        
    mask = np.full(X.shape, False, dtype=bool)                                                       

    for i in range(X.shape[0]):                                                                      
        for j in range(X.shape[1]):                                                                  
            pt = Point(X[i, j], Y[i, j])                                                              
            if room.contains(pt):                                                                    
                mask[i, j] = True                                                                     

    for (px, py, s) in get_pillars():                                                                
        in_x = (X >= px - s / 2) & (X <= px + s / 2)                                                 
        in_y = (Y >= py - s / 2) & (Y <= py + s / 2)                                                 
        mask[in_x & in_y] = False                                                                    

    return mask                                                                                      
#####################################################################################################



### Plotting Utilities ##############################################################################
def plot_room_and_pillars(ax):                                                                      
    """                                                                                             
    Adds outlines of the room and pillars to a matplotlib Axes object.                              
    Used for plotting wave fields over the room geometry.                                           
    """                                                                                             
    room = get_room_polygon()                                                                        
    x, y = room.exterior.xy                                                                          
    ax.plot(x, y, color='red', linewidth=2)  # Room boundary                                         

    for (px, py, s) in get_pillars():                                                                
        half = s / 2                                                                                 
        rect_x = [px - half, px + half, px + half, px - half, px - half]                             
        rect_y = [py - half, py - half, py + half, py + half, py - half]                             
        ax.plot(rect_x, rect_y, color='red', linewidth=2)  # Pillar outline                          
#####################################################################################################
