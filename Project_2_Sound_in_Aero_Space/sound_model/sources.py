###### IMPORTS ######################################################################################
import numpy as np                                                                                  #
#####################################################################################################



### Gaussian Pulse Source ###########################################################################
def gaussian_pulse(X, Y, x0, y0, sigma):                                                            
    """                                                                                             
    Returns a 2D Gaussian pulse centered at (x0, y0) with spread sigma.                             
    Used for initializing pressure fields (e.g., single wave pulses).                               
                                                                                                    
    Parameters:                                                                                     
        X, Y   : 2D meshgrid arrays (shape: Nx x Ny)                                                 
        x0, y0 : center coordinates of the pulse                                                    
        sigma  : standard deviation controlling pulse width                                         
                                                                                                    
    Returns:                                                                                        
        p0 : 2D array of the initial pressure field                                                 
    """                                                                                             
    return np.exp(-((X - x0)**2 + (Y - y0)**2) / (2 * sigma**2))                                     
#####################################################################################################



### Speech Burst Source #############################################################################
def speech_burst(t, burst_list, amplitude=0.05):                                                    
    """                                                                                             
    Returns the time-dependent signal for a synthetic speech-like burst.                            
                                                                                                    
    Parameters:                                                                                     
        t          : float, current time                                                             
        burst_list : list of dicts with keys {"t0", "sigma", "f"}                                    
                      representing timing, width, and frequency of each burst                        
        amplitude  : scaling factor for pressure amplitude                                           
                                                                                                    
    Returns:                                                                                        
        float : summed pressure from all bursts at time t                                           
    """                                                                                             
    return sum(                                                                                      
        amplitude *                                                                                  
        np.exp(-((t - b["t0"])**2) / (2 * b["sigma"]**2)) *                                           
        np.sin(2 * np.pi * b["f"] * t)                                                                
        for b in burst_list                                                                           
    )                                                                                                
#####################################################################################################
