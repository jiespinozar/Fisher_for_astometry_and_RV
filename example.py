import numpy as np
import Fisher_TESS_Gaia as FTG

t_astro = np.linspace(0,5,70)                     	#Astrometry campaign (yr)
t_rv = np.linspace(5,10,15)				                  #RV campaign (yr)
t_0 = 2.5						                                #Reference time (yr)
sigma_rv = 25						                            #RV precision (m/s)
sigma_fov = 0.034230 					                      #Astrometric precision (mas)	 		

q = 0.00429997402688885				                      #Mass-ratio
a = 3.28						                                #Semi-major axis (AU)
d = 18.28						                                #Stellar distance (pc)
P = 5.638420722173639					                      #Orbital period (yr)
e = 0.642						                                #Orbital eccentricity
M_0 = np.pi/4.0					                            #Reference position (rad)
w = 30.0*np.pi/180.0					                      #Argument of periapsis (rad)
O = 80.0*np.pi/180.0					                      #Longitude of the ascending node (rad)	
cosi = np.cos(np.pi/4.0)				                    #Cosine of orbital inclination

print(FTG.get_unc_only_astrometry(t_astro, t_0, q, a, d, P, e, M_0, w, O, cosi, sigma_fov)["cosi_err"])
print(FTG.get_unc_rv(t_rv, t_astro,t_0, q, a, d, P, e, M_0, w, O, cosi, sigma_rv, sigma_fov)["cosi_err"])
