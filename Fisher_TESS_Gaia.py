import numpy as np
from astropy import units as u

def Eccentric_anomally(t, t_0, M_0, P, e):
    mm = 2.0*np.pi/P
    M = M_0 + mm*(t-t_0)
    M = M%(2*np.pi)
    E = M+np.sign(np.sin(M))*0.85*e
    mtest = E-e*np.sin(E)
    while np.abs(M-mtest)>1e-10:
        dE = 1.-e*np.cos(E)
        E += (M-mtest)/dE
        mtest = E-e*np.sin(E)
    return E

def rho(q,a,d):
    return q*1e3*a/d  #mas

def K(r,d,cosi,P,e,E):
    return 2*np.pi*r*d*np.sqrt(1.0-cosi**2.0)/P/(1.0-e*np.cos(E))*1e-3 # AU/yr

# Derivatives of the eccentric anomaly

def dEdP(t,t_0,P,e,E):
    N = -2*np.pi*(t-t_0)
    D = (P**2)*(1-e*np.cos(E))
    return N/D

def dEde(e,E):
    N = np.sin(E)
    D = 1-e*np.cos(E)
    return N/D

def dEdM0(e,E):
    N = 1.0
    D = 1-e*np.cos(E)
    return N/D

# Derivatives of the radial velocity

def dRVdmx(t):
    return np.zeros(len(t))

def dRVdmy(t):
    return np.zeros(len(t))

def dRVdmz(t):
    return np.zeros(len(t))+1

def dRVdrho(r,d,cosi,P,E,e,w):
    return K(r,d,cosi,P,e,E)*(np.sqrt(1.0-e**2.0)*np.cos(w)*np.cos(E)-np.sin(w)*np.sin(E))/r

def dRVdP(r,d,cosi,P,E,e,w,t,t_0):
    return K(r,d,cosi,P,e,E)/P/(1.0-e*np.cos(E))*((e*np.cos(E)-1.0)*(np.sqrt(1.0-e**2.0)*np.cos(w)*np.cos(E)-np.sin(w)*np.sin(E))+P*((e-np.cos(E))*np.sin(w)-np.sqrt(1.0-e**2.0)*np.cos(w)*np.sin(E))*dEdP(t,t_0,P,e,E))
    
def dRVde(r,d,cosi,P,E,e,w):
    return K(r,d,cosi,P,e,E)/2.0/np.sqrt(1.0-e**2.0)/(1.0-e*np.cos(E))*(2.0*np.cos(w)*np.cos(E)*(np.cos(E)-e)-np.sqrt(1-e**2.0)*np.sin(w)*np.sin(2.0*E)+2.0*(np.sqrt(1-e**2.0)*(e-np.cos(E))*np.sin(w)+(e**2.0 -1.0)*np.cos(w)*np.sin(E))*dEde(e,E))

def dRVdM0(r,d,cosi,P,E,e,w):
    return K(r,d,cosi,P,e,E)/(1.0-e*np.cos(E))*dEdM0(e,E)*((e-np.cos(E))*np.sin(w)-np.sqrt(1-e**2.0)*np.cos(w)*np.sin(E))

def dRVdw(r,d,cosi,P,E,e,w):
    return -K(r,d,cosi,P,e,E)*(np.sqrt(1.0-e**2.0)*np.cos(E)*np.sin(w)+np.cos(w)*np.sin(E))

def dRVdcosi(r,d,cosi,P,E,e,w):
    return -K(r,d,cosi,P,e,E)*cosi*(np.sqrt(1.0-e**2.0)*np.cos(w)*np.cos(E)-np.sin(w)*np.sin(E))/(1.0-cosi**2.0)

def dRVdOmega(t):
    return np.zeros(len(t))

# Fisher matrix for RV

def Fisher_for_RV(t, t_0, q, a, d, P, e, M_0, w, O, cosi, sigma):
    E = []
    for i in range(len(t)):
        x = Eccentric_anomally(t[i], t_0, M_0, P, e)
        E.append(x)
    E = np.array(E)
    r = rho(q,a,d)

    dRV = np.array([dRVdmx(t),dRVdmy(t),dRVdmz(t),dRVdrho(r,d,cosi,P,E,e,w),dRVdP(r,d,cosi,P,E,e,w,t,t_0),dRVde(r,d,cosi,P,E,e,w),dRVdM0(r,d,cosi,P,E,e,w),dRVdw(r,d,cosi,P,E,e,w),dRVdcosi(r,d,cosi,P,E,e,w),dRVdOmega(t)])
    
    matrix = np.zeros((10,10))
    C1 = 1.0/(sigma**2.0)
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            matrix[i][j] = C1*np.nansum(dRV[i]*dRV[j])
    return matrix

# Derivatives for the astrometric motion \alpha_x

def ThieleInnes(q,a,d,w,O,cosi):
    A = rho(q,a,d)*(np.cos(w)*np.cos(O) - np.sin(w)*np.sin(O)*cosi)
    B = rho(q,a,d)*(np.cos(w)*np.sin(O) + np.sin(w)*np.cos(O)*cosi)
    F =-rho(q,a,d)*(np.sin(w)*np.cos(O) + np.cos(w)*np.sin(O)*cosi)
    G =-rho(q,a,d)*(np.sin(w)*np.sin(O) - np.cos(w)*np.cos(O)*cosi)
    return A,B,F,G

def dxdmx(t,t_0):
    return t-t_0

def dxdmy(t):
    return np.zeros(len(t))

def dxdmz(t):
    return np.zeros(len(t))

def dxdrho(r,E,e,A,F):
    return (A/r)*(np.cos(E)-e)+(F/r)*np.sqrt(1-e**2)*np.sin(E)

def dxdP(t,t_0,P,e,E,A,F):
    C1 = dEdP(t,t_0,P,e,E)
    return C1*(-A*np.sin(E)+F*np.sqrt(1-e**2)*np.cos(E))

def dxde(e,E,A,F):
    C1 = dEde(e,E)
    C2 = np.sqrt(1-e**2)
    return A*(-C1*np.sin(E)-1)+F*(C2*np.cos(E)*C1-(e/C2)*np.sin(E))

def dxdM0(e,E,A,F):
    C1 = dEdM0(e,E)
    return C1*(-A*np.sin(E)+F*np.sqrt(1-e**2)*np.cos(E))

def dxdw(e,E,A,F):
    return F*(np.cos(E)-e)-A*np.sqrt(1-e**2)*np.sin(E)

def dxdcosi(r,w,O,e,E):
    C1 = -r*np.sin(w)*np.sin(O)
    C2 = -r*np.cos(w)*np.sin(O)
    return C1*(np.cos(E)-e)+C2*np.sqrt(1-e**2)*np.sin(E)

def dxdO(e,E,B,G):
    return -B*(np.cos(E)-e)-G*np.sqrt(1-e**2)*np.sin(E)

# For \alpha_y

def dydmx(t):
    return np.zeros(len(t))

def dydmy(t,t_0):
    return  t-t_0

def dydmz(t):
    return np.zeros(len(t))

def dydcosi(r,w,O,e,E):
    C1 = r*np.sin(w)*np.cos(O)
    C2 = r*np.cos(w)*np.cos(O)
    return C1*(np.cos(E)-e)+C2*np.sqrt(1-e**2)*np.sin(E)

def dydO(e,E,A,F):
    return A*(np.cos(E)-e)+F*np.sqrt(1-e**2)*np.sin(E)

# Fisher matrix for astrometry

def Fisher_for_astrometry(t, t_0, q, a, d, P, e, M_0, w, O, cosi,s_fov):
    E = []
    for i in range(len(t)):
        x = Eccentric_anomally(t[i], t_0, M_0, P, e)
        E.append(x)
    E = np.array(E)
    #print(E)
    r = rho(q,a,d)
    a,b,f,g = ThieleInnes(q,a,d,w,O,cosi)

    vx = np.array([dxdmx(t,t_0),dxdmy(t),dxdmz(t),dxdrho(r,E,e,a,f),dxdP(t,t_0,P,e,E,a,f),dxde(e,E,a,f),dxdM0(e,E,a,f),dxdw(e,E,a,f),dxdcosi(r,w,O,e,E),dxdO(e,E,b,g)])
    vy = np.array([dydmx(t),dydmy(t,t_0),dydmz(t),dxdrho(r,E,e,b,g),dxdP(t,t_0,P,e,E,b,g),dxde(e,E,b,g),dxdM0(e,E,b,g),dxdw(e,E,b,g),dydcosi(r,w,O,e,E),dydO(e,E,a,f)])

    matrix = np.zeros((10,10))
    C1 = 0.5/(s_fov**2.0)
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            matrix[i][j] = C1*np.nansum(vx[i]*vx[j]+vy[i]*vy[j])
    return matrix

# For only astrometry (9 params)

def Fisher_only_astrometry(t, t_0, q, a, d, P, e, M_0, w, O, cosi,s_fov):
    E = []
    for i in range(len(t)):
        x = Eccentric_anomally(t[i], t_0, M_0, P, e)
        E.append(x)
    E = np.array(E)
    
    r = rho(q,a,d)
    a,b,f,g = ThieleInnes(q,a,d,w,O,cosi)

    vx = np.array([dxdmx(t,t_0),dxdmy(t),dxdrho(r,E,e,a,f),dxdP(t,t_0,P,e,E,a,f),dxde(e,E,a,f),dxdM0(e,E,a,f),dxdw(e,E,a,f),dxdcosi(r,w,O,e,E),dxdO(e,E,b,g)])
    vy = np.array([dydmx(t),dydmy(t,t_0),dxdrho(r,E,e,b,g),dxdP(t,t_0,P,e,E,b,g),dxde(e,E,b,g),dxdM0(e,E,b,g),dxdw(e,E,b,g),dydcosi(r,w,O,e,E),dydO(e,E,a,f)])

    matrix = np.zeros((9,9))
    C1 = 0.5/(s_fov**2)
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            matrix[i][j] = C1*np.sum(vx[i]*vx[j]+vy[i]*vy[j])
    return matrix
    
# To obtain the uncertainties

def Cov(F):
    return np.linalg.inv(F)

#To obtain the uncertainties using astrometry and RV

def get_unc_rv(t_rv, t_astro, t_0, q, a, d, P, e, M_0, w, O, cosi, sigma_rv,sigma_fov):
    sigma_rv = (sigma_rv*u.m/u.s).to(u.AU/u.yr).value	
    fish = Fisher_for_RV(t_rv, t_0, q, a, d, P, e, M_0, w, O, cosi, sigma_rv)+Fisher_for_astrometry(t_astro, t_0, q, a, d, P, e, M_0, w, O, cosi,sigma_fov)
    cov = Cov(fish)
    dct = {}
    params = ["mux_err","muy_err","muz_err","rho_err","per_err","e_err","M0_err","w_err","cosi_err","Omega_errr"]
    for i in range(len(cov)):
        dct[params[i]] = np.sqrt(cov[i][i])
    return dct

#To obtain the uncertainties using only astrometry

def get_unc_only_astrometry(t, t_0, q, a, d, P, e, M_0, w, O, cosi, sigma_fov):
    fish = Fisher_only_astrometry(t, t_0, q, a, d, P, e, M_0, w, O, cosi, sigma_fov)
    cov = Cov(fish)
    dct = {}
    params = ["mux_err","muy_err","rho_err","per_err","e_err","M0_err","w_err","cosi_err","Omega_err"]
    for i in range(len(cov)):
        dct[params[i]] = np.sqrt(cov[i][i])
    return dct
