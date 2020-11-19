import numpy as np

def gyre_read(name):
    data = np.genfromtxt(name,skip_header=5)
    # See https://bitbucket.org/rhdtownsend/gyre/wiki/Output%20Files%20(5.0)
    l = data[:,0]
    n = data[:,2]
    v = data[:,4]
    I = data[:,7]

    mask0 = l == 0
    mask1 = l == 1
    mask2 = l == 2

    return l[mask0], n[mask0], v[mask0], I[mask0], l[mask1], n[mask1], v[mask1], I[mask1], l[mask2], n[mask2], v[mask2], I[mask2]

def fakePspec(M,R,Teff,name):

    vmax = 3090*M/R**2*np.sqrt(5777/Teff)

    sigma = 0.66*vmax**0.88/(2*np.sqrt(2*np.log(2)))

    l0, n0, v0, I0, l1, n1, v1, I1, l2, n2, v2, I2 = gyre_read(name)

    num = 100

    v00 = np.linspace(min(v0),max(v0),num)

    noise_level = 0.03/(sigma*np.sqrt(2*np.pi))

    noise = np.random.normal(0,noise_level,num)

    ampl0 = np.asarray(list(noise) + list(1/(sigma*np.sqrt(2*np.pi))*np.exp(-0.5*((v0-vmax)/sigma)**2)))

    v00 = np.asarray(list(v00)+list(v0))

    index = v00.argsort()

    return v00[index], ampl0[index]
