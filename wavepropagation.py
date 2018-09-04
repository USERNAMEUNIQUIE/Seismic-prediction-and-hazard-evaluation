import numpy as np
from numba import jit    # library to  increase the performance of matrix operations
import matplotlib.pyplot as plt

import matplotlib.animation as animation   # for animations  :D


xmax = 2000.0 # size of the figure in meter in x direction (m)
zmax = xmax   # size in z-direction (m)

tmax = 3.20   # maximum recording time of the seismogram (s)

vp0  = 3000.  # P-wave speed in medium (m/s)

xsrc = 1000.0 # x-source position (m)
zsrc = 1000.0   # z-source position (m)

f0   = 500.0 # dominant frequency of the source (Hz)
t0   = 0.1   # source time shift (s)

isnap = 2  # snapshot interval (timesteps)

fig = plt.figure(figsize=(7, 3.5))  # define figure size
plt.tight_layout()
listData=[]


# define figure size


@jit(nopython=True)  # use JIT for C-performance
def update_d2px_d2pz(p, dx, dz, nx, nz, d2px, d2pz):
    for i in range(1, nx - 1):
        for j in range(1, nz - 1):
            d2px[i, j] = (p[i + 1, j] - 2 * p[i, j] + p[i - 1, j]) / dx ** 2
            d2pz[i, j] = (p[i, j + 1] - 2 * p[i, j] + p[i, j - 1]) / dz ** 2

    return d2px, d2pz



def model(nx, nz, vp, dx, dz):
    layer1 = 120.  # layer1
    n_ft = (int)(layer1 / dz)  # layer 1   converted to mesh scale
    vp += vp0
    vp[:,nz // 2 - n_ft: nz //2 + n_ft,] *= 0.8

    layer2= 300.   # layer2
    n_ft2 = (int)(layer2 / dz)   #layer2  converted to mesh scale
    print(n_ft2)
    vp[ :, 0:  n_ft2] *= 0.5 # modify the vp array
    print (vp)


    return vp


def absorb(nx, nz):
    FW = 60  # thickness of absorbing frame (gridpoints)
    a = 0.0053  # damping variation within the frame

    coeff = np.zeros(FW)

    # define coefficients
    for i in range(FW):
        coeff[i] = np.exp(-(a ** 2 * (FW - i) ** 2))

    # initialize array of absorbing coefficients
    absorb_coeff = np.ones((nx, nz))

    # compute coefficients for left grid boundaries (x-direction)
    zb = 0
    for i in range(FW):
        ze = nz - i - 1
        for j in range(zb, ze):
            absorb_coeff[i, j] = coeff[i]

    # compute coefficients for right grid boundaries (x-direction)
    zb = 0
    for i in range(FW):
        ii = nx - i - 1
        ze = nz - i - 1
        for j in range(zb, ze):
            absorb_coeff[ii, j] = coeff[i]

    # compute coefficients for bottom grid boundaries (z-direction)
    xb = 0
    for j in range(FW):
        jj = nz - j - 1
        xb = j
        xe = nx - j
        for i in range(xb, xe):
            absorb_coeff[i, jj] = coeff[j]

    return absorb_coeff


def FD_2D_acoustic_JIT_absorb(dt, dx, dz, f0):
    # define model discretization
    # ---------------------------
    global image

    nx = (int)(xmax / dx)  # number of grid points in x-direction
    print('nx = ', nx)

    nz = (int)(zmax / dz)  # number of grid points in x-direction
    print('nz = ', nz)

    nt = (int)(tmax / dt)  # maximum number of time steps
    print('nt = ', nt)

    isrc = (int)(xsrc / dx)  # source location in grid in x-direction
    jsrc = (int)(zsrc / dz)
    print('location of source =', isrc,'  x direction' ,jsrc ,' z dircetion'  )                         # source location in grid in z-direction

    # Source time function (Gaussian)
    # -------------------------------
    src = np.zeros(nt + 1)
    time = np.linspace(0 * dt, nt * dt, nt)

    # 1st derivative of Gaussian
    src = -2. * (time - t0) * (f0 ** 2) * (np.exp(- (f0 ** 2) * (time - t0) ** 2))

    # define clip value: 0.1 * absolute maximum value of source wavelet
    clip = 0.1 * max([np.abs(src.min()), np.abs(src.max())]) / (dx * dz) * dt ** 2

    # Define absorbing boundary frame
    # -------------------------------
    absorb_coeff = absorb(nx, nz)

    # Define model
    # ------------
    vp = np.zeros((nx, nz))
    vp = model(nx, nz, vp, dx, dz)
    vp2 = vp ** 2


    # Initialize empty pressure arrays
    # --------------------------------
    p = np.zeros((nx, nz))  # p at time n (now)
    pold = np.zeros((nx, nz))  # p at time n-1 (past)
    pnew = np.zeros((nx, nz))  # p at time n+1 (present)
    d2px = np.zeros((nx, nz))  # 2nd spatial x-derivative of p
    d2pz = np.zeros((nx, nz))  # 2nd spatial z-derivative of p

    # Initalize animation of pressure wavefield
    # -----------------------------------------

    extent = [0.0, xmax, zmax, 0.0]  # define model extension

    # Plot pressure wavefield movie



    # Plot Vp-model

    image = plt.imshow(p.T, animated=True, cmap="RdBu", extent=extent,
                       interpolation='nearest', vmin=-clip, vmax=clip)
    plt.title('Pressure wavefield ')
    plt.colorbar()
    plt.xlabel('x [m]')
    plt.ylabel('z [m]')


    for it in range(nt):

        # FD approximation of spatial derivative by 3 point operator
        d2px, d2pz = update_d2px_d2pz(p, dx, dz, nx, nz, d2px, d2pz)

        # Time Extrapolation
        # ------------------
        pnew = 2 * p - pold + vp2 * dt ** 2 * (d2px + d2pz)

        # Add Source Term at isrc
        # -----------------------
        # Absolute pressure w.r.t analytical solution
        pnew[isrc, jsrc] = pnew[isrc, jsrc] + src[it] / (dx * dz) * dt ** 2

        # Apply absorbing boundary frame
        p *= absorb_coeff
        pnew *= absorb_coeff

        # Remap Time Levels
        # -----------------
        pold, p = p, pnew



        # display pressure snapshots

        #image.set_data(p.T)
        fig.canvas.draw()

        listData.append(p.T)

    return


def animate(i):
    image.set_data(listData[i])
    return [image]


dx   = 5.0   # grid point distance in x-direction (m)
dz   = dx     # grid point distance in z-direction (m)

# calculate dt according to the CFL-criterion
dt = dx / (np.sqrt(2.0) * vp0)

print(dt)


ani = animation.FuncAnimation(fig, animate, init_func=FD_2D_acoustic_JIT_absorb(dt,dx,dz,f0), interval=1, blit=True, save_count=len(listData)-1)


from matplotlib.animation import FFMpegWriter
writer = FFMpegWriter(fps=20)
ani.save("movie source time.mp4")






