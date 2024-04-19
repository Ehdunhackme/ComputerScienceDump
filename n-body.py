#---------- READ ME ----------
#-----------------------------

# The following is a Python 3 code
# for an N-body simulation of a collapsing orbiting particle cloud
# under gravitational gradient.

# It is a cubic cluster of randomly placed bodies
# moving relative to a central object.
# The bodies have radii and apply friction once touching.
# The central mass has a larger radius and a "ground" that fixes arriving bodies to it.

# Three procedures are performed:
# 1. Initialising pos. & vel. vectors of all bodies
# 2. Running, creating a time series. Has all positions in all frames.
# 3. Displaying the time series, or saving it.
#    One can display a saved time series.

# NumPy and Matplotlib are required.

# Further explanations are given in the code.

# Written by Alon Granek, 2020.
# ---------------------------------

# ------------------------------------------------

import numpy as np
import matplotlib.pyplot as mplot
import matplotlib.animation as anim
from mpl_toolkits.mplot3d import Axes3D
import random


#-------------------------------
#---------- VARIABLES ----------
#-------------------------------

N = 500                         # Body count. MUST BE SMALLER THAN THE CUBE LENGTH (explained further below).
isInContact = np.zeros(N)       # Is body i in contact with another? [1 OR 0, for each i]
hasFallen = np.zeros(N)         # Has body i touched the "surface" of the central mass? [1 OR 0, for each i]

#### Scalar constants ####
# The units are arbitrary, working with each consistent unit system.
Gm = 5000000            # Gravitational parameter of each small body.
GM = Gm * 10000         # Gravitational parameter of the central object.
dt = 0.001              # Timestep.
SmallDiam = 12          # Body diameter.
VelDecreaseFactor = 0.95            # A factor of decrease in body velocity when touching another.
GenCubeLength = 500                 # Length of body generation cube.
SpaceLength = 7 * GenCubeLength     # Unit for graph dimensions.
LargeRadius = GenCubeLength/10         # Radius of central object.
HeavyMassLoc = SpaceLength/2 * np.ones(3)   # Central object position.

dtCount = 500   # Or 1000, 1500...      # Frame count.
TimeSeries = np.zeros([3,N,dtCount])    # Set time series dimensions. [(x,y,z), body index, frame index]

#### State vectors ####
AccRad = np.zeros([3,N])        # Acceleration produced by central mass. [:,body] = [x,y,z] of body.
AccInt = np.zeros([3,N])        # Acceleration produced by all other bodies.
Loc = np.zeros([3,N])           # Position. [:,body] = [x,y,z] of body.
VelRad = np.zeros([3,N])        # Velocity component of body i - cloud velocity.
VelInt = np.zeros([3,N])        # Velocity component of body i - internal velocity, relative to the cloud's reference frame.
Vel0 = 5000                     # Initial cloud speed - for tangential cloud velocity.


#-------------------------------
#---------- FUNCTIONS ----------
#-------------------------------

#### Initialise positions, velocity components ####
# For simplicity, the random positions are generated
# by taking lists of all possible values for each dimension, and shuffling them.
# This, though, means no two bodies have the same x values, y values or z values.
# Also, the particle count must be smaller than the cube length.
def initialiseStates():
    GenCubeCentre = (SpaceLength-LargeRadius)/4 # Centre of body generation cube.
    # Lists of all values for each dimension.
    x = list(range(int(GenCubeCentre-(GenCubeLength/2)) , int(GenCubeCentre+(GenCubeLength/2)))) # Between space edge and surface.
    y = list(range(int((SpaceLength/2)-(GenCubeLength/2)) , int((SpaceLength/2)+(GenCubeLength/2)))) # In space mid.
    z = list(range(int((SpaceLength/2)-(GenCubeLength/2)) , int((SpaceLength/2)+(GenCubeLength/2)))) # In space mid.
    random.shuffle(x)
    random.shuffle(y)
    random.shuffle(z)
    for i in range(N-1):
        Loc[:,i] = np.array([x[i],y[i],z[i]])   # Set positions.
        VelRad[:,i] = np.array([0,Vel0,0])      # Set initial cloud velocity, fully tangential.
        VelInt[:,i] = np.array([0,0,0])         # Set internal velocity. Bodies are initially stationary relative to each other.


#### Find acceleration of body i, for the current step ####
def findCurrentAcc(i):
    # r/|r|^3 vectors for acceleration summation:
    r_Abs_r3 = 0            # By other bodies.
    R_Abs_R3 = 0            # By central object.
    isInContact[i] = 0      # Reset whether i is in contact for current timestep.
    hasFallen[i] = 0        # Reset whether i has fallen to the central object.
    # Calculate acceleration caused by central object.
    R = np.subtract(HeavyMassLoc[:],Loc[:,i])   # Position difference vector R.
    Abs_R3 = np.linalg.norm(R)**3               # |R|^3
    if (Abs_R3 >= LargeRadius**3):              # If out of central object's radius.
        R_Abs_R3 += np.divide(R,0.001*Abs_R3)   # Add to R/|R|^3. Divided by 0.001, later to be multiplied by it - otherwise the fraction is too small, yielding zero.
        AccRad[:,i] = GM * R_Abs_R3 * 0.001     # Get acceleration by central object.
        # Then calculate acceleration caused by other bodies.
        for j in range(0,N-1):              # For each other body j.
            if (j != i):                    # As long as j isn't our body i.
                r = np.subtract(Loc[:,j],Loc[:,i])  # Position difference vector r.
                Abs_r3 = np.linalg.norm(r)**3       # |r|^3
                if (Abs_r3 < SmallDiam**3): # If i is in other body j's radius:
                    isInContact[i] = 1      # Set i to be in contact.
                else:                       # Otherwise:
                    r_Abs_r3 += np.divide(r,0.001*Abs_r3)   # Add j's contribution.
        AccInt[:,i] = Gm * r_Abs_r3 * 0.001     # Get resultant acceleration by all other bodies.
    else:       # If inside the central body's radius.
        hasFallen[i] = 1        # Set i to be on the central body's ground.
        AccRad[:,i] = 0,0,0     # Remove all acceleration.
        AccInt[:,i] = 0,0,0
        # If this happens, then no function will run on body i in the next timesteps.


#### Find velocity of body i for the next timestep ####
def findNextVel(i):
    VelRad[:,i] += AccRad[:,i]*dt   # External velocity, affected by the central object.
    VelInt[:,i] += AccInt[:,i]*dt   # Internal velocity, affected by the other bodies.
    if (isInContact[i] == 1):       # If internal velocity is in contact, found by previous function.
        VelInt[:,i] = VelDecreaseFactor * VelInt[:,i]   # Reduce internal velocity by some factor - roughly approximating impulse proportional to the body's speed.


#### Find position of body i for the next timestep ####
# This uses the velocity of the CURRENT timestep, found by the previous iteration of findNextVel(i).
def findNextLoc(i):
    Loc[:,i] += np.add(VelRad[:,i],VelInt[:,i])*dt + 0.5*np.add(AccRad[:,i],AccInt[:,i])*(dt**2)
    # ^ By adding the velocity components, and adding the acceleration components.


#### Run simulation, create time series [x/y/z,body,timestep] ####
def Run(dtCount):                   # Done given a frame count.
    for t in range(dtCount-1):      # For each timestep.
        for i in range(N-1):        # For each body i.
            if (hasFallen[i] == 0): # If i is still in space: (else, no calculation)
                findCurrentAcc(i)       # Find current acceleration components.
                findNextLoc(i)          # Find next position.
                findNextVel(i)          # Find next velocity components.
        TimeSeries[:,:,t] = Loc[:,:]    # Insert positions [x/y/z,body] for each timestep.
        print(" ",(t+1),"of",dtCount,"frames loaded.",end="\r")     # Display number of loaded frames, real-time.


#### Form a scatter, displaying a frame in the time-series given timestep t.
def Update(t):
    ax.clear()
    ax.grid(False)      # No grid.
    ax.set_xticks([])   # No ticks.
    ax.set_yticks([])
    ax.set_zticks([])
    #
    ax.w_xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))     # White axes.
    ax.w_yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    ax.w_zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))     # White panes except for the floor.
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    #
    ax.set_xlim3d(SpaceLength/4,SpaceLength/2)          # Aces limits according to SpaceLength and the generation cube's length.
    ax.set_ylim3d(SpaceLength/2,SpaceLength*3/4)
    ax.set_zlim3d(SpaceLength/2-GenCubeLength/2,SpaceLength/2+GenCubeLength*3/2)
    ax.scatter(TimeSeries[0,1:N-1,t],TimeSeries[1,1:N-1,t],TimeSeries[2,1:N-1,t],s=4)   # Display time series scatter.
    ax.scatter(HeavyMassLoc[0],HeavyMassLoc[1],HeavyMassLoc[2])                         # Display central object.


#### Save created simulation ####
# Parameter "name" holds the name (or adress with name, otherwise appears in user folder)
# The function saves the time series as three text files, one for each axis.
def saveNewSim(name):
    np.savetxt(name+"X.txt",TimeSeries[0,:,:])
    np.savetxt(name+"Y.txt",TimeSeries[1,:,:])
    np.savetxt(name+"Z.txt",TimeSeries[2,:,:])


#### Load saved time series ####
def loadSavedSim(name,n,dtcount,spacelength,gencubelength):
    TimeSeries = np.zeros([3,n,dtcount])
    TimeSeries[0,:,:] = np.loadtxt(name+"X.txt")    # Load time series from files.
    TimeSeries[1,:,:] = np.loadtxt(name+"Y.txt")
    TimeSeries[2,:,:] = np.loadtxt(name+"Z.txt")
    def Update(t):              # Same as Update shown before. Declared now too as the code will skip it when not creating new simulation.
        ax.clear()
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        #
        ax.w_xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
        ax.w_yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
        ax.w_zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
        ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        #
        ax.set_xlim3d(spacelength/4,spacelength/2)
        ax.set_ylim3d(spacelength/2,spacelength*3/4)
        ax.set_zlim3d((spacelength/2)-gencubelength/2,(spacelength/2)+gencubelength*3/2)
        ax.scatter(TimeSeries[0,1:N-1,t],TimeSeries[1,1:N-1,t],TimeSeries[2,1:N-1,t],s=4)
        heavymassloc = (spacelength/2) * np.ones([3,1])
        ax.scatter(heavymassloc[0],heavymassloc[1],heavymassloc[2])
    Scatter = ax.scatter(0,0,0)
    A = anim.FuncAnimation(fig,Update,interval=40)      # Animate scatters in time interval 40ms.
    mplot.show()



fig = mplot.figure()
ax = Axes3D(fig)

plt.show()