import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Attractor(object):
    def __init__(self, s=10.0, b=(8.0/3.0), p=28.0, start=0.0, end=80.0, points=10000):
        """Computes Euler and Runge-Kutta estimates and produces various plots.

        Computes Euler, Runge-Kutta order 2, and Runge-Kutta order 4 estimates from a numpy array of length 3 and returns a table with the computed t, x, y,         and z values.  Produces 1D (t vs. x; t vs. y; t vs. z), 2D (x vs. y; y vs. z; z vs. x) and 3D (x vs. y vs. z) plots.

        Keyword Arguments:
            s: initialization parameter (set to 10.0)
            b: initialization parameter (set to 8.0/3.0)
            p: initialization parameter (set to 28.0)
            start: starting point (initialized at 0.0)
            end: ending point (initialized at 80.0)
            points: number of points evaluated (initialized at 10000)

        Other Initial Values:
            params: a numpy array [s, b, p]
            dt: (end - start)/points
            t: np.linspace(start, end, points)
            x: np.zeros(points)
            y: np.zeros(points)
            z: np.zeros(points)
            """
        self.s = s
        self.b = b
        self.p = p
        self.start = start
        self.end = end
        self.points = points
        self.params = np.array([self.s, self.b, self.p])
        self.dt = (end - start)/points
        self.t = np.linspace(start, end, points)
        self.x = np.zeros(points)
        self.y = np.zeros(points)
        self.z = np.zeros(points)

    def deriv(self, r):
        """Returns derivates dx, dy, dz in a numpy array.

        Keyword Arguments:
                r: numpy array of length 3
                """
        x,y,z = r
        s, b, p = self.params
        dx = s * (y - x)
        dy = ((x * (p - z)) - y)
        dz = ((x * y) - (b * z))
        dr = np.array([dx, dy, dz])
        return dr
    
    def euler(self, r):
        """Returns deriv(r) * dt.
        
        Keyword Arguments:
            r: numpy array of length 3
            """
        return self.deriv(r) * self.dt
    
    def rk2(self, r):
        """Returns deriv(r1) * dt.
        
        Keyword Arguments:
            r: numpy array of length 3
            
        Other Values:
            r1: new increment (r + deriv(r) * dt/2)
        """
        r1 = r + self.deriv(r) * self.dt/2
        dr = self.deriv(r1) * self.dt
        return dr
    
    def rk4(self, r):
        """Returns an average of 4 Runge-Kutta derivative estimates.
        
        Keyword Arguments:
            r: numpy array of length  3
        """
        k1 = self.deriv(r)
        k2 = self.deriv(r + k1 * self.dt/2)
        k3 = self.deriv(r + k2 * self.dt/2)
        k4 = self.deriv(r + k3 * self.dt)
        dr = ((k1 + 2*k2 + 2*k3 + k4) * self.dt)/6
        return dr
    
    def evolve(self, r0=[0.1, 0.0, 0.0], order=4):
        """Returns a table of solutions, depending on selected order, with columns t, x, y, z and saves it to a CSV file.
        
        Keyword Arguments:
            r0: numpy array of lenth 3 (default 0.1, 0.0, 0.0)
            order: 1 for euler, 2 for rk2, 4 for rk4 (default 4)
        """
        x0, y0, z0 = r0
        self.x[0] = x0
        self.y[0] = y0
        self.z[0] = z0

        if order == 1:
            inc = self.euler
        elif order == 2:
            inc = self.rk2
        elif order == 4:
            inc = self.rk4

        for i in xrange(0, self.t.size-1):
            now = np.array([self.x[i], self.y[i], self.z[i]])
            dx, dy, dz = inc(now)
            self.x[i+1] = self.x[i] + dx
            self.y[i+1] = self.y[i] + dy
            self.z[i+1] = self.z[i] + dz

        self.solution = pd.DataFrame({"t": self.t, "x": self.x, "y": self.y, "z": self.z})
        return self.solution

    def save(self):
        """Save solution to CSV file on disk."""
        self.solution.to_csv("export.csv")
            
    def plotx(self):
        """Plots t vs. x."""
        plt.plot(self.t, self.x)
        
    def ploty(self):
        """Plots t vs. y."""
        plt.plot(self.t, self.y)
     
    def plotz(self):
        """Plots t vs. z."""
        plt.plot(self.t, self.z)
     
    def plotxy(self):
        """Plots x vs. y."""
        plt.plot(self.x, self.y)

    def plotyz(self):
        """Plots y vs. z."""
        plt.plot(self.y, self.z)

    def plotzx(self):
        """Plots z vs. x."""
        plt.plot(self.z, self.x)
        
    def plot3d(self):
        """Plots x vs. y vs. z in a 3D plane."""
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(self.x, self.y, self.z)
        plt.show()
        