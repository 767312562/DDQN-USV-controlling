from math import *


def linear_wave(H):
    omega = 2*pi*sqrt(0.001*9.81/H)
    T = 2 * pi / omega
    return omega, T

omega, T = linear_wave(1.25)
print('omega <', omega, '\nT >', T)

