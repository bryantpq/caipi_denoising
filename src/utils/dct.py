from numpy import r_
from scipy.fftpack import dct, dst, fft, ifft, idct

def dct2(a):
    return dct( dct(a, axis=1), axis=2)

def idct2(a):
    return idct( idct(a, axis=1), axis=2)