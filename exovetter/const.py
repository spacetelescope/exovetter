
import astropy.units as u

#Time offset constants
bjd = 0 * u.day
bkjd = bjd - 2_454_833 * u.day
btjd = bjd - 2_457_000 * u.day
bet  = bjd - 2_451_544.5 * u.day  #Barycentric Emphemeris time

#Handy units to express depth
ppk = 1e-3 * u.dimensionless_unscaled
ppm = 1e-3 * ppk


planck = 6.4e-34
avogardo = 6.02e23
...
