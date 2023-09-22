----------------------------------------------

Halo catalogs for TNG

----------------------------------------------


HOW TO READ THIS DATA:

	The HDF5 files were written using the pandas python package.
	You can quickly load them and turn them into 2D numpy arrays
	using the following commands:

	*****

	import pandas as pd

	Halos = pd.read_hdf('PATH TO HDF5 FILE', key = 'Halos').to_numpy()
	

	*****

	Halos will then be a 2D numpy array that you can index into using slicing. 

	If you leave them as pandas dataframes you can also quickly access
	an individual column and turn it into a numpy array via:

	******

	Pandas_Series = Halos[quantity_name]
	Numpy_array   = Halos[quantity_name].values

	
	******

FILE NAMING:
	Each file starts with the sim run (TNG300-1 for FP, TNG300-1-Dark for DMO)
	and then follows with the snapshot.

---------------------------------------
---------------------------------------

Halo Catalog properties

	Based on Kuan's merger selection code.
	Only compute quantities if Halo has more than 100 bound DM particles

---------------------------------------	
---------------------------------------

HaloID:
	The HaloID of the halo (FoF group) associated with this object, at a given snapshot. 
	It is NOT a unique identifier over all snapshots.


SubhaloID:
	The ID of this subhalo at a given snapshot. It is NOT a unique identifier over
	all snapshots.


SubhaloID_DMO:
	The SubhaloID of the counterpart DMO sim subhalo.


SubhaloID_LastDesc:
	The SubhaloID of this Subhalo at z = 0. So the last descendant of this subhalo.


Central:
	A tag where value 0 implies it is a satellite at this epoch, and value 1 implies
	it is a Central. Designations are taken from the SUBFIND catalog.


Main:
	Is 1 if the halo is part of the main sample, and 0 if it is part of the
	progenitor sample. Obtained directly from Kuan's selection.

	
Mvir_TNG:
	The mass of halo (defined as mass within sphere where average density is
	given by the Bryan&Norman definition). The precomputed quantity from TNG.
	In units log10(msun). Available for ALL halos.


Rvir_TNG:
	The associate radius of the sphere mentioned above. In units of physical kpc.
	Available for ALL halos.


pos_{x, y, z}:
	Position of the subhalo within the sim box in units physical kpc. Same as position of halo/FoF
	since all subhalos are centrals in our selection.


vel_{x, y, z}:
	Velocity of the subhalo in km/s. Similar to above, it is the same as FoF group
	if the subhalo is a central, and different otherwise.


delta_{2, 5, 10, 20, 40}cMpc:
	The density within different spherical aperture of comoving radii R.
	Computed using the subhalos (not the particles directly), and 
	accounts for all matter components.	


cvir_{init, init_err}:
	The NFW concentration from initial fit performed for the binding calculation. This
	uses ALL available particles (so stars, gas, SMBHs in FP runs). The error is
	a fractional uncertainty obtained from the fitter (COLOSSUS).


rho_s_{init, init_err}:
	Same as above, but for the characteristic density, as obtained by the COLOSSUS fitter.
	The rho_s is given in units of Msun/pkpc^3. The error is a fractional one again.


Bound_{Nfrac, Mfrac}:
	The number (or mass) fraction of FoF particles bound to this subhalo. Potential energy 
	estimated using the expressions in Mansfield & Kravtsov 2020. For gas cells, the
	internal energy is used as part of the kinetic term.


Bound_Nfrac_SUBFIND:
	The number fraction of bound particles according to SUBFIND.


Rvir, Mvir:
	The radius and mass of a halo, defined within a sphere with average density given
	by the Bryan&Normal definition. In units of physical kpc, and log10(msun) respectively.
	Slight negative bias due to using FoF particles, which don't contain all available
	particles within the aperture.


R200c, M200c, M500c, R500c, M2500c, R2500c:
	Same quantities as above but with threshold of [200, 500, 2500]*rho_c.


Core_vel_{x, y, z}:
	The average mass-weighted velocity of all particles within 0.1*Rvir. If there
	are fewer than 100 particles in this aperture, we take the closest 100 particles.
	Mimics rockstar computation. Used in computing V_off.


Ngas, Nstars:
	The number of gas cells or star particles used when computing the integrated quantities that follow.
	Gas has no temperature cut, and stars includes wind phase cells as well. We don't compute gas/star quantities
	if there are less than 30 bound cells/particles in the halo.


M{gas, stars}_{vir, 200c, 500c, 2500c, 30cpkc, 100cpkc}:
	The integrated gas/stellar mass within different apertures. The comoving aperture quantities act as BCG-related
	quantities. In units of log10(Msun).


Z{gas, stars}_{vir, 200c, 500c, 2500c, 30cpkc, 100cpkc}:
	The integrated gas/stellar metallicity within different apertures. The comoving aperture quantities act as BCG-related
	quantities. This is computed as the mass-weighted sum of the metallicities of individual gas cells or star particles.
	In units of log10(Metallicity).

SFR_{vir, 200c, 500c, 2500c, 30cpkc, 100cpkc}:
	The total star formation rate of gas cells within different apertures. The comoving aperture quantities act as BCG-related
	quantities. In units of log10(Msun/yr).


Y_{vir, 200c, 500c, 2500c, 30cpkc, 100cpkc}:
	The compton-y parameter (a proxy for pressure) within different apertures. Units of
	1/Mpc^2 (physical distance). Computed using ALL gas cells.


Tm_{vir, 200c, 500c, 2500c, 30cpkc, 100cpkc}:
	The mass-weighted temperature in different apertures. Units of keV. Computed using ALL gas cells.


K_{vir, 200c, 500c, 2500c, 30cpkc, 100cpkc}:
	The gas entropy within different apertures. Computed as K = n_e**(-2/3)*T.
	Units of keV/cm^2. Computed using ALL gas cells.


tstar_{vir, 200c, 500c, 2500c, 30cpkc, 100cpkc}:
	The mass-weighted average age of stars within different apertures.
	In units of Gyr.


Mbh_{Rvir, Cen}, Mbh_clean_{Rvir, Cen}:
	The mass of the SMBH in the central subhalo, or mass of all
	SMBHs in the halo


Nsat_Msub{8, 9, 10, 11, 12}, Nsat_Mstar{6, 7, 8, 9, 10}:
	The number of satellite subahalos with a mass threshold on either
	Subhalo mass or stellar mass. All thresholds are in log10 Msun units


Rhalf, Rhalf_{gas, stars}:
	The radius enclosing half the total mass, or half the gas/stellar mass
	contained within Rvir. Units of physical kpc.


R{0p2, 0p8}_stars:
	Radius enclosing 20% and 80% of the stars within Rvir.
	Used as a morphology parameter in some works.
	

cvir, cvir_err:
	The concentration (and fractional uncertainty) of all bound DM particles within Rvir.


rho_s, rho_s_err:
	The characteristic density, in units of Msun/kpc^3 of the DM NFW profile


M_s, M_4s:
	The enclosed mass M(<r_s) and M(<4*r_s). Uses all particle types. In units of log10(Msun).


M_s_gas, M_s_star:
	The enclosed mass M(<r_s). Uses either gas (cold and hot) or stars (includes wind phase cells).
	In units of log10(Msun).


Vmax_{DM, gas, stars}, Rmax_{DM, gas, stars}:
	The maximum circular velocity of a given component, and the location of this maxmimum.
	Given in units of km/s and physical kpc. Rmax is bounded at the lower end
	so it cannot be less than the force softening scale.

	The gravitational potential causing the velocity profile consists of
	ALL the matter.


Vmax_DM_self, Rmax_DM_self:
	Same as the above, but the gravitational potential is using just the DM.
	Useful only for inferring a DM density profile. Not a physical quantity otherwise
	as it removes gas and stars from the gravitational potential. 
	Is equivalent to Vmax_DM if in a DMO sim.


{s, q}_{DM, gas, star}_noiter:
	The two density-space axis ratios. Computed using an 1/r_ell^2 weighted tensor,
	and using all particles within Rvir. s = c/a and q = b/a, where a > b > c. Dimensionless.
	We compute this separately for DM, gas, and stars.


A_{DM, gas, star}_eig_{x, y, z}_noiter:
	The normalized eigenvector corresponding to the major axis. Dimensionless.
	We compute this separately for DM, gas, and stars.


shapes_N_iter:
	The number of iterations used in computing the final density shapes (following quantities).
	Maximum is 20. Iteration is only done for DM


shapes_N_partfrac:
	The fraction of total bound DM particles used in computing final shape (some fraction excluded due to
	being outside ellipsoid cut). Relevant only for iterative calculation with DM.


shapes_N_part:
	Total count of DM particles used for final fit. Relevant only for iterative calculation with DM.


{s, q}_DM:
	The two density-space axis ratios. Computed using an tensor with 1/r_ell^2 weighting. We iterate
	up to 20 times, and on each iteration, we morph our selection to go from sphere to ellipsoid.
	Ellipsoid shape is given by the current estimates of s and q.


{s, q}_DM_err:
	The final uncertainty in the shape estimates. Fractional difference between past iteration's
	estimates and current estimates. Iterative process stops when error on both is below 1%, or
	when iter = 20 is reached. Whichever is first.


A_DM_eig_{x, y, z}:
	The normalized eigenvector corresponding to the major axis.


X_off_{DM, star, gas}:
	The distance between the center of mass position and the coordinate of the 
	most gravitationally bound particle in the halo/subhalo. In units of physical kpc. 
	Center of mass coordinate is computed using just a given component.
	The "Most gravitationally bound" particle can be of any type, and its position 
	is used as the center of the halo/subhalo.


V_off_{DM, star, gas}:
	Same as above, but now computes the offset between the center of mass velocity
	and the core velocity (computed above) for different components. In units of km/s


Mean_vel_{x, y, z}_{DM, gas, star}:
	The mean velocities of the different components within Rvir. In km/s


sigma_{DM, gas, star}_{X, Y, Z}:
	The velocity dispersions (standard deviation) of the different components
	within Rvir. In km/s	


sigma_{DM, gas, star}_3D:
	The isotropic velocity dispersion, obtained as 
	sigma_3D^2 = 1/3(sigma_X^2 + sigma_Y^2 + sigma_Z^2). In km/s


sigma_{DM, gas, star}_R:
	The velocity dispersion along the radial vector. In km/s


Beta_anis_{DM, gas, star}:
	The anisotropic \beta = 1 - 0.5*sigma_t^2/sigma_r^2.
	Where sigma_t^2 = 3*sigma_3D^2 - sigma_r^2.
	

Spin_Bullock_{DM, gas, star}:
	The magnitude of the halo angular momentum (using all bound particles of a component
	within Rvir) normalized by the Bullock self-similar expectation, which is
	Jselfsim = np.sqrt(2) * Mvir * Rvir * V_vir


J_{DM, gas, star}_{x, y, z}:
	The angular momentum vector of different components. 
	Also normalized by Jselfsim as defined above.


Mstar_bulge:
	The total stellar mass in the galactic bulge. Selected as all particles with
	z-angular momentum j_z < 0.7*j_tot. Includes correction factor of 0.85


Endstate:
	A string variable that logs where the pipeline ended for a given halo. Possibilities are:
	
		'Check100DMpart'       --- FoF doesn't have more than 300 DM particles.
		'FailedCvirInit'       --- Failed to get c_vir needed for binding criteria
		'Check100Boundpart'    --- FoF has less than 100 bound particles
		'Only100BoundDMInRvir' --- Less than 100 bound DM particles in Rvir
		'Success'              --- Everything passed properly! :) 
