# nbody-relaxed

## Instalation

```
conda create -n relaxed python=3.7
conda install -c conda-forge poetry 
poetry install
```

## slurm resources

See [here](https://paper.dropbox.com/doc/slurm--BItc4vwhUPkv~lWVk68u0R9sAg-BiRTZcyZDW0QJsPgr0bG6).

## data location

```
/nfs/turbo/lsa-cavestru
/scratch/cavestru_root/cavestru/imendoza/

# create symbolic link
cd /home/imendoza
ln -s /nfs/turbo/lsa-cavestru/imendoza /home/imendoza/nbody-relaxed/data
```

## Variable descriptions (ROCKSTAR)

- `Scale`: Scale factor of halo.
- `ID`: ID of halo (unique across entire simulation).
- `Desc_Scale`: Scale of descendant halo, if applicable.
- `Descid`: ID of descendant halo, if applicable.
- `Num_prog`: Number of progenitors.
- `Pid`: ID of least massive host halo (-1 if distinct halo).
- `Upid`: ID of most massive host halo (different from Pid when the halo is within two or more larger halos).
- `Desc_pid`: Pid of descendant halo (if applicable).
- `Phantom`: Nonzero for halos interpolated across timesteps.
- `SAM_Mvir`: Halo mass, smoothed across accretion history; always greater than sum of halo masses of contributing progenitors (Msun/h).  Only for use with select semi-analytic models.
- `Mvir`: Halo mass (Msun/h).
- `Rvir`: Halo radius (kpc/h comoving).
- `Rs`: Scale radius (kpc/h comoving).
- `Vrms`: Velocity dispersion (km/s physical).
- `mmp?`: whether the halo is the most massive progenitor or not.
- `scale_of_last_MM`: scale factor of the last major merger (Mass ratio > 0.3).
- `Vmax`: Maxmimum circular velocity (km/s physical).
- `X/Y/Z`: Halo position (Mpc/h comoving).
- `VX/VY/VZ`: Halo velocity (km/s physical).
- `JX/JY/JZ`: Halo angular momenta ((Msun/h) *(Mpc/h)* km/s (physical)).
- `Spin`: Halo spin parameter.
- `Breadth_first_ID`: breadth-first ordering of halos within a tree.
- `Depth_first_ID`: depth-first ordering of halos within a tree.
- `Tree_root_ID`: ID of the halo at the last timestep in the tree.
- `Orig_halo_ID`: Original halo ID from halo finder.
- `Snap_num`: Snapshot number from which halo originated.
- `Next_coprogenitor_depthfirst_ID`: Depthfirst ID of next coprogenitor.
- `Last_progenitor_depthfirst_ID`: Depthfirst ID of last progenitor.
- `Last_mainleaf_depthfirst_ID`: Depthfirst ID of last progenitor on main progenitor branch.
- `Rs_Klypin`: Scale radius determined using Vmax and Mvir (see Rockstar paper)
- `Mvir_all`: Mass enclosed within the specified overdensity, including unbound particles (Msun/h)
- `M200b--M2500c`: Mass enclosed within specified overdensities (Msun/h)
- `Xoff`: Offset of density peak from average particle position (kpc/h comoving)
- `Voff`: Offset of density peak from average particle velocity (km/s physical)
- `Spin_Bullock`: Bullock spin parameter (J/(sqrt(2)*GMVR))
- `b_to_a, c_to_a`: Ratio of second and third largest shape ellipsoid axes (B and C) to largest shape ellipsoid axis (A) (dimensionless). Shapes are determined by the method in Allgood et al. (2006). (500c) indicates that only particles within R500c are considered.
- `A[x],A[y],A[z]`: Largest shape ellipsoid axis (kpc/h comoving)
- `T/|U|`: ratio of kinetic to potential energies
- `M_pe_*`: Pseudo-evolution corrected masses (very experimental); Consistent Trees Version 1.0+; Includes fix for Rockstar spins & T/|U| (assuming T/|U| = column 53)
- `Macc,Vacc`: Mass and Vmax at accretion.
- `Mpeak,Vpeak`: Peak mass and Vmax over mass accretion history.
- `Halfmass_Scale`: Scale factor at which the MMP reaches 0.5*Mpeak.
- `Acc_Rate_*`: Halo mass accretion rates in Msun/h/yr. [Also known as gamma]
  - `Inst`: instantaneous; 100Myr: averaged over past 100Myr,
  - `X*Tdyn`: averaged over past X*virial dynamical time.
  - `Mpeak`: Growth Rate of Mpeak, averaged from current z to z+0.5
- `Mpeak_Scale`: Scale at which Mpeak was reached.
- `Acc_Scale`: Scale at which satellites were (last) accreted.
- `First_Acc_Scale`: Scale at which current and former satellites first passed through a larger halo.
- `First_Acc_(Mvir|Vmax)`: Mvir and Vmax at First_Acc_Scale.
