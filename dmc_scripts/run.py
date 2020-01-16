#!/usr/bin/env python

import numpy as np
import os

from nexus import settings,Job,run_project,obj
from nexus import Structure, PhysicalSystem
from nexus import generate_physical_system
from nexus import generate_pwscf
from nexus import generate_pw2qmcpack, read_structure
from nexus import generate_qmcpack,vmc,loop,linear,dmc
from structure import optimal_tilematrix


# General Settings (Directories For I/O, Machine Type, etc.)
settings(
    pseudo_dir      = '../pseudopotentials',
    runs            = 'runs',
    results         = 'results',
    sleep           = 5,
    generate_only   = 0,
    status_only     = 1,
    machine         = 'cades',
    account         = 'qmc'
    )

#CHANGE_HERE_ONLY#
######################################################################
########################## Change here only ##########################`

#STRUCTURE

prim = read_structure('POSCAR')
prim.frozen = None

ph = generate_physical_system(
    structure = prim.copy(),
    Li = 1,
    Ni = 18,
    O = 6,
    net_spin = 1
)
mag = [0.7]
tot_mag = 1

#######################################################################
########################## Change here only ##########################

shared_scf = obj(
    input_type  = 'generic',
    occupations = 'smearing',
    smearing    = 'gaussian',
    degauss     = 0.005,
    tot_magnetization = tot_mag,
    ####################
    #plus-u is here
    input_DFT   = 'PBE',
    lda_plus_u  = True,
    ####################
    ecutwfc     = 320,
    start_mag   = obj(Ni=mag[0]),
    conv_thr    = 1.0e-7,
    mixing_beta = 0.2,
    # no_sym is deleted! jtk added
    use_folded  = True,
)
params = obj(
    electron_maxstep = 100,
    system           = ph,
    j                = 11,
    dft_pps          = ['Li.BFD.upf', 'Ni.opt.upf','O.opt.upf'],
    qmc_pps          = ['Li.BFD.xml', 'Ni.opt.xml','O.opt.xml'],
    dft_grid         = (5,5,5),
    tilings          = [[[1, 0, 0], [0, 1, 0], [0, 0, 1]]],
    qmc_grids        = [(3,3,3)],
    uatoms           = ['Ni'],
    ulist            = [3.0],
    structure        = ph,
    tot_mag          = tot_mag,
    machine          = 'cades',
    j3               = True,
    j2               = False,
    code             = 'cpu',
    nopre            = False,
    density          = (80, 80, 80),
    natoms           = 4,
    tmoves           = False,
    vmc_opt          = True
)

import workflows
sims = []
sims = workflows.mworkflow(shared_scf, params, sims)
run_project(sims)

