#!/usr/bin/env python
from nexus import settings,Job,run_project,obj
from nexus import Structure, PhysicalSystem
from nexus import generate_physical_system
from nexus import generate_pwscf
from nexus import generate_pw2qmcpack, read_structure
from nexus import generate_qmcpack,vmc,loop,linear,dmc
from structure import optimal_tilematrix

import numpy as np
import pdb, os
import sys
from xml.dom import minidom

sys.dont_write_bytecode = True

params_defaults = obj(
        electron_maxstep    = 200,
        system              = None,
        j                   = 11,
        dft_pps             = None,
        qmc_pps             = None,
        dft_grid            = (6,6,6),
        tilings             = [[[1, 0, 0], [0, 1, 0], [0, 0, 1]]],
        qmc_grids           = [(3,3,3)],
        uatoms              = None,
        ulist               = None,
        charge              = 0,
        tot_mag             = None,
        machine             = 'titan',
        j3                  = True,
        j2                  = False,
        code                = 'cpu',
        qp                  = False,
        big                 = False,
        density             = None,
        nopre               = False,
        tmoves              = False,
        meshfactor          = 1.0,
        hybrid_rcut         = None,
        hybrid_lmax         = None,
        rcut                = None,
        two_u               = False,
        natoms              = None,
        vmc                 = False,
        vmc_opt             = True,
        twistnum            = 1,
        excitation          = None,
        bundle              = False,
        qmc_skip_submit     = False,
        timesteps           = None,
        qmc                 = True,
        seed                = None,
        kshift              = (0,0,0),
        print_ke_corr       = False,
        relax               = False,
        relax_functional    = 'PBESOL',
        relax_u             = 4,
        relax_pps           = None,
        relax_pp_dir        = None,
        nscf_only           = False,
        nscf_skip           = False,
)

def mworkflow(shared_qe, params, sims):

        for i in params_defaults.keys():
                if i not in params:
                        setattr(params, i, params_defaults[i])
                #end if
        #end if
        if params.j3:
                if params.code == 'gpu':
                        print "GPU code and j3 not possible yet"
                        exit()
                #end if
        #end if
        if params.excitation is not None:
                params.twistnum = 0
        ########################
        #DMC and VMC parameters#
        ########################
        samples      = 25600 #25600
        vmcblocks    = 100
        dmcblocks    = 300
        dmceqblocks  = 200
        dmcwalkers   = 1024 # For cades it is multiplied by 2
        vmcdt = 0.3
        if params.vmc:
            vmcblocks = 9000
            vmcdt = 0.3
            dmcblocks = 1
            dmceqblocks = 1
        #end if
        #############################
        # Machine specific variables#
        #############################
        if params.machine == 'cades':
            minnodes  = 1
            scf_nodes = 2;  scf_hours = 48;
            p2q_nodes = scf_nodes;  p2q_hours = 1 ;

            qeapp='pw.x -npool 4 '
            if params.dft_grid == (1,1,1):
                scf_nodes = 1
                qeapp = 'pw.x'
            #end if
            qe_presub = 'module purge; module load PE-intel/3.0; module load hdf5_parallel/1.10.3'

            opt_nodes = 4;  opt_hours = 12;
            dmcnodespertwist = 0.5; dmc_hours = 24;
	    qmc_threads = 18
            walkers = qmc_threads
            blocks     = samples/(walkers*opt_nodes)
            params.code = 'cpu'
            qmcapp=' qmcpack_cades_cpu_comp_SoA'
            qmc_presub = 'module purge; module load PE-intel'

            opt_job  = Job(nodes=opt_nodes, hours=opt_hours, threads=qmc_threads,  app=qmcapp, presub=qmc_presub)

        elif params.machine == 'titan' or params.machine == 'eos':
            minnodes  = 1
	    dmcwalkers *= 2
            scf_nodes = 8;  scf_hours = 2;
            p2q_nodes = scf_nodes;  p2q_hours = 1;

            qeapp='pw.x -npool 4 '
            qe_presub = ''

            if params.machine == 'eos':
                opt_nodes = 16;  opt_hours = 2;
                qmc_threads = 16
                walkers = qmc_threads
                blocks     = samples/(walkers*opt_nodes)

                params.code = 'cpu'
                qmcapp      = '/ccs/home/kayahan/SOFTWARE/qmcpack/eos/qmcpack/qmcpack_eos_cpu_comp_SoA'
                qmc_presub  = ''

                params.rundmc = False
            elif params.machine == 'titan':
                if params.code == 'cpu':
                    dmcwalkers *=2
                    vmcblocks   =50
                    opt_nodes = 16;  opt_hours = 2;
                    dmcnodespertwist = 32; dmc_hours = 6;
                    qmc_threads = 16
                    walkers = qmc_threads
                    blocks     = samples/(walkers*opt_nodes)

                    qmcapp      = '/ccs/home/kayahan/SOFTWARE/qmcpack/titan/qmcpack/qmcpack_titan_cpu_comp_SoA'
                    qmc_presub  = ''

                    params.rundmc = True
                elif params.code == 'gpu':
                    dmcwalkers *=8
                    opt_nodes = 4;  opt_hours = 4;
                    dmcnodespertwist = 8; dmc_hours = 1.5;
                    qmc_threads = 16
                    walkers = qmc_threads
                    blocks     = samples/(walkers*opt_nodes)
                    qmcapp      = '/ccs/home/kayahan/SOFTWARE/qmcpack/titan/qmcpack/qmcpack_titan_gpu_comp_SoA'
                    qmc_presub  = 'module load cudatoolkit'
                #end if
            #end if

            opt_job  = Job(nodes=opt_nodes, hours=opt_hours, threads=qmc_threads,  app=qmcapp, presub=qmc_presub)

        elif params.machine == 'cetus' or params.machine == 'mira':
            if params.machine == 'cetus':
                minnodes = 128
                opt_hours = 1
                dmc_hours = 1
            elif params.machine == 'mira':
                minnodes = 128
                dmc_hours = 1
                dmc_hours = 1
            #end if

            scf_nodes = 128; scf_hours = 1;
            p2q_nodes = 128; p2q_hours = 1;

            qeapp='pw.x'
            qe_presub = ''

            opt_nodes = 128.0
            opt_nodes = np.round(opt_nodes/minnodes)*minnodes
            dmcnodespertwist = 18
            qmc_threads = 16
            walkers = qmc_threads
            qmc_processes_per_node = 16
            blocks     = samples/(walkers*opt_nodes)
            params.code = 'cpu'
            qmcapp      = 'qmcpack'
            qmc_presub  = ''
            opt_job  = Job(nodes=opt_nodes, hours=opt_hours, threads=qmc_threads,  processes_per_node = qmc_processes_per_node, app=qmcapp, presub=qmc_presub)
        elif params.machine == 'summit':
	    dmcwalkers *=8
            minnodes  = 1
            scf_nodes = 2;  scf_hours = 24;
            p2q_nodes = 2;  p2q_hours = 1;

            qeapp='pw.x -npool 4 '
            if params.dft_grid == (1,1,1):
                scf_nodes = 1
                qeapp = 'pw.x'
            #end if
            qe_presub = 'module purge; module load PE-intel/3.0'
            opt_nodes = 4;  opt_hours = 4;
            dmcnodespertwist = 8; dmc_hours = 6;
            qmc_threads = 42
            walkers = qmc_threads
            blocks     = samples/(walkers*opt_nodes)
            qmcapp      = '/ccs/home/kayahan/SOFTWARE/qmcpack/summit/qmcpack/build_summit_comp/bin/qmcpack'
        else:
            print "Machine is not defined!"
            exit()
        #end if

        scf_job = Job(nodes=scf_nodes, hours=scf_hours, app=qeapp, presub=qe_presub)
        p2q_job = Job(nodes=p2q_nodes, hours=p2q_hours, app='pw2qmcpack.x -npool 4 ', presub=qe_presub)

        
        
        if params.big or params.qp:
            dmcblocks        *=4
            dmcnodespertwist *=1
            dmc_hours        *=2
            dmcwalkers       *=2
        #end if
        shared_qe.wf_collect = False
        shared_qe.nosym      = True
	shared_qe.job     = scf_job
        shared_qe.pseudos = params.dft_pps
        if params.tot_mag is not None:
                shared_qe.tot_magnetization = params.tot_mag
        #end if
        if params.two_u:
                ulist = [params.ulist[0]]
        else:
                ulist = params.ulist
        #end if
        shared_relax = shared_qe.copy()

        # qmcs list to possibly bundle all DMC calculations
        qmcs = []
        scf_ks_energy = 0 
	for u in ulist:
                if u != 0.:
                        shared_qe.hubbard_u = obj()
                        shared_relax.hubbard_u = obj()
                        for inum, i in enumerate(params.uatoms):
                                if params.two_u:
                                        setattr(shared_qe.hubbard_u, i, params.ulist[inum])
                                else:
                                        setattr(shared_qe.hubbard_u, i, u)
                                        setattr(shared_relax.hubbard_u, i, params.relax_u)
                                        #end if
                        #end for
                #end if
                scf_inf = None
                scf_path = './scf-u-'+str(u)+'-inf'
                if params.relax:
                        shared_relax = shared_qe.copy()
                        shared_relax['ion_dynamics'] = 'bfgs'
                        shared_relax['cell_dynamics'] = 'bfgs'
                        shared_relax['calculation']  = 'vc-relax'
                        shared_relax['conv_thr']=10e-6
                        shared_relax['forc_conv_thr']=0.001
                        shared_relax['input_DFT'] = params.relax_functional
                        shared_relax.nosym = True
                        coarse_relax = generate_pwscf(
                                identifier   = 'scf',
                                path='./coarse_relax-u-'+str(params.relax_u)+'-inf',
                                electron_maxstep=params.electron_maxstep,
                                nogamma = True,
                                system=params.system,
                                kgrid = params.dft_grid,
                                **shared_relax
                                )
                        sims.append(coarse_relax)
                        
                        shared_relax['conv_thr']=10e-8
                        shared_relax['forc_conv_thr']=0.001
                        fine_relax = generate_pwscf(
                                identifier   = 'scf',
                                path='./fine_relax-u-'+str(params.relax_u)+'-inf',
                                electron_maxstep=params.electron_maxstep,
                                nogamma = True,
                                system=params.system,
                                kgrid = params.dft_grid,
                                dependencies = (coarse_relax, 'structure'),
                                **shared_relax
                        )
                        sims.append(fine_relax)

                        
		        scf_inf = generate_pwscf(
                                identifier   = 'scf',
                                path=scf_path,
                                electron_maxstep=params.electron_maxstep,
                                nogamma = True,
                                system=params.system,
                                #nosym = True,
                                kgrid = params.dft_grid,
                                calculation = 'scf',
                                dependencies = (fine_relax, 'structure'),
                                **shared_qe
                        )
                        #pdb.set_trace()
                        
                else:
                	scf_inf = generate_pwscf(
                                identifier   = 'scf',
                                path=scf_path,
                                electron_maxstep=params.electron_maxstep,
                                nogamma = True,
                                system=params.system,
                                #nosym = True,
                                kgrid = params.dft_grid,
                                calculation = 'scf',
                                **shared_qe
                        )
                        		
                sims.append(scf_inf)

                if params.nscf_only is True:
                        params.tilings = []
                        params.vmc_opt = False
                        params.qmc = False
                        
                if params.print_ke_corr:
                        scf_dir = scf_inf.remdir
                        pwscf_output = scf_inf.input.control.outdir
                        datafile_xml = scf_dir + '/' + pwscf_output + '/pwscf.save/data-file-schema.xml'
                        if os.path.isfile(datafile_xml):
                                xmltree = minidom.parse(datafile_xml)
                                ks_energies = xmltree.getElementsByTagName("ks_energies")
                                eig_tot = 0
                                for kpt in ks_energies:
                                        w = float(kpt.getElementsByTagName('k_point')[0].getAttribute('weight'))
                                        eigs = np.array(kpt.getElementsByTagName('eigenvalues')[0].firstChild.nodeValue.split(' '), dtype = 'f')
                                        occs = np.array(kpt.getElementsByTagName('occupations')[0].firstChild.nodeValue.split(' '), dtype = 'f') #Non-integer occupations from DFT
                                        eig_tot += w*sum(eigs*occs)/2 #Divide by two for both spins
                                #end for
                                print scf_dir, "e_tot eigenvalues in Ha ", eig_tot/2
                                if scf_ks_energy == 0:
					scf_ks_energy = eig_tot/2
                        #end if
                #end if
                for tl_num, tl in enumerate(params.tilings):
                        for k in params.qmc_grids:
                                knum = k[1]*k[2]*k[0]
                                if params.relax:
                                        relax = fine_relax #prim = scf_inf.load_analyzer_image().input_structure.copy()
                                        print 'Make sure the previous run is loaded to results!'
                                        #scf_inf.system.structure
                                else:
                                        relax = scf_inf
                                        prim = params.system.structure.copy()
                                        #system = params.system
                                prim.clear_kpoints()
                                super = prim.tile(tl)

                                tl_np = np.array(tl)
                                tl_det = int(np.abs(np.linalg.det(tl_np))+0.001)

                                if params.natoms is not None:
                                        natoms = tl_det * params.natoms
                                else:
                                        natoms = len(super.elem)
                                #end if
                                if params.rcut:
                                        rcut = params.rcut[tl_num]
                                else:
                                        rcut = super.rwigner()-0.00001
                                #end if
                                if params.tot_mag is not None and params.qp is False:
                                        system = generate_physical_system(
                                                structure  = prim,
                                                tiling     = tl,
                                                kgrid      = k,
                                                kshift     = params.kshift, #(0, 0, 0),
                                                net_charge = params.charge,
                                                net_spin   = params.tot_mag,
                                                use_prim   = False,
                                                **params.system.valency
                                        )
                                elif params.qp is False:
                                        system = generate_physical_system(
                                                structure  = prim,
                                                tiling     = tl,
                                                kgrid      = k,
                                                shift     = params.kshift, #(0, 0, 0),
                                                net_charge = params.charge,
                                                **params.system.valency
                                        )
                                else:
                                        system = generate_physical_system(
                                                structure  = prim,
                                                tiling     = tl,
                                                kgrid      = k,
                                                kshift     = params.kshift, #(0, 0, 0),
                                                use_prim   = False,
                                                **params.system.valency
                                        )
                                #end if
                                # DFT NSCF To Generate Wave Function At Specified K-points

                                if params.dft_grid == params.qmc_grids[0] and len(params.qmc_grids) == 1:
                                        params.nscf_skip = True

                                if not params.nscf_skip:
                                        if params.relax:
                                                nscf_dep = [(scf_inf, 'charge_density'), (relax, 'structure')]
                                        else:
                                                nscf_dep = [(scf_inf, 'charge_density')]
                                        #end if
                                        nscf_path = './nscf-u-'+str(u)+'-'+str(knum)+'-'+str(natoms)
                                        p2q_path  = nscf_path
                                        nscf = generate_pwscf(
                                                identifier   = 'nscf',
                                                path=nscf_path,
                                                system=system,
                                                #nosym = True,
                                                nogamma = True,
                                                dependencies= nscf_dep,
                                                calculation = 'nscf',
                                                **shared_qe
                                        )
                                        sims.append(nscf)
                                        
                                        if params.relax:
                                                p2q_dep = [(nscf, 'orbitals'), (relax, 'structure')]
                                        else:
                                                p2q_dep = [(nscf, 'orbitals')]
                                        #end if
                                else:
                                        p2q_path = scf_path
                                        if params.relax:
                                                p2q_dep = [(scf_inf, 'orbitals'), (relax, 'structure')]
                                        else:
                                                p2q_dep = [(scf_inf, 'orbitals')]
                                        #end if
                                #end if
                                # Convert DFT Wavefunction Into HDF5 File For QMCPACK
                                p2q = generate_pw2qmcpack(
                                        identifier='p2q',
                                        path=p2q_path,
                                        job=p2q_job,
                                        write_psir=False,
                                        dependencies=p2q_dep,
                                )
                                sims.append(p2q)
                                
                                if params.print_ke_corr and not params.nscf_skip:
                                        import h5py
                                        nscf_dir = nscf.remdir
                                        pwscf_output = nscf.input.control.outdir
                                        h5_file  = nscf_dir+'/'+pwscf_output+'/pwscf.pwscf.h5'
                                        #Assume equal weight for all k-points
                                        eig_tot = 0
                                        nkpts = 0
                                        if os.path.isfile(h5_file):
                                                f = h5py.File(h5_file,'r+')
                                                nelect = f['electrons']['number_of_electrons'][:]
                                                
                                                for group_e in f['electrons'].keys():
                                                        if group_e.startswith('kpoint'):
                                                                nkpts += 1
                                                                f_temp = f['electrons'][group_e]
                                                                for group_k in f_temp.keys():
                                                                        k_s_eig = [0]
                                                                        w = 0
                                                                        if group_k.startswith('spin_0'):
                                                                                k_s_eig = f_temp[group_k]['eigenvalues'][:]
                                                                                k_s_eig = k_s_eig[0:nelect[0]] #fixed occupations from the number of electrons
                                                                                w = f_temp['weight'][0]
                                                                        elif group_k.startswith('spin_1'):
                                                                                k_s_eig = f_temp[group_k]['eigenvalues'][:]
                                                                                k_s_eig = k_s_eig[0:nelect[1]]
                                                                                w = f_temp['weight'][0]
                                                                                #pdb.set_trace()
                                                                        #end if
                                                                        eig_tot += sum(k_s_eig)*w/2
                                                                #end for
                                                        #end if
                                                #end fof
                                                print nscf_dir, "e_tot eigenvalues in Ha", eig_tot/2-scf_ks_energy, "wigner " + str(system['structure'].rwigner())
                                        #end if
                                #end if

                                # DFT is complete, rest is QMC
                                
                                # change here for other AFM atoms
                                system.rename(Co1='Co',Co2='Co',Co3='Co', folded=False)
                                system.rename(Ni1='Ni',Ni2='Ni',Ni3='Ni', folded=False)
                                
				# VMC Optimization
                                linopt1 = linear(
                                        energy               = 0.0, # 0.95
                                        unreweightedvariance = 1.0,
                                        reweightedvariance   = 0.0, # 0.05
                                        timestep             = 0.3,
                                        samples              = samples,
                                        walkers              = walkers,
                                        warmupsteps          = 10,
                                        blocks               = blocks,
                                        steps                = 1,
                                        substeps             = 10,
                                        maxweight            = 1e9,
                                        gpu                  = True,
                                        minmethod            = 'OneShiftOnly',
                                        minwalkers           = 0.01,
                                        usebuffer            = True,
                                        exp0                 = -6,
                                        bigchange            = 10.0,
                                        alloweddifference    = 1e-04,
                                        stepsize             = 0.15,
                                        nstabilizers         = 1,
                                )
                                
                                # Quasiparticle calculation
                                if params.qp:
                                        etot  = system.particles.down_electron.count + system.particles.up_electron.count
                                        upe   = (etot + params.tot_mag* tl_det)/2 - params.charge
                                        downe = etot - params.charge - upe
                                        system.particles.down_electron.count = downe
                                        system.particles.up_electron.count   = upe
                                #end if

                                # 1. VMC optimization is requested
                                # 2. VMC optimization is done on the first element uf ulist
                                # 3. VMC optimization is done on the first element of qmc_grids
                                
                                if params.vmc_opt and u == params.ulist[0] and k == params.qmc_grids[0]:

                                        # VMC Variance minimization
                                        
                                        opt_varmin = generate_qmcpack(
                                                identifier     = 'opt-varmin',
                                                path           = './opt-u-'+str(u)+'-'+str(natoms)+ '-varmin-' + str(params.j),
                                                job            = opt_job,
                                                input_type     ='basic',
                                                system         = system,
                                                spin_polarized = True, # jtk: needs this
                                                meshfactor     = 1.0,
                                                hybrid_rcut    = params.hybrid_rcut,
                                                hybrid_lmax    = params.hybrid_lmax,
                                                #spline_radius  = params.spline_radius,
                                                twistnum       = params.twistnum,
                                                #bconds         = 'ppp',
                                                pseudos        = params.qmc_pps,
                                                jastrows       = [('J1', 'bspline', params.j, rcut),
                                                                  ('J2', 'bspline', params.j, rcut, 'init', 'zero')],
                                                calculations   = [loop(max=6, qmc=linopt1), loop(max=6, qmc=linopt1)],
                                                dependencies   = (p2q, 'orbitals')
                                        )
                                        sims.append(opt_varmin)
                                        
                                        # QMC Optimization Parameters - Finer Sampling Set -- Energy Minimization
                                        linopt2 = linopt1.copy()
                                        linopt2.minwalkers=0.5
                                        linopt2.energy=0.95
                                        linopt2.unreweightedvariance=0.0
                                        linopt2.reweightedvariance=0.05
                                        linopt3 = linopt2.copy()
                                        linopt3.minwalkers = 0.5
                                        ##
                                        emin_dep = []

                                        # Use preliminary optimization step made of two steps 
                                        if not params.nopre:
                                                preopt_emin = generate_qmcpack(
                                                        identifier     = 'opt-emin',
                                                        path           = './preopt-u-'+str(u)+'-'+str(natoms) + '-emin-' + str(params.j),
                                                        job            = opt_job,
                                                        input_type     ='basic',
                                                        system         = system,
                                                        spin_polarized = True, # jtk: needs this
                                                        meshfactor     = params.meshfactor,
                                                        hybrid_rcut    = params.hybrid_rcut,
                                                        hybrid_lmax    = params.hybrid_lmax,
                                                        #spline_radius  = params.spline_radius,
                                                        twistnum       = params.twistnum,
                                                        #bconds         = 'ppp',
                                                        pseudos        = params.qmc_pps,
                                                        jastrows       = [],
                                                        calculations   = [loop(max=2, qmc=linopt2)],
                                                        dependencies   = [(p2q, 'orbitals'),(opt_varmin, 'jastrow')]
                                                )
                                                sims.append(preopt_emin)
                                                emin_dep = preopt_emin
                                        else:
                                                emin_dep = opt_varmin
                                        #end if
                                        
                                        opt_emin = generate_qmcpack(
                                                identifier     = 'opt-emin',
                                                path           = './opt-u-'+str(u)+'-'+str(natoms) + '-emin-' + str(params.j),
                                                job            = opt_job,
                                                input_type     ='basic',
                                                system         = system,
                                                spin_polarized = True, # jtk: needs this
                                                meshfactor     = params.meshfactor,
                                                hybrid_rcut    = params.hybrid_rcut,
                                                hybrid_lmax    = params.hybrid_lmax,
                                                #spline_radius  = params.spline_radius,
                                                twistnum       = params.twistnum,
                                                #bconds         = 'ppp',
                                                pseudos        = params.qmc_pps,
                                                jastrows       = [],
                                                calculations   = [loop(max=4, qmc=linopt2), loop(max=4, qmc=linopt3)],
                                                dependencies   = [(p2q, 'orbitals'),(emin_dep, 'jastrow')]
                                        )
                                        sims.append(opt_emin)

                                        # 3-body jastrows
                                        
                                        if params.j3:
                                                j3_dep = []
                                                linopt2.walkers = 16
                                                linopt2.blocks = linopt2.blocks*2
                                                linopt2.samples = linopt1.samples*2

                                                j3_rcut=min(system['structure'].rwigner()-0.0001, 4.0)
                                                if not params.nopre:
                                                        preopt_emin_J3 = generate_qmcpack(
                                                                identifier     = 'opt-emin-J3',
                                                                path           = './preopt-u-'+str(u)+'-'+str(natoms)+ '-emin-J3-' + str(params.j),
                                                                job            = opt_job,
                                                                input_type     ='basic',
                                                                system         = system,
                                                                spin_polarized = True, # jtk: needs this
                                                                meshfactor     = params.meshfactor,
                                                                hybrid_rcut    = params.hybrid_rcut,
                                                                hybrid_lmax    = params.hybrid_lmax,
                                                                #spline_radius  = params.spline_radius,
                                                                twistnum       = params.twistnum,
                                                                #bconds         = 'ppp',
                                                                pseudos        = params.qmc_pps,
                                                                jastrows       = [('J3', 'polynomial', 3,3, j3_rcut)],
                                                                calculations   = [loop(max=2, qmc=linopt2)],
                                                                dependencies   = [(p2q, 'orbitals'),(opt_emin, 'jastrow')]
                                                        )
                                                        sims.append(preopt_emin_J3)
                                                        j3_dep = preopt_emin_J3
                                                else:
                                                        j3_dep = opt_emin
                                                #end if   

                                                opt_emin_J3 = generate_qmcpack(
                                                        identifier     = 'opt-emin-J3',
                                                        path           = './opt-u-'+str(u)+'-'+str(natoms)+ '-emin-J3-' + str(params.j),
                                                        job            = opt_job,
                                                        input_type     ='basic',
                                                        system         = system,
                                                        spin_polarized = True, # jtk: needs this
                                                        meshfactor     = params.meshfactor,
                                                        hybrid_rcut    = params.hybrid_rcut,
                                                        hybrid_lmax    = params.hybrid_lmax,
                                                        #spline_radius  = params.spline_radius,
                                                        twistnum       = params.twistnum,
                                                        #bconds         = 'ppp',
                                                        pseudos        = params.qmc_pps,
                                                        jastrows       = [('J3', 'polynomial', 3,3, j3_rcut)],
                                                        calculations   = [loop(max=4, qmc=linopt2), loop(max=4, qmc=linopt2)],
                                                        dependencies   = [(p2q, 'orbitals'),(j3_dep, 'jastrow')]
                                                )
                                                sims.append(opt_emin_J3)
                                        #end if
                                #end if

                                # VMC is complete, run DMC
                                
                                dmc_nodes = np.round(np.ceil(knum*dmcnodespertwist)/minnodes)*minnodes
				dmc_job   = Job(nodes=int(dmc_nodes), hours=dmc_hours, threads=qmc_threads,  app=qmcapp, presub=qmc_presub)

                                # Add any future estimators here
                                from qmcpack_input import spindensity,skall, density
                                if params.density is not None:
                                        est = [spindensity(grid=params.density)]
                                else:
                                        est = None
                                #end if
                                
                                #Initial VMC calculation to generate walkers for DMC
                                calculations    = [
                                                vmc(
                                                        warmupsteps = 25,
                                                        blocks  = vmcblocks,
                                                        steps = 1,
                                                        stepsbetweensamples=1,
                                                        walkers = walkers,
                                                        timestep=vmcdt,
                                                        substeps = 4,
                                                        samplesperthread = int(dmcwalkers/(dmcnodespertwist*walkers)),
                                                        ),
#                                                dmc(
#                                                        warmupsteps =0,
#                                                        blocks      =dmceqblocks/4,
#                                                        steps       =5,
#                                                        timestep    =0.04,
#                                                        nonlocalmoves=params.tmoves,
#                                                        ),
#                                                dmc(
#                                                        warmupsteps =0,
#                                                        blocks      =dmceqblocks/4,
#                                                        steps       =5,
#                                                        timestep    =0.02,
#                                                        nonlocalmoves=params.tmoves,
#                                                        ),
                                                ]
                                # Scan over timesteps
                                if params.timesteps is None:
                                        params.timesteps = [0.01]
                                #end if
                                for t in params.timesteps:
                                        calculations.append(dmc(warmupsteps =dmceqblocks/2,
                                                                blocks      =int(dmcblocks/(np.sqrt(t)*10)),
                                                                steps       =10,
                                                                timestep    =t,
                                                                nonlocalmoves=params.tmoves,
                                        ))
                                #end for

                                # Prepanding name for the path 
                                pathpre = 'dmc'
                                if params.vmc:
                                        pathpre = 'vmc'
                                #end if

                                deps = [(p2q, 'orbitals')]
                                
                                has_jastrow = False
                                # 3-body jastrow calculation with VMC or DMC
                                if params.j3:
                                        has_jastrow = True
                                        # add j3 dependenciies
                                        if params.vmc_opt:
                                                deps.append((opt_emin_J3, 'jastrow'))
                                        #end if

                                        if params.tmoves:
                                                path = './'+pathpre+'-j3-tm-u-'+str(u)+'-'+str(knum)+'-'+str(natoms)
                                        else:
                                                path = './'+pathpre+'-j3-nl-u-'+str(u)+'-'+str(knum)+'-'+str(natoms)
                                        #end if
                                #end if
                                # 2-body jastrow calculation with VMC or DMC
                                if params.j2:
                                        has_jastrow = True
                                        #add j2 dependencies
                                        if params.vmc_opt:
                                                deps.append((opt_emin, 'jastrow'))
                                        #end if
                                        if params.tmoves:
                                                path = './'+pathpre+'-j2-tm-u-'+str(u)+'-'+str(knum)+'-'+str(natoms)
                                        else:
                                                path = './'+pathpre+'-j2-nl-u-'+str(u)+'-'+str(knum)+'-'+str(natoms)
                                        #end if
                                #end if
                                # No jastrow calculation with VMC or DMC
                                if not has_jastrow:
                                        if params.tmoves:
                                                path = './'+pathpre+'-j0-tm-u-'+str(u)+'-'+str(knum)+'-'+str(natoms)
                                        else:
                                                path = './'+pathpre+'-j0-nl-u-'+str(u)+'-'+str(knum)+'-'+str(natoms)
                                        #end if
                                #end if
                                
                                det_format = 'new'
                                # Optical excitation
                                if params.excitation is not None:
                                        path += '_' + params.excitation[0] + '_'+ params.excitation[1].replace(" ", "_")
                                        det_format = 'old'
                                #end if
                                # Charged cell 
                                if params.charge != 0:
                                        path += '_' + str(params.charge)
                                #end if
                                
                                if params.qmc:
                                        qmc = generate_qmcpack(
                                                seed            = params.seed,
                                                skip_submit     = params.qmc_skip_submit,
                                                det_format      = det_format,
                                                identifier      = pathpre,
                                                path            = path,
                                                job             = dmc_job,
                                                input_type      = 'basic',
                                                system          = system,
                                                estimators      = est,
                                                meshfactor      = params.meshfactor,
                                                hybrid_rcut     = params.hybrid_rcut,
                                                hybrid_lmax     = params.hybrid_lmax,
                                                excitation      = params.excitation,
                                                pseudos         = params.qmc_pps,
                                                jastrows        = [],
                                                spin_polarized  = True,
                                                calculations    = calculations,
                                                dependencies    = deps
                                        )
                                        qmcs.append(qmc)
                                #end if
                        #end if
                #end for
        #end for
        if params.bundle:
                from bundle import bundle
                qmcb = bundle(qmcs)
                sims.append(qmcb)
        else:
                sims = sims + qmcs
                return sims
        #end if
#end def mworkflow
