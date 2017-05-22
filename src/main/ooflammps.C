/*
 *
 *                 #####    #####   ######  ######  ###   ###
 *               ##   ##  ##   ##  ##      ##      ## ### ##
 *              ##   ##  ##   ##  ####    ####    ##  #  ##
 *             ##   ##  ##   ##  ##      ##      ##     ##
 *            ##   ##  ##   ##  ##      ##      ##     ##
 *            #####    #####   ##      ######  ##     ##
 *
 *
 *             OOFEM : Object Oriented Finite Element Code
 *
 *               Copyright (C) 1993 - 2013   Borek Patzak
 *
 *
 *
 *       Czech Technical University, Faculty of Civil Engineering,
 *   Department of Structural Mechanics, 166 29 Prague, Czech Republic
 *
 *  This library is free software; you can redistribute it and/or
 *  modify it under the terms of the GNU Lesser General Public
 *  License as published by the Free Software Foundation; either
 *  version 2.1 of the License, or (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 *  Lesser General Public License for more details.
 *
 *  You should have received a copy of the GNU Lesser General Public
 *  License along with this library; if not, write to the Free Software
 *  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
 */

//  MAIN
//  Solves finite element problems.
//
#ifdef __PYTHON_MODULE
 #include <Python.h>
#endif

#include "engngm.h"
#include "oofemcfg.h"

#include "oofemtxtdatareader.h"
#include "util.h"
#include "error.h"
#include "logger.h"
#include "contextioerr.h"
#include "oofem_terminate.h"
#include "domain.h"
#include "boundarycondition.h"
#include "generalboundarycondition.h"
#include "manualboundarycondition.h"
#include "spatiallocalizer.h"
#include "dof.h"
#include "dofmanager.h"
#include "element.h"
#include "floatarray.h"
#include "dofiditem.h"
#include "unknownnumberingscheme.h"
#include "metastep.h"

#ifdef __PARALLEL_MODE
 #include "dyncombuff.h"
#endif

#ifdef __PETSC_MODULE
 #include <petsc.h>
#endif

#ifdef __SLEPC_MODULE
 #include <slepceps.h>
#endif

#ifdef __OOFEG
 #include "oofeggraphiccontext.h"
#endif

#ifdef __LAMMPS
  #include "lammps.h"
#define LAMMPS_NS LAMMPS_NS
  #include "library.h"
  #include "input.h"
  #include "modify.h"
  #include "fix.h"
  #include "compute.h"
#endif 

#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <new>
#include <sstream>
// For passing PETSc/SLEPc arguments.
#include <fstream>
#include <iterator>
#include <map>
#include <vector>

#include "classfactory.h"
#include "many2one.h"
#include "comm_memory.h"
#include "timestep.h"


using namespace oofem;

struct Fem_interface{
    std::map<int, int> bcMap;  // Map from bc number to node number
    int num_pad_atoms;         // Number of pad atoms
    int num_interface_atoms;   // Number of interface atoms = num manualboundarycondition
    std::vector<FloatArray> pad_atom_coords; 
    std::vector<FloatArray> pad_elem_coords;
    std::vector<int> pad_elem_map;
    std::vector<int> pad_atom_ids;
    std::unordered_map<int, int> pad_atom_map; 
    std::vector<int> interface_atom_ids;
    std::unordered_map<int, int> interface_atom_map; 
    std::unordered_map<int, int> interface_atom_inverse_map; 
    
    void create_bcMap(EngngModel *problem){
        Domain *domain = problem->giveDomain ( 1 );
        auto numDofman = domain->giveNumberOfDofManagers();
        for ( int i=1; i<=numDofman; i++ ) {
            auto dofman = domain->giveDofManager ( i );
            auto nodenum = dofman->giveLabel();
            auto numdofs = dofman->giveNumberOfDofs();
            for ( int idf = 1; idf <= numdofs; idf++ ) {
                auto dof = dofman->giveDofWithID ( idf );
                if ( dof->hasBc ( problem->giveCurrentStep() ) ) {
                    if ( dof->giveBcId() ) {
                        bcMap[nodenum] = dof->giveBcId();
//                          OOFEM_LOG_INFO("Boundary Condition map %d %d\n",nodenum, dof->giveBcId());
                    }
                }
            }
        }
    }
    void create_pad_elem_map(EngngModel *problem){
        Domain *domain = problem->giveDomain ( 1 );
//         pad_elem_map.reserve(num_pad_atoms);
//         pad_atom_map.reserve(num_pad_atoms);
        for (int i = 0; i < num_pad_atoms; i++){
            pad_atom_map[pad_atom_ids[i]] = i;
            FloatArray coord(3);
            coord = pad_atom_coords[i];
            coord[2] = 0.0;
            auto elem = domain->giveSpatialLocalizer()->giveElementContainingPoint(coord, nullptr);
            pad_elem_map.push_back(elem->giveLabel());
            FloatArray lcoord;
            elem->computeLocalCoordinates(lcoord, coord);
            pad_elem_coords.push_back(lcoord);
        }
        
    }
    void create_interface_node_map(EngngModel *problem, double **interface_atom_coords)
    {
        interface_atom_map.reserve(num_interface_atoms);
        interface_atom_inverse_map.reserve(num_interface_atoms);
        Domain *domain = problem->giveDomain ( 1 );
        for (int i = 0; i<num_interface_atoms; i++){
            FloatArray coord(3);
            for (int j=0; j<3; j++){
                coord[j] = interface_atom_coords[i][j];
            }
            coord[2]  = 0.0;
            for (auto& p : bcMap){
                auto fem_coord = domain->giveDofManager(p.first)->giveCoordinates();
                FloatArray diff(3);
                diff[0] = fem_coord->at(1);
                diff[1] = fem_coord->at(2);
                diff[2] = 0.0;
                diff.subtract(coord);
                if (diff.computeSquaredNorm() < 1.0e-6){
                    interface_atom_map[interface_atom_ids[i]] = p.first;
                    interface_atom_inverse_map[p.first] = interface_atom_ids[i];
//                     fprintf(stderr, "Interface atom id = %d %d %s %d\n", i, interface_atom_ids[i], " Node num = ",interface_atom_map[interface_atom_ids[i]]);
                }
            }
        }
    }
    void print_interface_atom_maps()
    {
        for (int i = 0; i< num_interface_atoms; i++){
            fprintf(stderr, "Interface atom ids = %d %d\n", i, interface_atom_ids[i]);
        }
    }
    
    void print_pad_atom_maps()
    {
        for (int i = 0; i< num_pad_atoms; i++)
        {
            int id2 = pad_atom_ids[i];
            int fem_pad_id = pad_atom_map[id2];
            fprintf(stdout, "Pad atom map %d %d %d %d [%f %f] [%f %f]\n", i, id2, fem_pad_id,pad_elem_map[fem_pad_id],
                    pad_atom_coords[fem_pad_id][0], pad_atom_coords[fem_pad_id][1],
                    pad_elem_coords[fem_pad_id][0], pad_elem_coords[fem_pad_id][1]);
        }
    }
    
    void get_pad_atom_coords(EngngModel *problem, int *pad_atom_ids, double **coords)
    {
        Domain *domain = problem->giveDomain ( 1 );
        
        for (int i = 0; i < num_pad_atoms; i++)
        {
            int id2 = pad_atom_ids[i];
            int fem_pad_id = pad_atom_map[id2];
            FloatArray coord = pad_atom_coords[fem_pad_id];
            FloatArray lcoord = pad_elem_coords[fem_pad_id];
            auto elem = domain->giveElement(pad_elem_map[fem_pad_id]);
            ValueModeType mode = VM_Total;
            FloatArray displacement;
            elem->computeField(mode,problem->giveCurrentStep(), lcoord, displacement);
            for (int j =0; j<2; j++ )
            {
                coords[i][j] = displacement[j] + coord[j];
            }
//             fprintf(stdout, "Pad_atom_coords %d %d %d [%f %f] [%f %f] [%f %f ]\n",
//                 i, id2, fem_pad_id, coord[0], coord[1], displacement[0], displacement[1],
//                 coords[i][0], coords[i][1]);
        }
    }
    void apply_fem_displacement(EngngModel *problem, int *interface_ids, double **interface_displacements)
    {
        Domain *domain = problem->giveDomain ( 1 );
        for (int i=0; i < num_interface_atoms; i++)
        {
            int fem_node_num = interface_atom_map[interface_ids[i]];
//             fprintf(stderr, "Interface atoms = %d %d %d \n",i, interface_ids[i], interface_atom_map[interface_ids[i]] );
            auto dofman = domain->giveDofManager(fem_node_num);
            GeneralBoundaryCondition * bc = domain->giveBc(bcMap[fem_node_num]);
            if (ManualBoundaryCondition * manbc = dynamic_cast<ManualBoundaryCondition *>(bc)){
//                 fprintf(stdout, "Applying_fem_displacement %d %d " , interface_ids[i], fem_node_num);
//             if(manbc == nullptr || manbc->giveType() != DirichletBT) { continue; }
                auto dofs = bc->giveDofIDs();
                double setbc;
                for (auto iddof : dofs){
                    auto dof = dofman->giveDofWithID(iddof);
                    if (dof->giveDofID() == D_u)
                        setbc = interface_displacements[i][0];
                    else if (dof->giveDofID() == D_v)
                        setbc = interface_displacements[i][1];
                    else
                        setbc = 0.0;
//                     fprintf(stdout, " %d %f ", iddof, setbc);
                    manbc->setManualValue(dof, setbc);
                }
//                 fprintf(stdout, "\n");
            }
        }
    }
    
};
    


void freeStoreError()
// This function is called whenever operator "new" is unable to allocate memory.
{
    OOFEM_FATAL("free store exhausted");
}

// debug
void oofem_debug(EngngModel *emodel);

void oofem_print_help();
void oofem_print_version();
void oofem_print_epilog();

// Finalize PETSc, SLEPc and MPI
void oofem_finalize_modules();

int main(int argc, char *argv[])
{
#ifndef _MSC_VER
    std :: set_new_handler(freeStoreError );   // prevents memory overflow
#endif

    int adaptiveRestartFlag = 0, restartStepInfo [ 2 ];
    bool parallelFlag = true, renumberFlag = false, debugFlag = false, contextFlag = false, restartFlag = false,
         inputFileFlag = false, outputFileFlag = false, errOutputFileFlag = false, lammps_inputFileFlag = false;
    std :: stringstream inputFileName, outputFileName, errOutputFileName, lammps_input_file;
    int n_lammps_iter =0, n_fem_update =0 , n_lammps_procs=0;
    std :: vector< const char * >modulesArgs;
    EngngModel *problem = 0;
    LAMMPS_NS::LAMMPS *lmp = 0;
    
    // MPI related quantities
    int rank = 0, local_rank;
    int world_size;
    int color; 
    int level;
    int rootproc;
    

    // Lammps variables common
    int nlocal_pad, nlocal_inter;
    int *local_pad_id; 
    int *local_interface_id; 
    int total_fem_steps;
    double **local_pad_coords;
    double **local_interface_coords;
    double **pad_atom_coords = NULL;
    double **interface_atom_coords = NULL;
    double **local_interface_disp; 
    double **interface_displacements;
    Fem_interface fem_interface;
    
    int total_lammps_steps; 
    
#ifdef __PARALLEL_MODE
 #ifdef __USE_MPI
    MPI_Init(& argc, & argv);
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Comm_rank(comm, & rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm split, intercomm; 
    int local_leader, remote_leader;
    
    //oofem_logger.setComm(MPI_COMM_WORLD);
 #else
    fprintf(stderr, "\n Must be compiled with MPI support to run coupled lammps\a\n\n");
    exit(EXIT_FAILURE);
 #endif
#endif

    
    //
    // check for options
    //
    if ( argc != 1 ) {
        // argv[0] is not read by PETSc and SLEPc.
        modulesArgs.push_back(argv [ 0 ]);
        for ( int i = 1; i < argc; i++ ) {
//             fprintf(stderr, "\n Arguments %d %d %s \a\n\n", argc, i, argv[i]);
            if ( ( strcmp(argv [ i ], "-context") == 0 ) || ( strcmp(argv [ i ], "-c") == 0 ) ) {
                contextFlag = true;
            } else if ( strcmp(argv [ i ], "-f") == 0 ) {
                if ( i + 1 < argc ) {
                    i++;
                    inputFileName << argv [ i ];
                    inputFileFlag = true;
                }
            } else if ( strcmp(argv [ i ], "-r") == 0 ) {
                if ( i + 1 < argc ) {
                    i++;
                    restartFlag = true;
                    restartStepInfo [ 0 ] = strtol(argv [ i ], NULL, 10);
                    restartStepInfo [ 1 ] = 0;
                }
            } else if ( strcmp(argv [ i ], "-rn") == 0 ) {
                renumberFlag = true;
            } else if ( strcmp(argv [ i ], "-ar") == 0 ) {
                if ( i + 1 < argc ) {
                    i++;
                    adaptiveRestartFlag = strtol(argv [ i ], NULL, 10);
                }
            } else if ( strcmp(argv [ i ], "-l") == 0 ) {
                if ( i + 1 < argc ) {
                    i++;
                    level = strtol(argv [ i ], NULL, 10);
                }
            } else if ( strcmp(argv [ i ], "-qe") == 0 ) {
                if ( i + 1 < argc ) {
                    i++;
                    errOutputFileFlag = true;
                    errOutputFileName << argv [ i ];
                    i++;
                }
            } else if ( strcmp(argv [ i ], "-qo") == 0 ) {
                if ( i + 1 < argc ) {
                    i++;
                    outputFileFlag = true;
                    outputFileName << argv [ i ];
                }
            } else if ( strcmp(argv [ i ], "-s") == 0 ) {
//                 fprintf(stderr, "\n Lammps %d %d \a\n\n", i+4, argc);
                if (i + 4 < argc ) {
                    i++; 
                    lammps_input_file << argv [i];
                    lammps_inputFileFlag = true;
//                     fprintf(stderr, "\n Lammps %s \a\n\n", lammps_input_file.str().c_str());
                    i++;
                    n_lammps_iter = strtol(argv [ i ], NULL, 10);
//                     fprintf(stderr, "\n Lammps %d \a\n\n", n_lammps_iter);
                    i++;
                    n_fem_update = strtol(argv [ i ], NULL, 10);
//                     fprintf(stderr, "\n Lammps %d \a\n\n", n_fem_update);
                    i++;
                    n_lammps_procs = strtol(argv [ i ], NULL, 10);
//                     fprintf(stderr, "\n Lammps %d \a\n\n", n_lammps_procs);
                } else {
                    fprintf(stderr, "\nNeed atleast 4 arguments after -c Couple flag \a\n\n");
                    exit(EXIT_FAILURE);
                }
            } else if ( strcmp(argv [ i ], "-d") == 0 ) {
                debugFlag = true;
            } else { // Arguments not handled by OOFEM is to be passed to PETSc
                modulesArgs.push_back(argv [ i ]);
            }
        }
    } else {
        if ( rank == 0 ) {
            oofem_print_help();
        }
        MPI_Finalize();
        exit(EXIT_SUCCESS);
    }

    // check if input file given
    if ( !inputFileFlag ) {
        if ( rank == 0 ) {
            fprintf(stderr, "\nInput file not specified\a\n\n");
        }
        MPI_Finalize();
        exit(EXIT_FAILURE);
    }
    // Check if lammps input file is given 
//     if (!lammps_inputFileFlag) {
//         if (rank == 0){
//             fprintf(stderr, "\nLammps Input file not specified\a\n\n");
//         }
//         MPI_Finalize();
//         exit(EXIT_FAILURE);        
//     }
    // Check if number of lammps procs is less than total number of procs
    // Check if lammps procs is even
    
    bool proccheck = true;
    if (world_size == 1) {
        n_lammps_procs = 1; // Override setting from input runs single processor mode
    } else if (n_lammps_procs != world_size -1){
        proccheck = false;
        if (rank == 0) {
            fprintf(stderr, "\nNumber of lammps processors has to be = total -1 %d\a\n\n", n_lammps_procs);
        }
    } else if (n_lammps_procs % 2 != 0) {
        proccheck = false;
        if (rank == 0) {
            fprintf(stderr, "\nNumber of lammps processors has to be an even number\a\n\n");
        }            
    }
    // Now Finalize MPI for the above error
    if (!proccheck) {
        MPI_Finalize();
        exit(EXIT_FAILURE);
    }
    
#if defined ( __PETSC_MODULE ) || defined ( __SLEPC_MODULE )
    int modulesArgc = modulesArgs.size();
    char **modulesArgv = const_cast< char ** >(& modulesArgs [ 0 ]);
#endif

#ifdef __PETSC_MODULE
            PetscInitialize(& modulesArgc, & modulesArgv, PETSC_NULL, PETSC_NULL);
#endif


#ifdef __SLEPC_MODULE
    SlepcInitialize(& modulesArgc, & modulesArgv, PETSC_NULL, PETSC_NULL);
#endif

#ifdef __PYTHON_MODULE
    Py_Initialize();
    // Adding . to the system path allows us to run Python functions stored in the working directory.
    PyRun_SimpleString("import sys");
    PyRun_SimpleString("sys.path.append(\".\")");
#endif
    // Add variables for broadcast
    int root, myrank;
    
    if (world_size > 1){
        if (rank < 1){
            color = 0;
            local_leader = 0;
            remote_leader = 1;
            rootproc = MPI_ROOT;
        } else {
            color = 1;
            local_leader = 0;
            remote_leader = 0;
            rootproc = remote_leader;
        }
        MPI_Comm_split(comm, color, 0, &split);
        MPI_Intercomm_create(split, local_leader, comm, remote_leader, 99, &intercomm);
    } else {
        split = MPI_COMM_WORLD;
    }
    MPI_Comm_rank(split, &local_rank);
    
    comm_Memory *memory = new comm_Memory(split);
    
    Many2One *lmp_fem_pad = new Many2One(intercomm, rootproc, local_leader, remote_leader);   
    Many2One *lmp_fem_inter = new Many2One(intercomm, rootproc, local_leader, remote_leader);
    
// Declare variables ??
    
    MPI_Errhandler_set(intercomm, MPI_ERRORS_RETURN);
    
    MPI_Comm_rank(split, &myrank);
    if (world_size == 1){
        
    } else {
        if (color == 0){
            
            fprintf(stderr, "\nStarting solution \a\n\n");
            oofem_logger.setComm(split);
            oofem_logger.setLogLevel(level);
            OOFEM_LOG_FORCED(PRG_HEADER_SM);
            OOFEMTXTDataReader dr( inputFileName.str ( ).c_str() );
            Setutil_comm(split);
            problem = :: InstanciateProblem(& dr, _processor, contextFlag, NULL, parallelFlag);
            dr.finish();
            if ( !problem ) {
                OOFEM_LOG_ERROR("Couldn't instanciate problem, exiting");
                exit(EXIT_FAILURE);
            }           
            problem->checkProblemConsistency();
            problem->init();
            problem->postInitialize();
            
            // Check that number of metasteps is exactly equal to 1
            if (problem->giveNumberOfMetaSteps() > 1){
                oofem_finalize_modules();
                OOFEM_ERROR("Cannot have more than 1 (default) metasteps")
                MPI_Finalize();
            }
            total_lammps_steps = problem->giveNumberOfSteps()*(n_lammps_iter/n_fem_update);
            MPI_Bcast(&total_lammps_steps, 1, MPI_INT, rootproc, intercomm);
            
            fem_interface.create_bcMap(problem);
            int ndest; 
            lmp_fem_pad->setup(nlocal_pad, local_pad_id, ndest);
            lmp_fem_inter->setup(nlocal_inter, local_interface_id, ndest);
            
            fem_interface.num_pad_atoms = lmp_fem_pad->get_nall();            
            fem_interface.pad_atom_ids.reserve(fem_interface.num_pad_atoms);
            std::copy(&lmp_fem_pad->get_idall()[0],&lmp_fem_pad->get_idall()[fem_interface.num_pad_atoms], std::back_inserter(fem_interface.pad_atom_ids));
            
            fem_interface.num_interface_atoms = lmp_fem_inter->get_nall();
            fem_interface.interface_atom_ids.reserve(fem_interface.num_interface_atoms);
            std::copy(&lmp_fem_inter->get_idall()[0],&lmp_fem_inter->get_idall()[fem_interface.num_interface_atoms], std::back_inserter(fem_interface.interface_atom_ids));
            
            
            // Create map for 
            fprintf(stderr, "Number of pad atoms = %d\n", fem_interface.num_pad_atoms);
            fprintf(stderr, "Number of interface atoms = %d\n", fem_interface.num_interface_atoms);
            
            pad_atom_coords = memory->create_2d_double_array(fem_interface.num_pad_atoms,3,"pad_atom_coords:fem");
            double *local_pad;
            lmp_fem_pad->gather(local_pad,3,&pad_atom_coords[0][0],false);
            
            
            double *local_inter;
            interface_atom_coords = memory->create_2d_double_array(fem_interface.num_interface_atoms,3,"interface_atom_coords:fem");
            lmp_fem_inter->gather(local_inter,3,&interface_atom_coords[0][0],false);
            
            
//             fem_interface.pad_atom_coords.reserve(fem_interface.num_pad_atoms);
            FloatArray coord(3); 

            for (int i = 0; i < fem_interface.num_pad_atoms; i++){
                for (int j=0; j<3; j++){
                    coord.at(j+1) = pad_atom_coords[i][j];
                }
                fem_interface.pad_atom_coords.push_back(coord);
            }
            fem_interface.create_pad_elem_map(problem);
            fem_interface.create_interface_node_map(problem, interface_atom_coords);
            
            memory->destroy_2d_double_array(pad_atom_coords);
            memory->destroy_2d_double_array(interface_atom_coords);
//             fem_interface.print_interface_atom_maps();

//             return 0;
        } else {
            lmp = new LAMMPS_NS::LAMMPS(0,NULL,split);
            int nlocal_lammps;
            int *id_lammps;
            int *type_lammps;
            lmp->input->file(lammps_input_file.str().c_str());
            nlocal_lammps = *((int *) lammps_extract_global(lmp,"nlocal"));
            id_lammps = (int *) lammps_extract_atom(lmp,"id");
            type_lammps = (int *) lammps_extract_atom(lmp,"type");
            double **x = (double **) lammps_extract_atom(lmp,"x");
            MPI_Bcast(&total_lammps_steps, 1, MPI_INT, rootproc, intercomm);
            
            
            nlocal_pad = 0;
            nlocal_inter = 0;
            
            for (int i = 0; i< nlocal_lammps; i++){
                if (type_lammps[i] == 2){
                    nlocal_pad++;
                } else if (type_lammps[i] == 3){
                    nlocal_inter++;
                }
            }
            fprintf(stderr, "Number of pad atoms = %d %d\n", local_rank, nlocal_pad);
            fprintf(stderr, "Number of interface atoms = %d %d\n", local_rank, nlocal_inter);
            local_pad_id  = (int *) memory->smalloc(nlocal_pad*sizeof(int),"lmp_fem_pad:padid");
            local_interface_id  = (int *) memory->smalloc(nlocal_inter*sizeof(int),"lmp_fem_inter:interfaceid");
            
            if (nlocal_pad > 0)
                local_pad_coords = memory->create_2d_double_array(nlocal_pad,3,"lmp_pad_coords:fem_interface");
            if (nlocal_inter > 0)
                local_interface_coords = memory->create_2d_double_array(nlocal_inter,3,"lmp_interface_coords:fem_interface");
            
            int loc = 0; 
            int loc2 = 0;
            for (int i = 0; i< nlocal_lammps; i++){
                if (type_lammps[i] == 2){
                    for (int j=0;j<3; j++){
                        local_pad_coords[loc][j] = x[i][j];
                        
                    }
                    local_pad_id[loc++] = id_lammps[i];
//                     fprintf(stderr,"Local pad atom coords are %d %f %f %f\n", 
//                             local_pad_id[loc-1], local_pad_coords[loc-1][0],local_pad_coords[loc-1][1],local_pad_coords[loc-1][2] );
                } else if (type_lammps[i] == 3){
                    for (int j=0; j<3; j++) {                        
                        local_interface_coords[loc2][j] = x[i][j];
                    }
                    local_interface_id[loc2++] = id_lammps[i];
//                     fprintf(stderr,"Local interface atom coords are %d %f %f %f\n", 
//                             local_interface_id[loc2-1], local_interface_coords[loc2-1][0],local_interface_coords[loc2-1][1],local_interface_coords[loc2-1][2] );
                    
                }
            }
//             
            int ndest;
            lmp_fem_pad->setup(nlocal_pad, local_pad_id, ndest);
            lmp_fem_inter->setup(nlocal_inter, local_interface_id, ndest);
            if (nlocal_pad > 0)
                lmp_fem_pad->gather(&local_pad_coords[0][0],3,NULL, false);
            else
                lmp_fem_pad->gather(NULL,3,NULL, false);
            if (nlocal_inter > 0)
                lmp_fem_inter->gather(&local_interface_coords[0][0],3,NULL, false);
            else
                lmp_fem_inter->gather(NULL,3,NULL, false);
            
            if (nlocal_pad > 0)
                memory->destroy_2d_double_array(local_pad_coords);
            if (nlocal_inter > 0)
                memory->destroy_2d_double_array(local_interface_coords);
            memory->sfree(local_interface_id);
            memory->sfree(local_pad_id);
        }
        
    }
    if (world_size > 1)
    {
        if (color == 0){
            pad_atom_coords = memory->create_2d_double_array(fem_interface.num_pad_atoms,3,"pad_atom_coords:fem");
            interface_displacements = memory->create_2d_double_array(fem_interface.num_interface_atoms,3,"pad_atom_coords:fem");
            int smstep = 1, sjstep = 1;


            if ( problem->giveCurrentStep() ) {
                smstep = problem->giveCurrentStep()->giveMetaStepNumber();
                sjstep = problem->giveMetaStep(smstep)->giveStepRelativeNumber( problem->giveCurrentStep()->giveNumber() ) + 1;
            }

            for ( int imstep = smstep; imstep <= problem->giveNumberOfMetaSteps(); imstep++, sjstep = 1 ) { //loop over meta steps
                MetaStep *activeMStep = problem->giveMetaStep(imstep);
                // update state according to new meta step
                problem->initMetaStepAttributes(activeMStep);

                int nTimeSteps = activeMStep->giveNumberOfSteps();
                for ( int jstep = sjstep; jstep <= nTimeSteps; jstep++ ) { //loop over time steps
                    problem->preInitializeNextStep();
                    problem->giveNextStep();

                    // renumber equations if necessary. Ensure to call forceEquationNumbering() for staggered problems
                    if ( problem->requiresEquationRenumbering( problem->giveCurrentStep() ) ) {
                        problem->forceEquationNumbering();
                    }

                    OOFEM_LOG_DEBUG("Number of equations %d\n", problem->giveNumberOfDomainEquations( 1, EModelDefaultEquationNumbering()) );

                    problem->initializeYourself( problem->giveCurrentStep() );
                    int lammps_loop = n_lammps_iter/n_fem_update;
                    for (int i=0; i<lammps_loop; i++)
                    {
                        int ndest; 
                        lmp_fem_pad->setup(nlocal_pad, local_pad_id, ndest);
                        int *idall = lmp_fem_pad->get_idall();
                        fem_interface.get_pad_atom_coords(problem, idall, pad_atom_coords);
                        lmp_fem_pad->scatter(&pad_atom_coords[0][0],3,NULL);
                        
                        
                        lmp_fem_inter->setup(nlocal_inter, local_interface_id, ndest);
                        double *local_inter_disp; 
                        
                        lmp_fem_inter->gather(local_inter_disp, 3, &interface_displacements[0][0]);
                        int *idall_inter = lmp_fem_inter->get_idall();
                        fem_interface.apply_fem_displacement(problem, idall_inter, interface_displacements);
                        problem->solveYourselfAt(problem->giveCurrentStep());
                        problem->updateYourself(problem->giveCurrentStep());
                    }
                    problem->terminate(problem->giveCurrentStep());
                }
            }
            memory->destroy_2d_double_array(pad_atom_coords);
            memory->destroy_2d_double_array(interface_displacements);
        } else 
          {
            bool lammps_equil = true;
            for (int i = 0; i< total_lammps_steps; i++)
            {
                int nlocal_lammps;
                int *id_lammps;
                int *type_lammps;
                nlocal_lammps = *((int *) lammps_extract_global(lmp,"nlocal"));
                id_lammps = (int *) lammps_extract_atom(lmp,"id");
                type_lammps = (int *) lammps_extract_atom(lmp,"type");
                double **x = (double **) lammps_extract_atom(lmp,"x");
                int nstep = *((int *) lammps_extract_global(lmp,"ntimestep"));
                
                if(nstep >= 2000 && lammps_equil){
                    lmp->input->one("velocity particle_atoms set NULL -20.0 NULL sum yes units box");
                    lammps_equil = false;
                }
                
                nlocal_pad = 0;
                nlocal_inter = 0;
                
                for (int i = 0; i<= nlocal_lammps; i++){
                    if (type_lammps[i] == 2){
                        nlocal_pad++;
                    } else if (type_lammps[i] == 3){
                        nlocal_inter++;
                    }
                }
    //             fprintf(stderr, "Number of interface atoms = %d\n", nlocal_inter);
                local_pad_id  = (int *) memory->smalloc(nlocal_pad*sizeof(int),"lmp_fem_pad:padid");
                int loc = 0; 
                for (int i = 0; i< nlocal_lammps; i++){
                    if (type_lammps[i] == 2)
                        local_pad_id[loc++] = id_lammps[i];
                }
                int ndest;
                lmp_fem_pad->setup(nlocal_pad, local_pad_id, ndest);
                
                if (nlocal_pad > 0)
                    local_pad_coords = memory->create_2d_double_array(nlocal_pad,3,"lmp_pad_coords:fem_interface");
                double *pad_coords;
                if (nlocal_pad > 0)
                    lmp_fem_pad->scatter(pad_coords,3,&local_pad_coords[0][0]);
                else
                    lmp_fem_pad->scatter(pad_coords,3,NULL);
                loc = 0; 
                for (int i = 0; i< nlocal_lammps; i++)
                {
                    if (type_lammps[i] == 2)
                    {
//                     fprintf(stderr, "Pad atom coords = %d %d [%f %f %f] ",i, id_lammps[i],x[i][0], x[i][1], x[i][2]);
                        for (int j=0; j<2; j++)
                        {
                            x[i][j] = local_pad_coords[loc][j];
                        }
//                     fprintf(stderr, " [%f %f %f] \n", x[i][0], x[i][1], x[i][2]);
                        loc++;
                    }
                }
                if (nlocal_pad > 0)
                    memory->destroy_2d_double_array(local_pad_coords);
                memory->sfree(local_pad_id);
                lmp->input->one("run 20 pre yes post no");
                
                
                nlocal_lammps = *((int *) lammps_extract_global(lmp,"nlocal"));
                id_lammps = (int *) lammps_extract_atom(lmp,"id");
                type_lammps = (int *) lammps_extract_atom(lmp,"type");
                
                loc = 0;
                nlocal_inter = 0;
                for (int i = 0; i< nlocal_lammps; i++){
                    if (type_lammps[i] == 3){
                        nlocal_inter ++; 
                    }
                }
//                 std::unordered_map<int, int> md_interface_dict;
                local_interface_id  = (int *) memory->smalloc(nlocal_inter*sizeof(int),"lmp_fem_inter:interfaceid");
                if (nlocal_inter > 0){
                    local_interface_disp = memory->create_2d_double_array(nlocal_inter,3,"lmp_interface_disp:fem_interface");
//                     md_interface_dict.reserve(nlocal_inter);
                }

                
                double **disp = (double **) lammps_extract_fix(lmp,"disp_ave",1,2,1,1);
                
                loc = 0;
                for (int i = 0; i< nlocal_lammps; i++){
                    if (type_lammps[i] == 3){
                        local_interface_id[loc] = id_lammps[i];
//                         md_interface_dict[id_lammps[i]] = i;
                        for (int j = 0; j<3; j++)
                            local_interface_disp[loc][j] = disp[i][j];
                        loc ++;
                    }
                }
//                 for (int i = 0; i < nlocal_inter; i++){
//                     int interfaceid = md_interface_dict[local_interface_id[i]];
//                     for (int j = 0; j<3; j++){
//                         local_interface_disp[i][j] = disp[interfaceid][j];
//                     }
//                 }
                
                lmp_fem_inter->setup(nlocal_inter, local_interface_id, ndest);
                MPI_Barrier(split);
//                 double *interface_disp; 
                if (nlocal_inter > 0) {
                    lmp_fem_inter->gather(&local_interface_disp[0][0], 3, NULL);
                    memory->destroy_2d_double_array(local_interface_disp);
                }
                else
                    lmp_fem_inter->gather(NULL, 3, NULL);
                
                memory->sfree(local_interface_id);
            }
        }
    }
    
    MPI_Finalize();
#ifdef __SLEPC_MODULE
    SlepcFinalize();
#endif
    
}

 

void oofem_print_help()
{
    printf("\nOptions:\n\n");
    printf("  -v  prints oofem version\n");
    printf("  -f  (string) input file name\n");
    printf("  -r  (int) restarts analysis from given step\n");
    printf("  -ar (int) restarts adaptive analysis from given step\n");
    printf("  -l  (int) sets treshold for log messages (Errors=0, Warnings=1,\n");
    printf("            Relevant=2, Info=3, Debug=4)\n");
    printf("  -rn turns on renumbering\n");
    printf("  -qo (string) redirects the standard output stream to given file\n");
    printf("  -qe (string) redirects the standard error stream to given file\n");
    printf("  -c  creates context file for each solution step\n");
    printf("\n");
    oofem_print_epilog();
}

#ifndef HOST_TYPE
 #define HOST_TYPE "unknown"
#endif

void oofem_print_version()
{
    printf("\n%s (%s, %s)\nof %s on %s\n", PRG_VERSION, HOST_TYPE, MODULE_LIST, __DATE__, HOST_NAME);
    oofem_print_epilog();
}

void oofem_print_epilog()
{
    printf("\n%s\n", OOFEM_COPYRIGHT);
    printf("This is free software; see the source for copying conditions.  There is NO\n");
    printf("warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.\n\n");
}

void oofem_finalize_modules()
{
#ifdef __PETSC_MODULE
    PetscFinalize();
#endif

// #ifdef __SLEPC_MODULE
//     SlepcFinalize();
// #endif

// #ifdef __USE_MPI
//     MPI_Finalize();
// #endif

#ifdef __PYTHON_MODULE
    Py_Finalize();
#endif
}

//#include "loadbalancer.h"
//#include "xfem/iga.h"

void oofem_debug(EngngModel *emodel)
{
    //FloatMatrix k;
    //((BsplinePlaneStressElement*)emodel->giveDomain(1)->giveElement(1))->giveCharacteristicMatrix(k, StiffnessMatrix, NULL);

#ifdef __PARALLEL_MODE
    //LoadBalancer* lb = emodel->giveDomain(1)->giveLoadBalancer();
    //lb->calculateLoadTransfer();
    //lb->migrateLoad();
    //exit(1);
#endif
}

// Empty functions just so that we can link to the library even with oofeg compilation.
#ifdef __OOFEG
void ESICustomize(Widget parent_pane) { }
oofegGraphicContext gc [ OOFEG_LAST_LAYER ];
EView *myview;
void deleteLayerGraphics(int iLayer) { }
#endif
