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
#include "dof.h"
#include "dofmanager.h"


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

using namespace oofem;

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
    std :: set_new_handler(freeStoreError);   // prevents memory overflow
#endif

    int adaptiveRestartFlag = 0, restartStepInfo [ 2 ];
    bool parallelFlag = true, renumberFlag = false, debugFlag = false, contextFlag = false, restartFlag = false,
         inputFileFlag = false, outputFileFlag = false, errOutputFileFlag = false, lammps_inputFileFlag = false;
    std :: stringstream inputFileName, outputFileName, errOutputFileName, lammps_input_file;
    int n_lammps_iter =0, n_fem_update =0 , n_lammps_procs=0;
    std :: vector< const char * >modulesArgs;
    EngngModel *problem = 0;
    
    // MPI related quantities
    int rank = 0;
    int world_size;
    int color; 
    int level;
    struct Fem_interface{
        std::map<int, int> bcMap;  // Map from bc number to node number
        int num_pad_atoms;         // Number of pad atoms
        int num_interface_atoms;   // Number of interface atoms = num manualboundarycondition
        std::vector<FloatArray> pad_atom_coords; 
        std::vector<int> pad_elem_map;
        std::vector<int> pad_atom_ids;
        std::vector<int> interface_atom_ids;
    };
    Fem_interface fem_interface;
    
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
        } else {
            color = 1;
            local_leader = 0;
            remote_leader = 0;
        }
        MPI_Comm_split(comm, color, 0, &split);
        MPI_Intercomm_create(split, local_leader, comm, remote_leader, 99, &intercomm);
    } else {
        split = MPI_COMM_WORLD;
    }
    
// Declare variables ??
    
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
            
            
            Domain *domain = problem->giveDomain(1);
            auto numDofman = domain->giveNumberOfDofManagers();
            for (int i=1; i<=numDofman; i++){
                auto dofman = domain->giveDofManager(i);
                auto nodenum = dofman->giveLabel();
                auto numdofs = dofman->giveNumberOfDofs();
                for (int idf = 1; idf <= numdofs; idf++){
                    auto dof = dofman->giveDofWithID(idf);
                    if (dof->hasBc(problem->giveCurrentStep())){
                        if(dof->giveBcId()){
                            fem_interface.bcMap[nodenum] = dof->giveBcId();
//                             OOFEM_LOG_INFO("Boundary Condition map %d %d\n",nodenum, dof->giveBcId());
                        }
                    }
                }
            }
//             return 0;
        } else {
            LAMMPS_NS::LAMMPS *lmp = new LAMMPS_NS::LAMMPS(0,NULL,split);
            lmp->input->file(lammps_input_file.str().c_str());
        }
        
    }
    if (world_size > 1){
        if (color == 0){
            
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
