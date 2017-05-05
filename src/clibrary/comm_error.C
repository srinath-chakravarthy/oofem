#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include "comm_error.h"

/* ---------------------------------------------------------------------- */

comm_Error::comm_Error(MPI_Comm caller)
{
  comm = caller;
  MPI_Comm_rank(comm,&me);
}

/* ----------------------------------------------------------------------
   called by all procs
------------------------------------------------------------------------- */

void comm_Error::all(const char *str)
{
  if (me == 0) printf("ERROR: %s\n",str);
  MPI_Finalize();
  exit(1);
}

/* ----------------------------------------------------------------------
   called by one proc
------------------------------------------------------------------------- */

void comm_Error::one(const char *str)
{
  printf("ERROR on proc %d: %s\n",me,str);
  MPI_Abort(comm,1);
}

/* ----------------------------------------------------------------------
   called by one proc
------------------------------------------------------------------------- */

void comm_Error::warning(const char *str)
{
  printf("WARNING: %s\n",str);
}

