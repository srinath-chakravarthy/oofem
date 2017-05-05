
#ifndef COMM_ERROR_H
#define COMM_ERROR_H

#include <mpi.h>

class comm_Error {
 public:
  comm_Error(MPI_Comm);

  void all(const char *);
  void one(const char *);
  void warning(const char *);

 private:
  MPI_Comm comm;
  int me;
};

#endif
