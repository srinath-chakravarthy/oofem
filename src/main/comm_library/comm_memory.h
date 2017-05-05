#ifndef COMM_MEMORY_H
#define COMM_MEMORY_H

#include <mpi.h>

class comm_Memory {
 public:
    comm_Memory (MPI_Comm);
  ~comm_Memory();

  void *smalloc(int n, const char *);
  void sfree(void *);
  void *srealloc(void *, int n, const char *name);

  double **create_2d_double_array(int, int, const char *);
  double **grow_2d_double_array(double **, int, int, const char *);
  void destroy_2d_double_array(double **);

 private:
  class comm_Error *error;
};

#endif
