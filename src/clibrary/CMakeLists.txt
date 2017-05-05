set (clibrary
  comm_memory.C
  comm_error.C
  many2one.C
  )
add_library(clibrary
  ${LIB_TYPE}
  ${clibrary}
  )

