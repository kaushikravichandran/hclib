#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <exstack.h>
extern "C" {
#include "spmat.h"
}

#define NV_THREADS nvshmem_n_pes()
#define NV_MYTHREAD nvshmem_my_pe()

void nvshmem_write_upc_array_int64(__shared__ int64_t *addr, size_t index, size_t blocksize, int64_t val) {
  int pe;
  size_t local_index;
  int64_t *local_ptr;


  pe = index % nvshmem_n_pes();
  local_index = (index / nvshmem_n_pes())*blocksize;

  local_ptr =(int64_t*)(( (char*)addr ) + local_index);

  nvshmem_int_p ( local_ptr, val, pe );
}

/*! \brief Read a sparse matrix in matrix market format on one PE and create a distributed matrix
  from that.
  * Only PE 0 reads the matrix file.
  * 
  * \param name The name of the file.
  * \return The sparsemat_t struct.
  * \ingroup spmatgrp
  */
sparsemat_t * read_matrix_mm_to_dist_nv(char * name) {
  typedef struct pkg_rowcol_t{
    int64_t row;    
    int64_t col;
  }pkg_rowcol_t;

  int64_t nr, nc, nnz = 0, i, pe;
  __shared__ int64_t * sh_data;
  sh_data = nvshmem_calloc (THREADS*4, sizeof(int64_t));

  int64_t * rowcount;
  edge_t * edges;
  w_edge_t * tri;
  if(!MYTHREAD){
    int fscanfret;
    int64_t * nnz_per_th = calloc(THREADS, sizeof(int64_t));
    
    FILE * fp = fopen(name, "r");
    if( fp == NULL ) {
      fprintf(stderr,"read_matrix_mm: can't open file %s \n", name);
      nvshmem_global_exit(1);
    }
    
    // Read the header line of the MatrixMarket format 
    char * object = calloc(64, sizeof(char));
    char * format = calloc(64, sizeof(char));
    char * field = calloc(64, sizeof(char));;
    fscanfret = fscanf(fp,"%%%%MatrixMarket %s %s %s\n", object, format, field);
    if( (fscanfret != 3 ) || strncmp(object,"matrix",24) || strncmp(format,"coordinate",24) ){
      fprintf(stderr,"read_matrix_mm: Incompatible matrix market format.\n");
      fprintf(stderr,"                First line should be either:\n");
      fprintf(stderr,"                matrix coordinate pattern\n");
      fprintf(stderr,"                OR\n");
      fprintf(stderr,"                matrix coordinate real\n");
      fprintf(stderr,"                OR\n");
      fprintf(stderr,"                matrix coordinate integer\n");
      nvshmem_global_exit(1);
    }

    // Make sure that this is a format we support
    if(strncmp(field,"pattern",24) && strncmp(field,"real",24) && strncmp(field,"integer",24) ){
      fprintf(stderr,"read_matrix_mm: Incompatible matrix market field.\n");
      fprintf(stderr,"                Last entry on first line should be pattern, real, or integer\n");
      nvshmem_global_exit(1);
    }
    int64_t values;
    if(strncmp(field,"pattern",7) == 0){
      values = 0L; // no values
    }else if(strncmp(field,"real",4) == 0){
      values = 1L; // real values
    }else{
      values = 2L; // integer values
    }
    
    // Read the header (nr, nc, nnz)
    fscanfret = fscanf(fp,"%"PRId64" %"PRId64" %"PRId64"\n", &nr, &nc, &nnz);
    if( (fscanfret != 3 ) || (nr<=0) || (nc<=0) || (nnz<=0) ) {
      fprintf(stderr,"read_matrix_mm: reading nr, nc, nnz\n");
      nvshmem_global_exit(1);
    }

    // allocate space to store the matrix data    
    rowcount = calloc(nr, sizeof(int64_t));
    if(!rowcount){
      T0_printf("ERROR: read_matrix_mm_to_dist: could not allocate arrays\n");
      for(i = 0; i < THREADS; i++) nvshmem_write_upc_array_int64(sh_data, i, sizeof(int64_t), -1);
    }
    
    // read the data
    int64_t row, col, val, pos = 0;
    if(values == 0){
      edges = calloc(nnz, sizeof(edge_t));
      while(fscanf(fp,"%"PRId64" %"PRId64"\n", &row, &col) != EOF){
        row--;//MM format is 1-up
        col--;
        edges[pos].row   = row;
        edges[pos++].col = col;
        nnz_per_th[row % THREADS]++;
        rowcount[row]++;
      }
      qsort( edges, nnz, sizeof(edge_t), edge_comp);
    }else{
      tri = calloc(nnz, sizeof(w_edge_t));    
      while(fscanf(fp,"%"PRId64" %"PRId64" %"PRId64"\n", &row, &col, &val) != EOF){
        tri[pos].row = row - 1;
        tri[pos].col = col - 1;
        tri[pos++].val = val;
        nnz_per_th[row % THREADS]++;
        rowcount[row]++;
      }
      qsort( tri, nnz, sizeof(w_edge_t), w_edge_comp);
    }
    
    fclose(fp);
    if(nnz != pos){
      T0_printf("ERROR: read_matrix_mm_to_dist: nnz (%"PRId64") != pos (%"PRId64")\n", nnz, pos);
      for(i = 0; i < THREADS; i++) nvshmem_write_upc_array_int64(sh_data, i, sizeof(int64_t), -1);
    }
    for(i = 0; i < THREADS; i++){
      nvshmem_write_upc_array_int64(sh_data, i, sizeof(int64_t), nnz_per_th[i]);
      nvshmem_write_upc_array_int64(sh_data, i+THREADS, sizeof(int64_t), nr);
      nvshmem_write_upc_array_int64(sh_data, i+2*THREADS, sizeof(int64_t), nc);
      nvshmem_write_upc_array_int64(sh_data, i+3*THREADS, sizeof(int64_t), values);
    }
    free(nnz_per_th);

  }
  
  nvshmem_barrier();

  int64_t * lsh_data = (( int64_t * )((sh_data)+MYTHREAD));
  
  if(lsh_data[0] == -1)
    return(NULL);
  
  int64_t lnnz = lsh_data[0];
  nr = lsh_data[1];
  nc = lsh_data[2];
  int value = (lsh_data[3] != 0L);
  
  sparsemat_t * A = init_matrix(nr, nc, lnnz, value);
  __shared__ int64_t * tmp_offset = nvshmem_calloc(nr + THREADS, sizeof(int64_t));
  if(!A || !tmp_offset){
    T0_printf("ERROR: read_matrix_mm_to_dist: failed to init matrix or tmp_offset!\n");
    return(NULL);
  }

  /* set up offset array and tmp_offset */
  nvshmem_barrier();
  nvshmem_free(sh_data);
  
  if(!MYTHREAD){
    for(i = 0; i < nr; i++)
      nvshmem_write_upc_array_int64(tmp_offset, i, sizeof(int64_t), rowcount[i]);
    free(rowcount);
  }

  nvshmem_barrier();

  int64_t * ltmp_offset = ( int64_t * )((tmp_offset)+MYTHREAD);
  A->loffset[0] = 0;
  for(i = 1; i <= A->lnumrows; i++){
    A->loffset[i] = A->loffset[i-1] + ltmp_offset[i-1];
    ltmp_offset[i-1] = 0;
  }

  int64_t fromth;
  w_edge_t pkg;
  exstack_t * ex = exstack_init(256, sizeof(w_edge_t));
  if( ex == NULL ) return(NULL);
  
  /* distribute the matrix to all other PEs */
  /* pass around the nonzeros */
  /* this is a strange exstack loop since only PE0 has data to push */
  i = 0;
  while(exstack_proceed(ex, (i == nnz))){
    while(i < nnz){
      if(value == 0){
        pkg.row = edges[i].row;
        pkg.col = edges[i].col;
      }else{
        pkg.row = tri[i].row;
        pkg.col = tri[i].col;
        pkg.val = tri[i].val;
      }
      pe = pkg.row % THREADS;
      if(!exstack_push(ex, &pkg, pe))
        break;
      i++;
    }
    exstack_exchange(ex);

    while(exstack_pop(ex, &pkg, &fromth)){
      int64_t row = pkg.row/THREADS;
      int64_t pos = A->loffset[row] + ltmp_offset[row];
      //printf("pos = %ld row = %ld col = %ld\n", pos, row, pkg.col);fflush(0);
      A->lnonzero[pos] = pkg.col;
      if(value) A->lvalue[pos] = pkg.val;
      ltmp_offset[row]++;
    }
  }

  nvshmem_barrier();
  if(!MYTHREAD){
    if(value == 0)
      free(edges);
    else
      free(tri);
  }

  nvshmem_free(tmp_offset);
  exstack_clear(ex);
  sort_nonzeros(A);
  return(A);
}