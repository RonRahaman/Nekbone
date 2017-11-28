#include "size.h"

__global__ void ax_cuda_kernel(
    double * __restrict__ w,
    double * __restrict__ u,
    double * __restrict__ ur,
    double * __restrict__ us,
    double * __restrict__ ut,
    double * __restrict__ gxyz,
    double * __restrict__ dxm1) {

  __shared__ double s_d[lx1*(lx1+1)];
  __shared__ double s_u_ur[lx1*(lx1+1)];
  __shared__ double s_us[lx1*(lx1+1)];

  int e = blockIdx.x;
  int i = threadIdx.x;
  int j = threadIdx.y;

  s_d[i+j*lx1] = dxm1[i+j*lx1];

  for (int k = 0; k < lz1; k++) {
    const int ij = i + j*lx1;
    const int ijke = i + j*lx1 + k*lx1*ly1 + e*lx1*ly1*lz1;

    s_u_ur[ij] = u[ijke];

    __syncthreads();

    double wr = 0.0;
    double ws = 0.0;
    double wt = 0.0;
    for (int l = 0; l < lx1; l++) {
      const int ijle = i + j*lx1 + l*lx1*ly1 + e*lx1*ly1*lz1;
      const int il = i * l*lx1;
      const int jl = j + l*lx1;
      const int kl = k + l*lx1;
      const int lj = l + j*lx1;
      wr += s_d[il]*s_u_ur[lj];
      ws += s_d[jl]*s_u_ur[il];
      wt += s_d[kl]*u[ijle];
    }
    const int g1 = i + j*lx1 + k*lx1*ly1 + 0*lx1*ly1*lz1 + e*lx1*ly1*lz1*6;
    const int g2 = i + j*lx1 + k*lx1*ly1 + 1*lx1*ly1*lz1 + e*lx1*ly1*lz1*6;
    const int g3 = i + j*lx1 + k*lx1*ly1 + 2*lx1*ly1*lz1 + e*lx1*ly1*lz1*6;
    const int g4 = i + j*lx1 + k*lx1*ly1 + 3*lx1*ly1*lz1 + e*lx1*ly1*lz1*6;
    const int g5 = i + j*lx1 + k*lx1*ly1 + 4*lx1*ly1*lz1 + e*lx1*ly1*lz1*6;
    const int g6 = i + j*lx1 + k*lx1*ly1 + 5*lx1*ly1*lz1 + e*lx1*ly1*lz1*6;
    s_u_ur[ij] = gxyz[g1]*wr + gxyz[g2] + gxyz[g3];
    s_us[ij]   = gxyz[g2]*wr + gxyz[g4] + gxyz[g5];
    ut[ijke]   = gxyz[g3]*wr + gxyz[g5] + gxyz[g6];

    __syncthreads();

    double wtemp = 0.0;

    for (int l = 0; l < lx1; l++) {
      const int il = i + l*lx1;
      const int li = l + i*lx1;
      const int lj = l + j*lx1;
      wtemp += s_d[li]*s_u_ur[lj] + s_d[lj]*s_us[il];
    }

    __syncthreads();

    for (int l = 0; l < lx1; l++) {
      const int ijle = i + j*lx1 +  l*lx1*ly1 + e*lx1*ly1*lz1;
      const int lk = l + k*lx1;
      wtemp += s_d[lk]*ut[ijle];
    }

    w[ijke] = wtemp;

  }
}

extern "C" void ax_cuda_wrapper(
    double * __restrict__ w,
    double * __restrict__ u,
    double * __restrict__ ur,
    double * __restrict__ us,
    double * __restrict__ ut,
    double * __restrict__ gxyz,
    double * __restrict__ dxm1) {

  dim3 grid_dims(lelt, 1, 1);
  dim3 block_dims(lx1, ly1, 1);
  
  ax_cuda_kernel <<<grid_dims, block_dims>>> (w, u, ur, us, ut, gxyz, dxm1);

  return;
}




