#ifdef _CUDA
      attributes(global) subroutine ax_cuf(
     $   w,u,ur,us,ut,gxyz,dxm1)

      include 'SIZE'

      real :: w(lx1,ly1,lz1,lelt)
      real :: u(lx1,ly1,lz1,lelt)
      real :: ur  (lx1,ly1,lz1,lelt)
      real :: us  (lx1,ly1,lz1,lelt)
      real :: ut  (lx1,ly1,lz1,lelt)
      real :: gxyz(lx1,ly1,lz1,2*ldim,lelt)
      real :: dxm1(lx1,lx1)

      real, shared :: s_d(lx1+1,lx1)
      real, shared :: s_u_ur(lx1+1,lx1)
      real, shared :: s_us(lx1+1,lx1)

      real :: wr, ws, wt, wtemp
      integer :: e, i, j, k, l

      e = blockIdx % x
      i = threadIdx % x
      j = threadIdx % y

      s_d(i,j) = dxm1(i,j)

      do k=1,lz1

         s_u_ur(i,j) = u(i,j,k,e)

         call syncthreads()

         wr = 0.0
         ws = 0.0
         wt = 0.0
         do l=1,lx1
            wr = wr + s_d(i,l)*s_u_ur(l,j)
            ws = ws + s_d(j,l)*s_u_ur(i,l)
            wt = wt + s_d(k,l)*u(i,j,l,e)
         enddo
         s_u_ur(i,j) = gxyz(i,j,k,1,e)*wr
     $               + gxyz(i,j,k,2,e)*ws
     $               + gxyz(i,j,k,3,e)*wt
         s_us(i,j)   = gxyz(i,j,k,2,e)*wr
     $               + gxyz(i,j,k,4,e)*ws
     $               + gxyz(i,j,k,5,e)*wt
         ut(i,j,k,e) = gxyz(i,j,k,3,e)*wr
     $               + gxyz(i,j,k,5,e)*ws
     $               + gxyz(i,j,k,6,e)*wt
         
         call syncthreads()

         wtemp = 0.0
         do l=1,lx1
            wtemp = wtemp 
     $            + s_d(l,i)*s_u_ur(l,j)
     $            + s_d(l,j)*s_us(i,l)
         enddo
         w(i,j,k,e) = wtemp

         call syncthreads()

         wtemp = w(i,j,k,e)
         do l=1,lx1
            wtemp = wtemp + s_d(l,k)*ut(i,j,l,e)
         enddo
         w(i,j,k,e) = wtemp

      enddo

      end subroutine

#endif
