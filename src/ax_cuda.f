      attributes(global) subroutine ax_cuf_naive(
     $   w,u,gxyz,ur,us,ut,dxm1,dxtm1)

      include 'SIZE'

      real, intent(out) :: w(lx1,ly1,lz1,lelt)
      real :: u  (lx1,ly1,lz1,lelt)
      real :: ur (lx1,ly1,lz1,lelt)
      real :: us (lx1,ly1,lz1,lelt)
      real :: ut (lx1,ly1,lz1,lelt)

      real :: gxyz(lx1,ly1,lz1,2*ldim,lelt)

      real, intent(in) :: dxm1(lx1,lx1)
      real, intent(in) :: dxtm1(lx1,lx1)

      integer :: i,j,k,e
      real    :: wr,ws,wt

      e = blockIdx%x
      k = threadIdx%z
      j = threadIdx%y
      i = threadIdx%x

      wr = 0.0
      ws = 0.0
      wt = 0.0
      do l=1,lx1
         wr = wr + dxm1(i,l)*u(l,j,k,e)
         ws = ws + dxm1(j,l)*u(i,l,k,e)
         wt = wt + dxm1(k,l)*u(i,j,l,e)
      enddo
      ur(i,j,k,e) = gxyz(i,j,k,1,e)*wr
     $            + gxyz(i,j,k,2,e)*ws
     $            + gxyz(i,j,k,3,e)*wt
      us(i,j,k,e) = gxyz(i,j,k,2,e)*wr
     $            + gxyz(i,j,k,4,e)*ws
     $            + gxyz(i,j,k,5,e)*wt
      ut(i,j,k,e) = gxyz(i,j,k,3,e)*wr
     $            + gxyz(i,j,k,5,e)*ws
     $            + gxyz(i,j,k,6,e)*wt

      call syncthreads()

      w(i,j,k,e) = 0.0
      do l=1,lx1    ! serial loop, no reduction needed
         w(i,j,k,e) = w(i,j,k,e) + dxtm1(i,l)*ur(l,j,k,e)
     $                           + dxtm1(j,l)*us(i,l,k,e)
     $                           + dxtm1(k,l)*ut(i,j,l,e)
      enddo

      end subroutine


