c-----------------------------------------------------------------------
      subroutine cg(x,f,g,c,r,w,p,z,n,niter,flop_cg)
      include 'SIZE'
      include 'DXYZ'

c     Solve Ax=f where A is SPD and is invoked by ax()
c
c     Output:  x - vector of length n
c
c     Input:   f - vector of length n
c     Input:   g - geometric factors for SEM operator
c     Input:   c - inverse of the counting matrix
c
c     Work arrays:   r,w,p,z  - vectors of length n
c
c     User-provided ax(w,z,n) returns  w := Az,  
c
c     User-provided solveM(z,r,n) ) returns  z := M^-1 r,  
c

      common /mymask/cmask(-1:lx1*ly1*lz1*lelt)
      parameter (lt=lx1*ly1*lz1*lelt)
      real ur(lt),us(lt),ut(lt),wk(lt)

      real x(n),f(n),r(n),w(n),p(n),z(n),g(2*ndim,n),c(n)

      character*1 ans

      pap = 0.0

c     set machine tolerances
      one = 1.
      eps = 1.e-20
      if (one+eps .eq. one) eps = 1.e-14
      if (one+eps .eq. one) eps = 1.e-7

      rtz1=1.0

!$ACC DATA COPY(ur,us,ut,wk,dxm1,dxtm1)

!$ACC KERNELS PRESENT(x,r,f)
      do i=1,n
         x(i) = 0.0
         r(i) = f(i)
      enddo
!$ACC END KERNELS

!$ACC UPDATE HOST(x,r)

      call maskit (r,cmask,nx1,ny1,nz1) ! Zero out Dirichlet conditions

      rnorm = sqrt(glsc3(r,c,r,n))
      iter = 0
      if (nid.eq.0)  write(6,6) iter,rnorm

      miter = niter
c     call tester(z,r,n)  

!$ACC UPDATE DEVICE(x,r,c)

      do iter=1,miter
!$ACC UPDATE HOST(z,r,c)
         call solveM(z,r,n)    ! preconditioner here

         rtz2=rtz1                                                       ! OPS
         rtz1=glsc3(r,c,z,n)   ! parallel weighted inner product r^T C z ! 3n
!$ACC UPDATE DEVICE(z,r,c)

         beta = rtz1/rtz2
         if (iter.eq.1) beta=0.0

!$ACC KERNELS PRESENT(p,z)
         do i=1,n
            p(i) = beta * p(i) + z(i)
         enddo
!$ACC END KERNELS
!$ACC UPDATE HOST(p)

         call ax(w,p,g,ur,us,ut,wk,n)                                    ! flopa

!$ACC UPDATE HOST(w)
         pap=glsc3(w,c,p,n)                                              ! 3n
!$ACC UPDATE DEVICE(w)

         alpha=rtz1/pap
         alphm=-alpha

!$ACC KERNELS PRESENT(x,r)
         do i=1,n
            x(i) = x(i) + alpha * p(i)
            r(i) = r(i) + alphm * w(i)
         enddo
!$ACC END KERNELS

!$ACC UPDATE HOST(r,c)
         rtr = glsc3(r,c,r,n)                                            ! 3n
!$ACC UPDATE DEVICE(r,c)

         if (iter.eq.1) rlim2 = rtr*eps**2
         if (iter.eq.1) rtr0  = rtr
         rnorm = sqrt(rtr)
c        if (nid.eq.0.and.mod(iter,100).eq.0) 
c    $      write(6,6) iter,rnorm,alpha,beta,pap
         write(6,6) iter,rnorm,alpha,beta,pap
    6    format('cg:',i4,1p4e12.4)
c        if (rtr.le.rlim2) goto 1001

      enddo

 1001 continue

!$ACC END DATA

      if (nid.eq.0) write(6,6) iter,rnorm,alpha,beta,pap

      flop_cg = flop_cg + iter*15.*n

      return
      end
c-----------------------------------------------------------------------
      subroutine solveM(z,r,n)
      include 'INPUT'
      real z(n),r(n)

      nn = n
      call h1mg_solve(z,r,nn)

      return
      end
c-----------------------------------------------------------------------
      subroutine ax(w,u,gxyz,ur,us,ut,wk,n) ! Matrix-vector product: w=A*u
#ifdef _CUDA
      use cudafor
      use cublas
#endif
      include 'SIZE'
#ifdef _CUDA
      include 'NEKCUBLAS'
#endif
      include 'TOTAL'

      interface
      attributes(global) subroutine ax_cuf_naive(w,u,ur,us,ut,
     &                gxyz,dxm1,dxtm1)

      real, intent(out) :: w(nx1,ny1,nz1,nelt)
      real, intent(in)  :: u(nx1,ny1,nz1,nelt)
      real ur  (nx1,ny1,nz1,lelt)
      real us  (nx1,ny1,nz1,lelt)
      real ut  (nx1,ny1,nz1,lelt)

      real gxyz(nx1,ny1,nz1,2*ldim,lelt)

      real, intent(in) :: dxm1(nx1,nx1)
      real, intent(in) :: dxtm1(nx1,nx1)
      end subroutine
      end interface

      real w(nx1*ny1*nz1,nelt),u(nx1*ny1*nz1,nelt)
      real gxyz(nx1*ny1*nz1,2*ldim,nelt)

      parameter (lt=lx1*ly1*lz1*lelt)
      real ur(lt),us(lt),ut(lt),wk(lt)
      common /mymask/cmask(-1:lx1*ly1*lz1*lelt)

      integer e
      real(4) diff_sec
      real tstart, tstop

      tstart = dnekclock()
#ifdef _CUDA
      call cudaProfilerStart()
      !istat = cudaEventRecord(ax_e_start, 0)
#endif
      call ax_lelt(w,u,gxyz,ur,us,ut,wk,dxm1,dxtm1)
c !$ACC DATA PRESENT(w,u,gxyz,ur,us,ut,wk,dxm1,dxtm1)
c !$ACC HOST_DATA USE_DEVICE(w,u,gxyz,ur,us,ut,wk,dxm1,dxtm1)
c      call ax_cuf_naive<<<lelt,dim3(lx1,ly1,lz1)>>>(w,u,gxyz,ur,us,ut,
c     $   dxm1,dxtm1) 
c !$ACC END HOST_DATA
c !$ACC END DATA
#ifdef _CUDA
      call cudaProfilerStop()
      !istat = cudaEventRecord(ax_e_stop, 0)
      !istat = cudaEventElapsedTime(diff_sec, ax_e_start, ax_e_stop)
      !ax_e_sec = ax_e_sec + diff_sec / 1000.0

      ax_e_sec = ax_e_sec + dnekclock() - tstart
#endif


!$ACC UPDATE HOST(w)
      call dssum(w)         ! Gather-scatter operation  ! w   = QQ  w

      do e=1,nelt
      do ijk=1,nx1*ny1*nz1
         w(ijk,e) = w(ijk,e) + 0.1 * u(ijk,e)
      enddo
      enddo

      call maskit(w,cmask,nx1,ny1,nz1)  ! Zero out Dirichlet conditions
!$ACC UPDATE DEVICE(w)

      nxyz=nx1*ny1*nz1
      flop_a = flop_a + (19*nxyz+12*nx1*nxyz)*nelt

      return
      end
c-------------------------------------------------------------------------
      subroutine ax1(w,u,n)
      include 'SIZE'
      real w(n),u(n)
      real h2i
  
      h2i = (n+1)*(n+1)  
      do i = 2,n-1
         w(i)=h2i*(2*u(i)-u(i-1)-u(i+1))
      enddo
      w(1)  = h2i*(2*u(1)-u(2  ))
      w(n)  = h2i*(2*u(n)-u(n-1))

      return
      end
c-------------------------------------------------------------------------
      subroutine ax_lelt(w,u,g,ur,us,ut,wk,dxm1,dxtm1) ! Local matrix-vector product
#ifdef _CUDA
      use openacc
      use cublas
      use cudafor
#endif
      include 'SIZE'
#ifdef _CUDA
      include 'NEKCUBLAS'
#endif

      parameter (lxyz=lx1*ly1*lz1)
      parameter (n=lx1-1)
      parameter (m1=n+1)
      parameter (m2 = m1*m1)
      real ur(0:n,0:n,0:n),us(0:n,0:n,0:n),ut(0:n,0:n,0:n)
      real wk(0:n,0:n,0:n)
      real w(0:n,0:n,0:n,1:lelt),u(0:n,0:n,0:n,1:lelt)
      real g(0:n,0:n,0:n,1:2*ldim,1:lelt)
      real dxm1(0:n,0:n), dxtm1(0:n,0:n)
      integer e, stream, nstreams
      parameter(nstreams=8)

      real ur_e(0:n,0:n,0:n,1:lelt)
      real us_e(0:n,0:n,0:n,1:lelt)
      real ut_e(0:n,0:n,0:n,1:lelt)
      real wk_e(0:n,0:n,0:n,1:lelt)

!$ACC UPDATE DEVICE(dxm1,dxtm1,u,g,w)
!$ACC DATA CREATE(ur_e,us_e,ut_e,wk_e)

      do e=1,nelt

!$ACC HOST_DATA USE_DEVICE(dxm1,dxtm1,u,ur_e,us_e,ut_e)
         call cublasDgemm(
     $      'N', 'N', m1, m2, m1,
     $      1.0, dxm1, m1, 
     $      u(0,0,0,e), m1,
     $      0.0, ur_e(0,0,0,e), m1)
         do k=0,n
            call cublasDgemm(
     $         'N','N', m1, m1, m1,
     $         1.0, u(0,0,k,e), m1,
     $         dxtm1, m1,
     $         0.0, us_e(0,0,k,e), m1)
         enddo
         call cublasDgemm(
     $      'N', 'N', m2, m1, m1,
     $      1.0, u(0,0,0,e), m2,
     $      dxtm1, m1,
     $      0.0, ut_e(0,0,0,e), m2)
!$ACC END HOST_DATA

!$ACC KERNELS PRESENT(g,ur_e,us_e,ut_e)
         do k=0,n
         do j=0,n
         do i=0,n
            wr = g(i,j,k,1,e)*ur_e(i,j,k,e) + 
     $           g(i,j,k,2,e)*us_e(i,j,k,e) + 
     $           g(i,j,k,3,e)*ut_e(i,j,k,e)
            ws = g(i,j,k,2,e)*ur_e(i,j,k,e) + 
     $           g(i,j,k,4,e)*us_e(i,j,k,e) + 
     $           g(i,j,k,5,e)*ut_e(i,j,k,e)
            wt = g(i,j,k,3,e)*ur_e(i,j,k,e) + 
     $           g(i,j,k,5,e)*us_e(i,j,k,e) + 
     $           g(i,j,k,6,e)*ut_e(i,j,k,e)
            ur_e(i,j,k,e) = wr
            us_e(i,j,k,e) = ws
            ut_e(i,j,k,e) = wt
         enddo
         enddo
         enddo
!$ACC END KERNELS

!$ACC HOST_DATA USE_DEVICE(dxm1,dxtm1,ur_e,us_e,w,wk_e)
         call cublasDgemm(
     $      'N', 'N', m1, m2, m1,
     $      1.0, dxtm1, m1,
     $      ur_e(0,0,0,e), m1,
     $      0.0, w(0,0,0,e), m1)

         do k=0,N
            call cublasDgemm(
     $         'N', 'N', m1, m1, m1,
     $         1.0, us_e(0,0,k,e), m1,
     $         dxm1, m1,
     $         0.0, wk_e(0,0,k,e), m1)
         enddo
!$ACC END HOST_DATA

!$ACC KERNELS PRESENT(w,wk_e)
         do k=0,N
         do j=0,N
         do i=0,N
            w(i,j,k,e) = w(i,j,k,e) + wk_e(i,j,k,e)
         enddo
         enddo
         enddo
!$ACC END KERNELS

!$ACC HOST_DATA USE_DEVICE(dxm1,ut_e,wk_e)
         call cublasDgemm(
     $      'N', 'N', m2, m1, m1,
     $      1.0, ut_e(0,0,0,e), m2,
     $      dxm1, m1,
     $      0.0, wk_e(0,0,0,e), m2)
!$ACC END HOST_DATA

!$ACC KERNELS PRESENT(w,wk_e)
         do k=0,N
         do j=0,N
         do i=0,N
            w(i,j,k,e) = w(i,j,k,e) + wk_e(i,j,k,e)
         enddo
         enddo
         enddo
!$ACC END KERNELS

      enddo
!$ACC END DATA
!$ACC UPDATE HOST(dxm1,dxtm1,u,g,w)

      return
      end

c-------------------------------------------------------------------------

#ifdef _CUDA
      subroutine ax_lelt_devicecublas(w,u,g,ur,us,ut,wk)
c     Though this compiles and runs, the solution diverges
      use openacc_cublas
      include 'SIZE'
      include 'TOTAL'

      parameter (lxyz=lx1*ly1*lz1)
      parameter (n=lx1-1)
      parameter (m1=n+1)
      parameter (m2 = m1*m1)
      real ur(0:n,0:n,0:n),us(0:n,0:n,0:n),ut(0:n,0:n,0:n)
      real wk(0:n,0:n,0:n)
      real w(0:n,0:n,0:n,1:lelt),u(0:n,0:n,0:n,1:lelt)
      real g(0:n,0:n,0:n,1:2*ldim,1:lelt)
      integer e
      type(cublasHandle) handle 

!$ACC PARALLEL PRESENT(dxm1,dxtm1,u,ur,us,ut,wk,w,g) CREATE(handle)
      istat = cublasCreate(handle)
!$ACC LOOP SEQ
      do e=1,nelt
         istat = cublasDgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,
     $      m1,m2,m1,1.0,dxm1,m1,u(0,0,0,e),m1,0.0,ur,m1)
!$ACC LOOP SEQ
         do k=0,n
            istat = cublasDgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,
     $         m1,m1,m1,1.0,u(0,0,k,e),m1,dxtm1,m1,0.0,us(0,0,k),m1)
         enddo
         istat = cublasDgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,
     $      m2,m1,m1,1.0,u(0,0,0,e),m2,dxtm1,m1,0.0,ut,m2)

!$ACC LOOP SEQ
         do k=0,n
!$ACC LOOP SEQ
         do j=0,n
!$ACC LOOP SEQ
         do i=0,n
            wr = g(i,j,k,1,e)*ur(i,j,k) + 
     $           g(i,j,k,2,e)*us(i,j,k) + 
     $           g(i,j,k,3,e)*ut(i,j,k)
            ws = g(i,j,k,2,e)*ur(i,j,k) + 
     $           g(i,j,k,4,e)*us(i,j,k) + 
     $           g(i,j,k,5,e)*ut(i,j,k)
            wt = g(i,j,k,3,e)*ur(i,j,k) + 
     $           g(i,j,k,5,e)*us(i,j,k) + 
     $           g(i,j,k,6,e)*ut(i,j,k)
            ur(i,j,k) = wr
            us(i,j,k) = ws
            ut(i,j,k) = wt
         enddo
         enddo
         enddo

         istat = cublasDgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,
     $      m1,m2,m1,1.0,dxtm1,m1,ur,m1,0.0,w(0,0,0,e),m1)

!$ACC LOOP SEQ
         do k=0,N
            istat = cublasDgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,
     $         m1,m1,m1,1.0,us(0,0,k),m1,dxm1,m1,0.0,wk(0,0,k),m1)
         enddo

!$ACC LOOP SEQ
         do k=0,N
!$ACC LOOP SEQ
         do j=0,N
!$ACC LOOP SEQ
         do i=0,N
            w(i,j,k,e) = w(i,j,k,e) + wk(i,j,k)
         enddo
         enddo
         enddo

         istat = cublasDgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,
     $      m2,m1,m1,1.0,ut,m2,dxm1,m1,0.0,wk,m2)

!$ACC LOOP SEQ
         do k=0,N
!$ACC LOOP SEQ
         do j=0,N
!$ACC LOOP SEQ
         do i=0,N
            w(i,j,k,e) = w(i,j,k,e) + wk(i,j,k)
         enddo
         enddo
         enddo
      enddo
      istat = cublasDestroy(handle)
!$ACC END PARALLEL

      return
      end
#endif
c-------------------------------------------------------------------------
      subroutine local_grad3(ur,us,ut,u,n,D,Dt)
c     Output: ur,us,ut         Input:u,n,D,Dt
      real ur(0:n,0:n,0:n),us(0:n,0:n,0:n),ut(0:n,0:n,0:n)
      real u (0:n,0:n,0:n)
      real D (0:n,0:n),Dt(0:n,0:n)
      integer e

      m1 = n+1
      m2 = m1*m1

      call mxm(D ,m1,u,m1,ur,m2)
      do k=0,n
         call mxm(u(0,0,k),m1,Dt,m1,us(0,0,k),m1)
      enddo
      call mxm(u,m2,Dt,m1,ut,m1)

      return
      end
c-----------------------------------------------------------------------
      subroutine local_grad3_t(u,ur,us,ut,N,D,Dt,w)
c     Output: ur,us,ut         Input:u,N,D,Dt
      real u (0:N,0:N,0:N)
      real ur(0:N,0:N,0:N),us(0:N,0:N,0:N),ut(0:N,0:N,0:N)
      real D (0:N,0:N),Dt(0:N,0:N)
      real w (0:N,0:N,0:N)
      integer e

      m1 = N+1
      m2 = m1*m1
      m3 = m1*m1*m1

      call mxm(Dt,m1,ur,m1,u,m2)

      do k=0,N
         call mxm(us(0,0,k),m1,D ,m1,w(0,0,k),m1)
      enddo
      
!$ACC KERNELS PRESENT(u,w)
      do k=0,N
      do j=0,N
      do i=0,N
         u(i,j,k) = u(i,j,k) + w(i,j,k)
      enddo
      enddo
      enddo
!$ACC END KERNELS

      call mxm(ut,m2,D ,m1,w,m1)

!$ACC KERNELS PRESENT(u,w)
      do k=0,N
      do j=0,N
      do i=0,N
         u(i,j,k) = u(i,j,k) + w(i,j,k)
      enddo
      enddo
      enddo
!$ACC END KERNELS

      return
      end
c-----------------------------------------------------------------------
      subroutine maskit(w,pmask,nx,ny,nz)   ! Zero out Dirichlet conditions
      include 'SIZE'
      include 'PARALLEL'

      real pmask(-1:lx1*ly1*lz1*lelt)
      real w(1)
      integer e

      nxyz = nx*ny*nz
      nxy  = nx*ny
      if(pmask(-1).lt.0) then
        j=pmask(0)
        do i = 1,j
           k = pmask(i)
           w(k)=0.0
        enddo
      else
c         Zero out Dirichlet boundaries.
c
c                      +------+     ^ Y
c                     /   3  /|     |
c               4--> /      / |     |
c                   +------+ 2 +    +----> X
c                   |   5  |  /    /
c                   |      | /    /
c                   +------+     Z   
c

        nn = 0
        do e  = 1,nelt
          call get_face(w,nx,e)
          do i = 1,nxyz
             if(w(i).eq.0) then
               nn=nn+1
               pmask(nn)=i
             endif
          enddo
        enddo     
        pmask(-1) = -1.
        pmask(0) = nn
      endif


      return
      end
c-----------------------------------------------------------------------
      subroutine masko(w)   ! Old 'mask'
      include 'SIZE'
      real w(1)

      if (nid.eq.0) w(1) = 0.  ! suitable for solvability

      return
      end
c-----------------------------------------------------------------------
      subroutine masking(w,nx,e,x0,x1,y0,y1,z0,z1)
c     Zeros out boundary
      include 'SIZE'
      integer e,x0,x1,y0,y1,z0,z1
      real w(nx,nx,nx,nelt)
      
c       write(6,*) x0,x1,y0,y1,z0,z1
      do k=z0,z1
      do j=y0,y1
      do i=x0,x1
          w(i,j,k,e)=0.0
      enddo
      enddo
      enddo
      return
      end
c-----------------------------------------------------------------------
      subroutine tester(z,r,n)
c     Used to test if solution to precond. is SPD
      real r(n),z(n)

      do j=1,n
         call rzero(r,n)
         r(j) = 1.0
         call solveM(z,r,n)
         do i=1,n
            write(79,*) z(i)
         enddo
      enddo
      call exitt0
      return
      end
c-----------------------------------------------------------------------
      subroutine get_face(w,nx,ie)
c     zero out all boundaries as Dirichlet
c     to change, change this routine to only zero out 
c     the nodes desired to be Dirichlet, and leave others alone.
      include 'SIZE'
      include 'PARALLEL'
      real w(1)
      integer nx,ie,nelx,nely,nelz
      integer x0,x1,y0,y1,z0,z1
      
      x0=1
      y0=1
      z0=1
      x1=nx
      y1=nx
      z1=nx
      
      nelxy=nelx*nely
      ngl = lglel(ie)        !global element number

      ir = 1+(ngl-1)/nelxy   !global z-count
      iq = mod1(ngl,nelxy)   !global y-count
      iq = 1+(iq-1)/nelx     
      ip = mod1(ngl,nelx)    !global x-count

c     write(6,1) ip,iq,ir,nelx,nely,nelz, nelt,' test it'
c  1  format(7i7,a8)

      if(mod(ip,nelx).eq.1.or.nelx.eq.1)   then  ! Face4
         x0=1
         x1=1
         call masking(w,nx,ie,x0,x1,y0,y1,z0,z1)
      endif
      if(mod(ip,nelx).eq.0)               then   ! Face2
         x0=nx
         x1=nx
         call masking(w,nx,ie,x0,x1,y0,y1,z0,z1)
      endif

      x0=1
      x1=nx
      if(mod(iq,nely).eq.1.or.nely.eq.1) then    ! Face1
         y0=1
         y1=1
         call masking(w,nx,ie,x0,x1,y0,y1,z0,z1)
      endif
      if(mod(iq,nely).eq.0)              then    ! Face3
         y0=nx
         y1=nx
         call masking(w,nx,ie,x0,x1,y0,y1,z0,z1)
      endif

      y0=1
      y1=nx
      if(mod(ir,nelz).eq.1.or.nelz.eq.1) then    ! Face5
         z0=1
         z1=1
         call masking(w,nx,ie,x0,x1,y0,y1,z0,z1)
      endif
      if(mod(ir,nelz).eq.0)              then    ! Face6
         z1=nx
         z0=nx
         call masking(w,nx,ie,x0,x1,y0,y1,z0,z1)
      endif

      return
      end
c-----------------------------------------------------------------------
