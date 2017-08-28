c-----------------------------------------------------------------------
      subroutine proxy_setup(a,b,c,d,z,w,g)

      include 'SIZE'
      include 'TOTAL'

      real a(lx1*lx1),b(lx1),c(lx1*lx1),d(lx1*lx1),z(lx1)
     $               , w(lx1*2),g(nx1*ny1*nz1,6,nelt)

      call semhat(a,b,c,d,z,w,nx1-1)

      n = nx1*nx1
      call copy(dxm1,d,n)
      call transpose(dxtm1,nx1,dxm1,nx1)

      call copy(zgm1,z,nx1)   ! GLL points
      call copy(wxm1,b,nx1)   ! GLL weights

      call setup_g(g)
     
c     m = nx1*ny1*nz1*nelt
c     call outmat(g,6,m,'gxyz 1',m)

      return
      end
c-------------------------------------------------------------------------
      subroutine setup_g(g)

      include 'SIZE'
      include 'TOTAL'
      real g(nx1,ny1,nz1,6,nelt)
      integer e

      n = nx1*ny1*nz1*nelt


      do e=1,nelt
      do k=1,nz1
      do j=1,ny1
      do i=1,nx1
         do ii=1,6
            g(i,j,k,ii,e) = 0.0
         enddo
         g(i,j,k,1,e) = wxm1(i)*wxm1(j)*wxm1(k)
         g(i,j,k,4,e) = wxm1(i)*wxm1(j)*wxm1(k)
         g(i,j,k,6,e) = wxm1(i)*wxm1(j)*wxm1(k)
         g(i,j,k,6,e) = wxm1(i)*wxm1(j)*wxm1(k)
      enddo
      enddo
      enddo
      enddo

      return
      end
c-------------------------------------------------------------------------
      subroutine transpose(a,lda,b,ldb)
      real a(lda,1),b(ldb,1)
c
      do j=1,ldb
         do i=1,lda
            a(i,j) = b(j,i)
         enddo
      enddo
      return
      end
c-----------------------------------------------------------------------
      subroutine outmat(a,m,n,name6,ie)
      real a(m,n)
      character*6 name6
c
      n10 = min(n,10)
      write(6,*) 
      write(6,*) ie,' matrix: ',name6,m,n
      do i=1,m
         write(6,6) ie,name6,(a(i,j),j=1,n10)
      enddo
    6 format(i3,1x,a6,1p10e12.4)
      write(6,*) 
      return
      end
c-----------------------------------------------------------------------
      subroutine outmat1(a,m,n,name6,ie)
      real a(m,n)
      character*6 name6
c
      n10 = min(n,10)
      write(ie,*) 
      write(ie,*) ie,' matrix: ',name6,m,n
      do i=1,m
         write(ie,6) ie,name6,(a(i,j),j=1,n10)
      enddo
    6 format(i3,1x,a6,1p10e12.4)
      write(ie,*) 
      return
      end
c-----------------------------------------------------------------------
      function randx(seed)

      arg   = 1.e9*seed
      arg   = 1.e9*cos(arg)
      randx = sin(arg)
      seed  = randx
      seed  = randx

      return
      end
c-----------------------------------------------------------------------
