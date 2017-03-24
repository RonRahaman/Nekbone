c-----------------------------------------------------------------------
      subroutine h1mg_setup_acc()

      return
      end
c----------------------------------------------------------------------
      subroutine h1mg_solve_acc(z,rhs,n)  !  Solve preconditioner: Mz=rhs
      real z(n),rhs(n)
  
      call copy_acc(z,rhs,n)

      return
      end
c-----------------------------------------------------------------------
