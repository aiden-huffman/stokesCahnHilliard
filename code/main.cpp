// Deal.II Libraries
#include <algorithm>
#include <boost/iostreams/categories.hpp>

#include <boost/qvm/mat_operations.hpp>
#include <deal.II/base/data_out_base.h>
#include <deal.II/base/exceptions.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/quadrature.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/subscriptor.h>
#include <deal.II/base/symmetric_tensor.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/types.h>
#include <deal.II/base/function.h>
#include <deal.II/base/work_stream.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/index_set.h>
#include <deal.II/base/conditional_ostream.h>

#include <deal.II/distributed/tria.h>
#include <deal.II/distributed/grid_refinement.h>
#include <deal.II/distributed/solution_transfer.h>

#include <deal.II/fe/fe.h>
#include <deal.II/grid/filtered_iterator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_generator.h>

#include <deal.II/lac/block_sparsity_pattern.h>
#include <deal.II/lac/vector_operation.h>
#include <deal.II/numerics/data_component_interpretation.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/solution_transfer.h>

#include <deal.II/fe/component_mask.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_values_extractors.h>
#include <deal.II/fe/fe_update_flags.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_dgp.h>
#include <deal.II/fe/fe_system.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/dofs/dof_renumbering.h>

#include <deal.II/lac/vector_memory.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/linear_operator.h>
#include <deal.II/lac/schur_complement.h>
#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/precondition_selector.h>
#include <deal.II/lac/sparse_ilu.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/linear_operator_tools.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/affine_constraints.h>

#include <deal.II/lac/trilinos_parallel_block_vector.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_block_sparse_matrix.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_solver.h>

#include <deal.II/numerics/vector_tools_boundary.h>
#include <ostream>
#include <random>
#include <unordered_map>
#include <fstream>

namespace stokesCahnHilliard {
    using namespace dealii;

namespace LinearSolvers
{
template <class Matrix, class Preconditioner>
class InverseMatrix : public Subscriptor
{
public:
InverseMatrix(const Matrix &m, const Preconditioner &preconditioner);

template <typename VectorType>
void vmult(VectorType &dst, const VectorType &src) const;

private:
const SmartPointer<const Matrix> matrix;
const Preconditioner &           preconditioner;
};

template <class Matrix, class Preconditioner>
InverseMatrix<Matrix, Preconditioner>::InverseMatrix(
const Matrix &        m,
const Preconditioner &preconditioner)
: matrix(&m)
, preconditioner(preconditioner)
{}

template <class Matrix, class Preconditioner>
template <typename VectorType>
void
InverseMatrix<Matrix, Preconditioner>::vmult(VectorType &      dst,
                                           const VectorType &src) const
{
    SolverControl        solver_control(src.size(),
                                        std::max(1e-8,
                                        1e-8 * src.l2_norm()));
    SolverGMRES<VectorType> cg(solver_control);
    dst = 0;

    try
      {
        cg.solve(*matrix, dst, src, preconditioner);
      }
    catch (std::exception &e)
      {
        Assert(false, ExcMessage(e.what()));
      }
}

template <class PreconditionerA, class PreconditionerS>
class BlockDiagonalPreconditioner : public Subscriptor
{
public:
    BlockDiagonalPreconditioner(const PreconditionerA &preconditioner_A,
                                const PreconditionerS &preconditioner_S);

    void vmult(TrilinosWrappers::MPI::BlockVector &      dst,
               const TrilinosWrappers::MPI::BlockVector &src) const;

private:
    const PreconditionerA &preconditioner_A;
    const PreconditionerS &preconditioner_S;
};

template <class PreconditionerA, class PreconditionerS>
BlockDiagonalPreconditioner<PreconditionerA, PreconditionerS>::
BlockDiagonalPreconditioner(const PreconditionerA &preconditioner_A,
                            const PreconditionerS &preconditioner_S)
: preconditioner_A(preconditioner_A)
, preconditioner_S(preconditioner_S)
{}

template <class PreconditionerA, class PreconditionerS>
void BlockDiagonalPreconditioner<PreconditionerA, PreconditionerS>::vmult(
TrilinosWrappers::MPI::BlockVector &      dst,
const TrilinosWrappers::MPI::BlockVector &src) const
{
    preconditioner_A.vmult(dst.block(0), src.block(0));
    preconditioner_S.vmult(dst.block(1), src.block(1));
}

}// LinearSolvers

namespace EquationData
{

template<int dim>
class InitialValuesPhi : public Function<dim>
{
public:
    InitialValuesPhi(double eps);

    virtual double value(
        const Point<dim> &p,
        const uint       component = 0) const override;

private:
    double eps;
};

template<int dim>
InitialValuesPhi<dim>::InitialValuesPhi(double eps)
    : Function<dim>(2)
    , eps(eps)
{}

template<int dim>
double InitialValuesPhi<dim>::value(
    const Point<dim> &p,
    const uint       component) const
{
    if(component == 0)
    {

        Point<dim> shifted_p1 = p;
        Point<dim> shifted_p2 = p;
        Point<dim> shifted_p3 = p;
        Point<dim> shifted_p4 = p;

        shifted_p1[0] -= 0.5;
        shifted_p1[1] -= 0.5;

        shifted_p2[0] += 0.5;
        shifted_p2[1] += 0.5;
        
        shifted_p3[0] -= 0.5;
        shifted_p3[1] += 0.5;
        
        shifted_p4[0] += 0.5;
        shifted_p4[1] -= 0.5;

        std::vector<double> droplets(4);

        droplets[0] = std::tanh(
            (shifted_p1.norm()-0.1) / (std::sqrt(2) * this->eps)
        );

        droplets[1] = std::tanh(
            (shifted_p2.norm()-0.11) / (std::sqrt(2) * this->eps)
        );

        droplets[2] = std::tanh(
            (shifted_p3.norm()-0.12) / (std::sqrt(2) * this->eps)
        );

        droplets[3] = std::tanh(
            (shifted_p4.norm()-0.13) / (std::sqrt(2) * this->eps)
        );

        return std::accumulate(droplets.begin(),
                               droplets.end(),
                               1.0,
                               std::multiplies<double>());

    } else {
        return 0;
    }
}

} // EquationData

namespace Assembly
{

    namespace Scratch
    {
    template<int dim>
    struct StokesPreconditioner
    {
    StokesPreconditioner(const FiniteElement<dim> &fe_stokes,
                         const Quadrature<dim>    &quad_stokes,
                         const UpdateFlags        update_flags_stokes);

    StokesPreconditioner(const StokesPreconditioner & scratch);

    FEValues<dim> fe_val_stokes;

    std::vector<Tensor<2,dim>>  grad_phi_u;
    std::vector<double>         phi_p;
    };

    template<int dim>
    StokesPreconditioner<dim>::StokesPreconditioner(
        const FiniteElement<dim> &fe_stokes,
        const Quadrature<dim>    &quad_stokes,
        const UpdateFlags        update_flags_stokes)
    : fe_val_stokes(fe_stokes, quad_stokes, update_flags_stokes)
    , grad_phi_u(fe_stokes.n_dofs_per_cell())
    , phi_p(fe_stokes.n_dofs_per_cell())
    {}

    template<int dim>
    StokesPreconditioner<dim>::StokesPreconditioner(
        const StokesPreconditioner &scratch
    )
    : fe_val_stokes(scratch.fe_val_stokes.get_fe(),
                    scratch.fe_val_stokes.get_quadrature(),
                    scratch.fe_val_stokes.get_update_flags())
    , grad_phi_u(scratch.grad_phi_u)
    , phi_p(scratch.phi_p)
    {}

    template<int dim>
    struct StokesMatrix : public StokesPreconditioner<dim>
    {

        StokesMatrix(const FiniteElement<dim> &fe_stokes,
                     const Quadrature<dim>    &quad_stokes,
                     const UpdateFlags        update_flags_stokes);

        StokesMatrix(const StokesMatrix<dim> &scratch);

        std::vector<Tensor<1,dim>>          phi_u;
        std::vector<SymmetricTensor<2,dim>> symgrad_phi_u;
        std::vector<double>                 div_phi_u;

    };

    template<int dim>
    StokesMatrix<dim>::StokesMatrix(
        const FiniteElement<dim>    &fe_stokes,
        const Quadrature<dim>       &quad_stokes,
        const UpdateFlags           update_flags_stokes)
    : StokesPreconditioner<dim>(fe_stokes,
                                quad_stokes,
                                update_flags_stokes)
    , phi_u(fe_stokes.n_dofs_per_cell())
    , symgrad_phi_u(fe_stokes.n_dofs_per_cell())
    , div_phi_u(fe_stokes.n_dofs_per_cell())
    {}

    template<int dim>
    StokesMatrix<dim>::StokesMatrix(const StokesMatrix<dim> &scratch)
    : StokesPreconditioner<dim>(scratch)
    , phi_u(scratch.phi_u)
    , symgrad_phi_u(scratch.symgrad_phi_u)
    , div_phi_u(scratch.div_phi_u)
    {}

    template<int dim>
    struct StokesRHS : public StokesPreconditioner<dim>
    {
        StokesRHS(const FiniteElement<dim>  &fe_stokes,
                  const Quadrature<dim>     &quad_stokes,
                  const UpdateFlags         update_flags_stokes,
                  const FiniteElement<dim>  &fe_ch,
                  const UpdateFlags         update_flags_ch);

        StokesRHS(const StokesRHS<dim>  &scratch);

        FEValues<dim>   fe_val_ch;

        std::vector<Tensor<2,dim>> grad_phi_u;

        std::vector<Tensor<2,dim>> grad_outer_phi_q;
        std::vector<Tensor<1,dim>> grad_phi_q;
    };

    template<int dim>
    StokesRHS<dim> :: StokesRHS(
        const FiniteElement<dim>  &fe_stokes,
        const Quadrature<dim>     &quad_stokes,
        const UpdateFlags         update_flags_stokes,
        const FiniteElement<dim>  &fe_ch,
        const UpdateFlags         update_flags_ch
    ) : StokesPreconditioner<dim>(fe_stokes,
                                  quad_stokes,
                                  update_flags_stokes)
    , fe_val_ch(fe_ch, quad_stokes, update_flags_ch)
    , grad_phi_u(fe_stokes.n_dofs_per_cell())
    , grad_outer_phi_q(quad_stokes.size())
    , grad_phi_q(quad_stokes.size())
    {}

    template<int dim>
    StokesRHS<dim> :: StokesRHS(const StokesRHS<dim> &scratch)
    : StokesPreconditioner<dim>(scratch.fe_val_stokes.get_fe(),
                                scratch.fe_val_stokes.get_quadrature(),
                                scratch.fe_val_stokes.get_update_flags())
    , fe_val_ch(
        scratch.fe_val_ch.get_fe(),
        scratch.fe_val_ch.get_quadrature(),
        scratch.fe_val_ch.get_update_flags())
    , grad_phi_u(scratch.grad_phi_u)
    , grad_outer_phi_q(scratch.grad_outer_phi_q)
    , grad_phi_q(scratch.grad_phi_q)
    {}

    template<int dim>
    struct CahnHilliardMatrix
    {

        CahnHilliardMatrix(const FiniteElement<dim> &fe_ch,
                           const Quadrature<dim>    &quad_ch,
                           const UpdateFlags        update_flags_ch);

        CahnHilliardMatrix(const CahnHilliardMatrix<dim> &scratch);
        
        FEValues<dim> fe_val_ch;
        
        std::vector<double>         phi_val;
        std::vector<Tensor<1,dim>>  phi_grad;
        std::vector<double>         eta_val;
        std::vector<Tensor<1,dim>>  eta_grad;

    };
    
    template<int dim>
    CahnHilliardMatrix<dim> :: CahnHilliardMatrix(
        const FiniteElement<dim> &fe_ch,
        const Quadrature<dim>    &quad_ch,
        const UpdateFlags        update_flags_ch
    ) : fe_val_ch(fe_ch, quad_ch, update_flags_ch)
    , phi_val(fe_ch.n_dofs_per_cell())
    , phi_grad(fe_ch.n_dofs_per_cell())
    , eta_val(fe_ch.n_dofs_per_cell())
    , eta_grad(fe_ch.n_dofs_per_cell())
    {}

    template<int dim>
    CahnHilliardMatrix<dim> :: CahnHilliardMatrix(
        const CahnHilliardMatrix<dim> &scratch
    ) : fe_val_ch(scratch.fe_val_ch.get_fe(),
                  scratch.fe_val_ch.get_quadrature(),
                  scratch.fe_val_ch.get_update_flags())
    , phi_val(scratch.phi_val)
    , phi_grad(scratch.phi_grad)
    , eta_val(scratch.eta_val)
    , eta_grad(scratch.eta_grad)
    {}

    template<int dim>
    struct CahnHilliardRHS
    {
        
        CahnHilliardRHS(const FiniteElement<dim> &fe_ch,
                        const Quadrature<dim>    &quad_ch,
                        const UpdateFlags        update_flags_ch,
                        const FiniteElement<dim> &fe_stokes,
                        const UpdateFlags        update_flags_stokes);
        CahnHilliardRHS(const CahnHilliardRHS<dim> &scratch);

        FEValues<dim> fe_val_ch;
        FEValues<dim> fe_val_stokes;

        std::vector<double>         phi_val;
        std::vector<Tensor<1,dim>>  phi_grad;

        std::vector<double>         phi_old_q;
        std::vector<double>         phi_old_old_q;

        std::vector<Tensor<1,dim>>  phi_grad_old_q;
        std::vector<Tensor<1,dim>>         phi_grad_old_old_q;

        std::vector<Tensor<1,dim>>  vel_q;
        std::vector<Tensor<1,dim>>  vel_old_q;
 
    };

    template<int dim>
    CahnHilliardRHS<dim> :: CahnHilliardRHS(
        const FiniteElement<dim> &fe_ch,
        const Quadrature<dim>    &quad_ch,
        const UpdateFlags        update_flags_ch,
        const FiniteElement<dim> &fe_stokes,
        const UpdateFlags        update_flags_stokes
    ) : fe_val_ch(fe_ch,
                  quad_ch,
                  update_flags_ch)
    , fe_val_stokes(fe_stokes,
                    quad_ch,
                    update_flags_stokes)
    , phi_val(fe_ch.n_dofs_per_cell())
    , phi_grad(fe_ch.n_dofs_per_cell())
    , phi_old_q(quad_ch.size())
    , phi_old_old_q(quad_ch.size())
    , phi_grad_old_q(quad_ch.size())
    , phi_grad_old_old_q(quad_ch.size())
    , vel_q(quad_ch.size())
    , vel_old_q(quad_ch.size())
    {}

    template<int dim>
    CahnHilliardRHS<dim> :: CahnHilliardRHS(
        const CahnHilliardRHS<dim> &scratch
    ) : fe_val_ch(
        scratch.fe_val_ch.get_fe(),
        scratch.fe_val_ch.get_quadrature(),
        scratch.fe_val_ch.get_update_flags())
    , fe_val_stokes(
        scratch.fe_val_stokes.get_fe(),
        scratch.fe_val_stokes.get_quadrature(),
        scratch.fe_val_stokes.get_update_flags())
    , phi_val(scratch.phi_val)
    , phi_grad(scratch.phi_grad)
    , phi_old_q(scratch.phi_old_q)
    , phi_old_old_q(scratch.phi_old_q)
    , phi_grad_old_q(scratch.phi_grad_old_q)
    , phi_grad_old_old_q(scratch.phi_grad_old_q)
    , vel_q(scratch.vel_old_q)
    , vel_old_q(scratch.vel_old_q)
    {}

    }
   
    namespace CopyData
    {

    template<int dim>
    struct StokesPreconditioner
    {
        StokesPreconditioner(const FiniteElement<dim> &fe_stokes);
        StokesPreconditioner(const StokesPreconditioner<dim> &data);
        StokesPreconditioner &operator=
            (const StokesPreconditioner<dim> &)=default;

        FullMatrix<double>                      local_matrix;
        std::vector<types::global_dof_index>    local_dof_indices;
    };

    template<int dim>
    StokesPreconditioner<dim> :: StokesPreconditioner(
        const FiniteElement<dim> &fe_stokes
    ) : local_matrix(fe_stokes.n_dofs_per_cell(),
                     fe_stokes.n_dofs_per_cell())
    , local_dof_indices(fe_stokes.n_dofs_per_cell())
    {}

    template<int dim>
    StokesPreconditioner<dim> :: StokesPreconditioner(
        const StokesPreconditioner<dim> &data
    ) : local_matrix(data.local_matrix)
    , local_dof_indices(data.local_dof_indices)
    {}

    template<int dim>
    struct StokesMatrix
    {
        StokesMatrix(const FiniteElement<dim> &fe_stokes);
        StokesMatrix(const StokesMatrix<dim>  &data);
        StokesMatrix &operator=(const StokesMatrix<dim> &) = default;

        FullMatrix<double>                      local_matrix;
        std::vector<types::global_dof_index>    local_dof_indices;
    };

    template<int dim>
    StokesMatrix<dim> :: StokesMatrix(const FiniteElement<dim> &fe_stokes)
    : local_matrix(fe_stokes.n_dofs_per_cell(),
                   fe_stokes.n_dofs_per_cell())
    , local_dof_indices(fe_stokes.n_dofs_per_cell())
    {}
    
    template<int dim>
    StokesMatrix<dim> :: StokesMatrix(const StokesMatrix<dim> &data)
    : local_matrix(data.local_matrix)
    , local_dof_indices(data.local_dof_indices)
    {}

    template<int dim>
    struct StokesRHS
    {
        StokesRHS(const FiniteElement<dim> &fe_stokes);
        StokesRHS(const StokesRHS<dim>  &data);
        StokesRHS &operator=(const StokesRHS<dim> &) = default;

        Vector<double>                          local_rhs;
        std::vector<types::global_dof_index>    local_dof_indices;
    };

    template<int dim>
    StokesRHS<dim> :: StokesRHS(const FiniteElement<dim> &fe_stokes)
    : local_rhs(fe_stokes.n_dofs_per_cell())
    , local_dof_indices(fe_stokes.n_dofs_per_cell())
    {}

    template<int dim>
    StokesRHS<dim> :: StokesRHS(const StokesRHS<dim> &data)
    : local_rhs(data.local_rhs)
    , local_dof_indices(data.local_dof_indices)
    {}

    template<int dim>
    struct CahnHilliardMatrix
    {
        CahnHilliardMatrix(const FiniteElement<dim> &fe_ch);
        CahnHilliardMatrix(const CahnHilliardMatrix<dim> &data);
        CahnHilliardMatrix &operator=(const CahnHilliardMatrix<dim> &)=default;

        FullMatrix<double>                      local_matrix;
        std::vector<types::global_dof_index>    local_dof_indices;
    };

    template<int dim>
    CahnHilliardMatrix<dim> :: CahnHilliardMatrix(
        const FiniteElement<dim> &fe_ch
    ) : local_matrix(fe_ch.n_dofs_per_cell(), fe_ch.n_dofs_per_cell())
    , local_dof_indices(fe_ch.n_dofs_per_cell())
    {}

    template<int dim>
    CahnHilliardMatrix<dim> :: CahnHilliardMatrix(
        const CahnHilliardMatrix<dim> &data
    ) : local_matrix(data.local_matrix)
    , local_dof_indices(data.local_dof_indices)
    {}

    template<int dim>
    struct CahnHilliardRHS
    {
        CahnHilliardRHS(const FiniteElement<dim> &fe_ch);
        CahnHilliardRHS(const CahnHilliardRHS<dim> &data);
        CahnHilliardRHS &operator=(const CahnHilliardRHS<dim> &)=default;

        Vector<double>                          local_rhs;
        std::vector<types::global_dof_index>    local_dof_indices;
    };

    template<int dim>
    CahnHilliardRHS<dim> :: CahnHilliardRHS(const FiniteElement<dim> &fe_ch)
    : local_rhs(fe_ch.n_dofs_per_cell())
    , local_dof_indices(fe_ch.n_dofs_per_cell())
    {}

    template<int dim>
    CahnHilliardRHS<dim> :: CahnHilliardRHS(const CahnHilliardRHS<dim> &data)
    : local_rhs(data.local_rhs)
    , local_dof_indices(data.local_dof_indices)
    {}

    } // CopyData

} // Assembly

template<int dim>
class SCHSolver
{
public:
    SCHSolver();
    void run(bool debug=false);

private:

    MPI_Comm            mpi_communicator;
    ConditionalOStream  pcout;

    double eps;

    uint min_refine, max_refine;

    uint                                        degree;
    parallel::distributed::Triangulation<dim>   triangulation;
    FESystem<dim>                               fe_stokes;
    FESystem<dim>                               fe_ch;
    QGauss<dim>                                 quad_formula;

    DoFHandler<dim> dof_handler_stokes;
    DoFHandler<dim> dof_handler_ch;

    AffineConstraints<double>   constraints_stokes;
    AffineConstraints<double>   constraints_ch;

    TrilinosWrappers::BlockSparseMatrix matrix_stokes;
    TrilinosWrappers::BlockSparseMatrix precon_stokes;

    TrilinosWrappers::MPI::BlockVector  solution_stokes;
    TrilinosWrappers::MPI::BlockVector  solution_old_stokes;
    TrilinosWrappers::MPI::BlockVector  rhs_stokes;

    TrilinosWrappers::BlockSparseMatrix matrix_ch;

    TrilinosWrappers::MPI::BlockVector  solution_ch;
    TrilinosWrappers::MPI::BlockVector  solution_old_ch;
    TrilinosWrappers::MPI::BlockVector  solution_old_old_ch;
    TrilinosWrappers::MPI::BlockVector  rhs_ch;

    double timestep, timestep_old;
    uint timestep_number;

    void setupTriang();

    // DoF setup
    void setupDoFsStokes();
    void setupDoFsCahnHilliard();
    void setupDoFs();

    // Reinitialize matrices
    void reinitStokesMatrix(
        const std::vector<IndexSet> stokes_partitioning,
        const std::vector<IndexSet> stokes_relevant_paritioning
    );
    void reinitCahnHilliardMatrix(
        const std::vector<IndexSet> ch_partitioning,
        const std::vector<IndexSet> ch_relevant_partitioning
    );

    void initializeValues();

    // Stokes Assembly
    void assembleStokesPreconLocal(
        const typename DoFHandler<dim>::active_cell_iterator &cell,
        Assembly::Scratch::StokesPreconditioner<dim>         &scratch,
        Assembly::CopyData::StokesPreconditioner<dim>        &data
    );
    void copyStokesPreconLocalToGlobal(
        const Assembly::CopyData::StokesPreconditioner<dim>  &data
    );
    void assembleStokesPrecon();

    void assembleStokesMatrixLocal(
        const typename DoFHandler<dim>::active_cell_iterator &cell,
        Assembly::Scratch::StokesMatrix<dim>&        scratch,
        Assembly::CopyData::StokesMatrix<dim>&       data
    );
    void copyStokesMatrixLocalToGlobal(
        const Assembly::CopyData::StokesMatrix<dim> & data
    );
    void assembleStokesMatrix();

    void assembleStokesRHSLocal(
        const typename DoFHandler<dim>::active_cell_iterator &cell,
        Assembly::Scratch::StokesRHS<dim>&        scratch,
        Assembly::CopyData::StokesRHS<dim>&       data
    );
    void copyStokesRHSLocalToGlobal(
        const Assembly::CopyData::StokesRHS<dim> &data
    );
    void assembleStokesRHS();
    
    // Cahn-Hilliard Assembly
    void assembleCahnHilliardMatrixLocal(
        const typename DoFHandler<dim>::active_cell_iterator &cell,
        Assembly::Scratch::CahnHilliardMatrix<dim>&        scratch,
        Assembly::CopyData::CahnHilliardMatrix<dim>&       data
    );
    void copyCahnHilliardMatrixLocalToGlobal(
        const Assembly::CopyData::CahnHilliardMatrix<dim> &data
    );
    void assembleCahnHilliardMatrix();

    void assembleCahnHilliardRHSLocal(
        const typename DoFHandler<dim>::active_cell_iterator &cell,
        Assembly::Scratch::CahnHilliardRHS<dim>&        scratch,
        Assembly::CopyData::CahnHilliardRHS<dim>&       data
    );
    void copyCahnHilliardRHSLocalToGlobal(
        const Assembly::CopyData::CahnHilliardRHS<dim> &data
    );
    void assembleCahnHilliardRHS();

    // Solvers
    void solveStokes();
    void computeTimestep();
    void solveCahnHilliard();

    // Grid Refinement
    void refineGrid();

    // Output functions
    void outputStokes();
    void outputTimestep(const uint timestep_number);
};

template<int dim>
SCHSolver<dim>::SCHSolver()
: mpi_communicator(MPI_COMM_WORLD)
, pcout(std::cout, (Utilities::MPI::this_mpi_process(mpi_communicator) == 0))
, eps(1e-3)
, min_refine(4)
, max_refine(11)
, degree(1)
, triangulation(mpi_communicator,
                typename Triangulation<dim>::MeshSmoothing(
                Triangulation<dim>::smoothing_on_refinement |
                Triangulation<dim>::smoothing_on_coarsening)
                )
, fe_stokes(FE_Q<dim>(degree+1), dim, FE_Q<dim>(degree), 1)
, fe_ch(FE_Q<dim>(degree), 2)
, quad_formula(degree+3)
, dof_handler_stokes(this->triangulation)
, dof_handler_ch(this->triangulation)
, timestep(1e-2)
, timestep_old(timestep)
, timestep_number(0)
{}

template<int dim>
void SCHSolver<dim>::setupTriang()
{ 
    this->pcout << "Generating triangulation... " << std::endl;
 
    GridGenerator::hyper_cube(
      triangulation, -1, 1, true);
 
    this->pcout << "\tRefining grid" << std::endl;
    triangulation.refine_global(6);

    this->pcout << "\tActive cells: "
                << triangulation.n_global_active_cells()
                << std::endl;

    this->pcout << "Completed." << std::endl;
}

template<int dim>
void SCHSolver<dim>::setupDoFsStokes()
{
    this->dof_handler_stokes.distribute_dofs(this->fe_stokes);

    std::vector<uint> stokes_sub_blocks(dim+1,0);
    stokes_sub_blocks[dim] = 1;

    DoFRenumbering::component_wise(this->dof_handler_stokes,
                                   stokes_sub_blocks);

    const std::vector<types::global_dof_index> dofs_per_block =
        DoFTools::count_dofs_per_fe_block(this->dof_handler_stokes,
                                          stokes_sub_blocks);

    const types::global_dof_index n_u = dofs_per_block[0],
                                  n_p = dofs_per_block[1];

    std::vector<IndexSet> stokes_partitioning, stokes_relevant_partitioning;
    IndexSet stokes_relevant_set;
    {
        IndexSet stokes_index_set = dof_handler_stokes.locally_owned_dofs();
        stokes_partitioning.push_back(stokes_index_set.get_view(0, n_u));
        stokes_partitioning.push_back(stokes_index_set.get_view(n_u, n_u+n_p));

        DoFTools::extract_locally_relevant_dofs(this->dof_handler_stokes,
                                                stokes_relevant_set);
        stokes_relevant_partitioning.push_back(
            stokes_relevant_set.get_view(0, n_u));
        stokes_relevant_partitioning.push_back(
            stokes_relevant_set.get_view(n_u, n_u+n_p));
    }

    // Hanging node and pressure constraints
    {
        this->constraints_stokes.clear();
        this->constraints_stokes.reinit(stokes_relevant_set);

        DoFTools::make_hanging_node_constraints(this->dof_handler_stokes,
                                                this->constraints_stokes);

        if(Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
        {
            FEValuesExtractors::Scalar pressure(dim);
            ComponentMask pressure_mask = 
                this->fe_stokes.component_mask(pressure);

            std::vector<IndexSet> pressure_dofs =
                DoFTools::locally_owned_dofs_per_component(
                    this->dof_handler_stokes,
                    pressure_mask
                );
            const types::global_dof_index first_pressure_dof
                = pressure_dofs[dim].nth_index_in_set(0);
            this->constraints_stokes.add_line(first_pressure_dof);
        }

        const FEValuesExtractors::Vector velocity(0);

        for(uint i = 0; i < 4; i++){
            VectorTools::interpolate_boundary_values(
                this->dof_handler_stokes,
                i, Functions::ZeroFunction<dim>(dim+1),
                this->constraints_stokes,
                this->fe_stokes.component_mask(velocity)
            );
        }

        this->constraints_stokes.close();
    }

    reinitStokesMatrix(
        stokes_partitioning,
        stokes_relevant_partitioning
    );

    this->solution_stokes.reinit(
        stokes_relevant_partitioning,
        mpi_communicator
    );

    this->solution_old_stokes.reinit(
        this->solution_stokes
    );

    this->rhs_stokes.reinit(
        stokes_partitioning,
        stokes_relevant_partitioning,
        mpi_communicator,
        true
    );

}

template<int dim>
void SCHSolver<dim>::reinitStokesMatrix(
    const std::vector<IndexSet> stokes_partitioning,
    const std::vector<IndexSet> stokes_relevant_partitiong
)
{
    this->matrix_stokes.clear();
    this->precon_stokes.clear();

    // Setup Stokes DoFs
    {
    TrilinosWrappers::BlockSparsityPattern sp(
        stokes_partitioning,
        stokes_partitioning,
        stokes_relevant_partitiong,
        mpi_communicator
    );

    DoFTools::make_sparsity_pattern(
        this->dof_handler_stokes,
        sp, this->constraints_stokes,
        false,
        Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)
    );

    sp.compress();

    this->matrix_stokes.reinit(sp);
    }

    // Setup Preconditioner DoFs
    {
    TrilinosWrappers::BlockSparsityPattern sp(
        stokes_partitioning,
        stokes_partitioning,
        stokes_relevant_partitiong,
        mpi_communicator
    );

    DoFTools::make_sparsity_pattern(
        this->dof_handler_stokes,
        sp, this->constraints_stokes,
        false,
        Utilities::MPI::this_mpi_process(mpi_communicator)
    );

    sp.compress();

    this->precon_stokes.reinit(sp);
    }
}

template<int dim>
void SCHSolver<dim>::setupDoFsCahnHilliard()
{

    this->dof_handler_ch.distribute_dofs(this->fe_ch);

    std::vector<uint> ch_sub_blocks(2,0);
    ch_sub_blocks[1] = 1;

    DoFRenumbering::component_wise(this->dof_handler_ch,
                                   ch_sub_blocks);

    const std::vector<types::global_dof_index> dofs_per_block =
        DoFTools::count_dofs_per_fe_block(this->dof_handler_ch,
                                          ch_sub_blocks);

    const types::global_dof_index n_phi = dofs_per_block[0],
                                  n_eta = dofs_per_block[1];

    std::vector<IndexSet> ch_partitioning, ch_relevant_partitioning;
    IndexSet ch_relevant_set;
    {
        IndexSet ch_index_set = dof_handler_ch.locally_owned_dofs();
        ch_partitioning.push_back(ch_index_set.get_view(0, n_phi));
        ch_partitioning.push_back(ch_index_set.get_view(
            n_phi, n_phi+n_eta
        ));

        DoFTools::extract_locally_relevant_dofs(this->dof_handler_ch,
                                                ch_relevant_set);
        ch_relevant_partitioning.push_back(
            ch_relevant_set.get_view(0, n_phi));
        ch_relevant_partitioning.push_back(
            ch_relevant_set.get_view(n_phi, n_phi+n_eta));
    }

    {
        this->constraints_ch.clear();
        this->constraints_ch.reinit(ch_relevant_set);

        DoFTools::make_hanging_node_constraints(this->dof_handler_ch,
                                                this->constraints_ch);
        // Periodicity constraints
            if(dim == 2){

            std::vector<GridTools::PeriodicFacePair<
                typename DoFHandler<dim>::cell_iterator
            >> periodicity_vectorX;

            std::vector<GridTools::PeriodicFacePair<
                typename DoFHandler<dim>::cell_iterator
            >> periodicity_vectorY;

            DoFTools::make_periodicity_constraints<dim,dim>(
                periodicity_vectorX,
                this->constraints_ch
            );
            DoFTools::make_periodicity_constraints<dim,dim>(
                periodicity_vectorY,
                this->constraints_ch
            );

        } else if(dim == 3){

            std::vector<GridTools::PeriodicFacePair<
                typename DoFHandler<dim>::cell_iterator
            >> periodicity_vectorX;

            std::vector<GridTools::PeriodicFacePair<
                typename DoFHandler<dim>::cell_iterator
            >> periodicity_vectorY;
            
            std::vector<GridTools::PeriodicFacePair<
                typename DoFHandler<dim>::cell_iterator
            >> periodicity_vectorZ;

            DoFTools::make_periodicity_constraints<dim,dim>(
                periodicity_vectorX,
                this->constraints_ch
            );

            DoFTools::make_periodicity_constraints<dim,dim>(
                periodicity_vectorY,
                this->constraints_ch
            );

            DoFTools::make_periodicity_constraints<dim,dim>(
                periodicity_vectorZ,
                this->constraints_ch
            );

        }

        this->constraints_ch.close();
    }

    reinitCahnHilliardMatrix(ch_partitioning,
                             ch_relevant_partitioning);

    this->solution_ch.reinit(
        ch_relevant_partitioning,
        mpi_communicator
    );
    this->solution_old_ch.reinit(
        this->solution_ch
    );
    this->solution_old_old_ch.reinit(
        this->solution_ch
    );
    this->rhs_ch.reinit(
        ch_partitioning,
        ch_relevant_partitioning,
        mpi_communicator,
        true
    );
}

template<int dim>
void SCHSolver<dim>::reinitCahnHilliardMatrix(
    const std::vector<IndexSet> ch_partitioning,
    const std::vector<IndexSet> ch_relevant_partitiong
)
{
    this->matrix_ch.clear();
    
    {
    TrilinosWrappers::BlockSparsityPattern sp(
        ch_partitioning,
        ch_partitioning,
        ch_relevant_partitiong,
        mpi_communicator
    );

    DoFTools::make_sparsity_pattern(
        this->dof_handler_ch,
        sp, this->constraints_ch,
        false,
        Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)
    );

    sp.compress();

    this->matrix_ch.reinit(sp);
    }
}

template<int dim>
void SCHSolver<dim>::setupDoFs()
{
    this->pcout << "Setting up DoFs" << std::endl;

    this->setupDoFsStokes();
    this->setupDoFsCahnHilliard();

    this->pcout << "Total DoFs: "
                << this->dof_handler_stokes.n_dofs()
                +  this->dof_handler_ch.n_dofs()
                << std::endl;

    this->pcout << "Completed." << std::endl;
}

template<int dim>
void SCHSolver<dim>::initializeValues()
{
    this->pcout << "Initializing values for phi" << std::endl;

    std::vector<uint> ch_sub_blocks(2,0);
    ch_sub_blocks[1] = 1;

    const std::vector<types::global_dof_index> dofs_per_block =
        DoFTools::count_dofs_per_fe_block(this->dof_handler_ch,
                                          ch_sub_blocks);

    const types::global_dof_index n_phi = dofs_per_block[0],
                                  n_eta = dofs_per_block[1];

    std::vector<IndexSet> ch_partitioning;
    IndexSet ch_relevant_set;
    {
        IndexSet ch_index_set = dof_handler_ch.locally_owned_dofs();
        ch_partitioning.push_back(ch_index_set.get_view(0, n_phi));
        ch_partitioning.push_back(ch_index_set.get_view(
            n_phi, n_phi+n_eta
        ));
    }

    TrilinosWrappers::MPI::BlockVector interp_tmp;
    interp_tmp.reinit(
        ch_partitioning,
        mpi_communicator
    );
    
    VectorTools::interpolate(this->dof_handler_ch,
                             EquationData::InitialValuesPhi<dim>(this->eps),
                             interp_tmp);

    this->constraints_ch.distribute(interp_tmp);

    this->solution_ch = interp_tmp;
    this->solution_old_ch       = this->solution_ch;
    this->solution_old_old_ch   = this->solution_ch;

    auto phi_range = std::minmax_element(
        this->solution_ch.block(0).begin(),
        this->solution_ch.block(0).end());
    auto eta_range = std::minmax_element(
        this->solution_ch.block(1).begin(),
        this->solution_ch.block(1).end());

    this->pcout << "Initial values propagated:\n"
                << "    Phi Range: (" 
                    << Utilities::MPI::min(*phi_range.first,
                                           mpi_communicator) << ", " 
                    << Utilities::MPI::max(*phi_range.second,
                                           mpi_communicator)
                << ")" 
                << std::endl;
    this->pcout << "    Eta Range: (" 
                    << Utilities::MPI::min(*eta_range.first,
                                           mpi_communicator) << ", " 
                    << Utilities::MPI::max(*eta_range.second,
                                           mpi_communicator)
                << ")" 
                << std::endl;
}

template<int dim>
void SCHSolver<dim>::assembleStokesPreconLocal(
    const typename DoFHandler<dim>::active_cell_iterator &cell,
    Assembly::Scratch::StokesPreconditioner<dim>&        scratch,
    Assembly::CopyData::StokesPreconditioner<dim>&       data
)
{
    const uint dofs_per_cell =
        scratch.fe_val_stokes.get_fe().n_dofs_per_cell();
    const uint n_q_points =
        scratch.fe_val_stokes.get_quadrature().size();

    scratch.fe_val_stokes.reinit(cell);
    cell->get_dof_indices(data.local_dof_indices);

    data.local_matrix = 0;

    for(uint q = 0; q < n_q_points; q++)
    {

        for(uint k = 0; k < n_q_points; k++)
        {
            scratch.phi_p[k] = scratch.fe_val_stokes.shape_value(k, q);
        }

        for(uint i = 0; i < dofs_per_cell; i++)
        {
            for(uint j = 0; j < dofs_per_cell; j++)
            {
                data.local_matrix(i,j) +=
                    scratch.phi_p[i] * scratch.phi_p[j]
                    * scratch.fe_val_stokes.JxW(q);
            }
        }
    }
}

template<int dim>
void SCHSolver<dim>::copyStokesPreconLocalToGlobal(
    const Assembly::CopyData::StokesPreconditioner<dim> &data
)
{
    this->constraints_stokes.distribute_local_to_global(
        data.local_matrix,
        data.local_dof_indices,
        this->precon_stokes
    );
}

template<int dim>
void SCHSolver<dim>::assembleStokesPrecon()
{


    this->precon_stokes = 0;

    this->pcout << "Assembling Stokes preconditioner" << std::endl;

    auto worker = [this](
        const typename DoFHandler<dim>::active_cell_iterator &cell,
        Assembly::Scratch::StokesPreconditioner<dim>         &scratch,
        Assembly::CopyData::StokesPreconditioner<dim>        &data)
    {this->assembleStokesPreconLocal(cell, scratch, data);};
    
    auto copier = [this](
        const Assembly::CopyData::StokesPreconditioner<dim> &data)
    {this->copyStokesPreconLocalToGlobal(data);};

    using CellFilter = FilteredIterator<typename 
        DoFHandler<dim>::active_cell_iterator>;

    WorkStream::run(
        CellFilter(IteratorFilters::LocallyOwnedCell(),
                   this->dof_handler_stokes.begin_active()),
        CellFilter(IteratorFilters::LocallyOwnedCell(),
                   this->dof_handler_stokes.end()),
        worker,
        copier,
        Assembly::Scratch::StokesPreconditioner<dim>(this->fe_stokes,
                                                     this->quad_formula,
                                                     update_values |
                                                     update_JxW_values),
        Assembly::CopyData::StokesPreconditioner<dim>(this->fe_stokes)
    );

    this->precon_stokes.compress(VectorOperation::add);

    this->pcout << "Completed" << std::endl;                                                     
}

template<int dim>
void SCHSolver<dim>::assembleStokesMatrixLocal(
    const typename DoFHandler<dim>::active_cell_iterator &cell,
    Assembly::Scratch::StokesMatrix<dim>&        scratch,
    Assembly::CopyData::StokesMatrix<dim>&       data
)
{

    data.local_matrix = 0;

    scratch.fe_val_stokes.reinit(cell);

    const uint dofs_per_cell =
        scratch.fe_val_stokes.get_fe().n_dofs_per_cell();
    const uint n_q_points =
        scratch.fe_val_stokes.get_quadrature().size();

    cell->get_dof_indices(data.local_dof_indices);

    FEValuesExtractors::Vector velocities(0);
    FEValuesExtractors::Scalar pressure(dim);

    for(uint q = 0; q < n_q_points; q++)
    {

        for(uint k = 0; k < dofs_per_cell; k++)
        {
            scratch.symgrad_phi_u[k] = 
                scratch.fe_val_stokes[velocities].symmetric_gradient(k,q);
            scratch.div_phi_u[k] =
                scratch.fe_val_stokes[velocities].divergence(k,q);

            scratch.phi_u[k] = scratch.fe_val_stokes[velocities].value(k,q);
            scratch.phi_p[k] = scratch.fe_val_stokes[pressure].value(k,q);

        }

        for(uint i = 0; i < dofs_per_cell; i++)
        {
            for(uint j = 0; j < dofs_per_cell; j++)
            {
                    data.local_matrix(i,j) +=
                        (2 * (scratch.symgrad_phi_u[i] * scratch.symgrad_phi_u[j])
                         - scratch.div_phi_u[i] * scratch.phi_p[j]
                         - scratch.phi_p[i] * scratch.div_phi_u[j])
                        * scratch.fe_val_stokes.JxW(q);
            }
        }

    }
}

template<int dim>
void SCHSolver<dim>::copyStokesMatrixLocalToGlobal(
    const Assembly::CopyData::StokesMatrix<dim> &data
)
{
    this->constraints_stokes.distribute_local_to_global(
        data.local_matrix,
        data.local_dof_indices,
        this->matrix_stokes
    );
}

template<int dim>
void SCHSolver<dim>::assembleStokesMatrix()
{

    this->matrix_stokes = 0;

    this->pcout << "Assembling Stokes matrix" << std::endl;

    auto worker = [this](
        const typename DoFHandler<dim>::active_cell_iterator    &cell,
        Assembly::Scratch::StokesMatrix<dim>            &scratch,
        Assembly::CopyData::StokesMatrix<dim>           &data)
    {
        this->assembleStokesMatrixLocal(cell, scratch, data);
    };

    auto copier = [this](
        const Assembly::CopyData::StokesMatrix<dim> &data
    ) {
        this->copyStokesMatrixLocalToGlobal(data);
    };
    
    using CellFilter = FilteredIterator<typename 
        DoFHandler<dim>::active_cell_iterator>;

    WorkStream::run(
        CellFilter(IteratorFilters::LocallyOwnedCell(),
                   this->dof_handler_stokes.begin_active()),
        CellFilter(IteratorFilters::LocallyOwnedCell(),
                   this->dof_handler_stokes.end()),
        worker,
        copier,
        Assembly::Scratch::StokesMatrix<dim>(
            this->fe_stokes,
            this->quad_formula,
            update_values | update_JxW_values |
            update_gradients),
        Assembly::CopyData::StokesMatrix<dim>(
            this->fe_stokes
        )
    );

    this->matrix_stokes.compress(VectorOperation::add);

    this->pcout << "Block Norms:\n"
                << "\tBlock (0,0): " << this->matrix_stokes.block(0,0).frobenius_norm()
                << "\n\tBlock (0,1): " << this->matrix_stokes.block(0,1).frobenius_norm()
                << "\n\tBlock (1,0): " << this->matrix_stokes.block(1,0).frobenius_norm()
                << "\n\tBlock (1,1): " << this->matrix_stokes.block(1,1).frobenius_norm()
                << std::endl;

}

template<int dim>
void SCHSolver<dim>::assembleStokesRHSLocal(
    const typename DoFHandler<dim>::active_cell_iterator &cell,
    Assembly::Scratch::StokesRHS<dim>&        scratch,
    Assembly::CopyData::StokesRHS<dim>&       data
)
{
    data.local_rhs = 0;
    scratch.fe_val_stokes.reinit(cell);

    typename DoFHandler<dim>::active_cell_iterator cell_ch(
        &this->triangulation, cell->level(),
        cell->index(), &this->dof_handler_ch
    );

    scratch.fe_val_ch.reinit(cell_ch);

    Assert(cell->is_locally_owned() && cell_ch->is_locally_owned(),
           ExcGhostsPresent());

    cell->get_dof_indices(data.local_dof_indices);

    const uint dofs_per_cell = scratch.fe_val_stokes.get_fe().n_dofs_per_cell();
    const uint n_q_points    = scratch.fe_val_stokes.get_quadrature().size();

    FEValuesExtractors::Vector  velocities(0);
    FEValuesExtractors::Scalar  pressure(dim);
    FEValuesExtractors::Scalar  phi(0);

    scratch.fe_val_ch[phi].get_function_gradients(
        this->solution_old_ch,
        scratch.grad_phi_q
    );

    for(uint q = 0; q < n_q_points; q++)
    {
        // Build outer product
        for(uint i = 0; i < dim; i++)
        {
            for(uint j = 0; j < dim; j++)
            {
                scratch.grad_outer_phi_q[q][i][j] =
                    -scratch.grad_phi_q[q][i] * scratch.grad_phi_q[q][j];

                if(i == j) scratch.grad_outer_phi_q[q][i][j]
                    += std::pow(scratch.grad_phi_q[q].norm(),2);
            }
        }

        for(uint k = 0; k < dofs_per_cell; k++)
        {
            scratch.grad_phi_u[k] 
                = scratch.fe_val_stokes[velocities].gradient(k,q);
        }

        for(uint i = 0; i < dofs_per_cell; i++)
        {
            data.local_rhs(i) +=
                -24 * std::sqrt(2) * this->eps * 1e-2
                * scalar_product(scratch.grad_phi_u[i],
                                 scratch.grad_outer_phi_q[q])
                * scratch.fe_val_stokes.JxW(q);
        }
    }
    
}

template<int dim>
void SCHSolver<dim>::copyStokesRHSLocalToGlobal(
    const Assembly::CopyData::StokesRHS<dim> &data
)
{
    this->constraints_stokes.distribute_local_to_global(
        data.local_rhs,
        data.local_dof_indices,
        this->rhs_stokes
    );
}

template<int dim>
void SCHSolver<dim>::assembleStokesRHS()
{

    this->rhs_stokes = 0;

    this->pcout << "Assembling Stokes RHS" << std::endl;

    auto worker = [this](
        const typename DoFHandler<dim>::active_cell_iterator    &cell,
        Assembly::Scratch::StokesRHS<dim>            &scratch,
        Assembly::CopyData::StokesRHS<dim>           &data)
    {
        this->assembleStokesRHSLocal(cell, scratch, data);
    };

    auto copier = [this](
        const Assembly::CopyData::StokesRHS<dim> &data
    ) {
        this->copyStokesRHSLocalToGlobal(data);
    };
    
    using CellFilter = FilteredIterator<typename 
        DoFHandler<dim>::active_cell_iterator>;

    WorkStream::run(
        CellFilter(IteratorFilters::LocallyOwnedCell(),
                   this->dof_handler_stokes.begin_active()),
        CellFilter(IteratorFilters::LocallyOwnedCell(),
                   this->dof_handler_stokes.end()),
        worker,
        copier,
        Assembly::Scratch::StokesRHS<dim>(
            this->fe_stokes,
            this->quad_formula,
            update_values | update_JxW_values |
            update_gradients,
            this->fe_ch,
            update_values | update_JxW_values |
            update_gradients),
        Assembly::CopyData::StokesRHS<dim>(
            this->fe_stokes
        )
    );

    this->rhs_stokes.compress(VectorOperation::add);
    this->pcout << "Completed" << std::endl;

}

template<int dim>
void SCHSolver<dim>::assembleCahnHilliardMatrixLocal(
    const typename DoFHandler<dim>::active_cell_iterator &cell,
    Assembly::Scratch::CahnHilliardMatrix<dim>           &scratch,
    Assembly::CopyData::CahnHilliardMatrix<dim>          &data
)
{

    data.local_matrix = 0;

    double timestep_ratio = timestep / timestep_old;

    scratch.fe_val_ch.reinit(cell);

    cell->get_dof_indices(data.local_dof_indices);

    uint dofs_per_cell  = scratch.fe_val_ch.get_fe().n_dofs_per_cell();
    uint n_q_points     = scratch.fe_val_ch.get_quadrature().size();

    FEValuesExtractors::Scalar  phi(0);
    FEValuesExtractors::Scalar  eta(1);

    for(uint q = 0; q < n_q_points; q++)
    {

        for(uint k = 0; k < dofs_per_cell; k++)
        {
            scratch.phi_val[k]  = scratch.fe_val_ch[phi].value(k,q);
            scratch.eta_val[k]  = scratch.fe_val_ch[eta].value(k,q);

            scratch.phi_grad[k] = scratch.fe_val_ch[phi].gradient(k,q);
            scratch.eta_grad[k] = scratch.fe_val_ch[eta].gradient(k,q);
        }

        for(uint i = 0; i < dofs_per_cell; i++)
        {
            for(uint j = 0; j < dofs_per_cell; j++)
            {
                // (0,0): M
                data.local_matrix(i,j)
                    +=  (1 + 2 * timestep_ratio) / (1 + timestep_ratio)
                    *   scratch.phi_val[i] * scratch.phi_val[j]
                    *   scratch.fe_val_ch.JxW(q);
                
                // (0,1): kA
                data.local_matrix(i,j)
                    +=  this->timestep 
                    *   scratch.phi_grad[i] * scratch.eta_grad[j]
                    *   scratch.fe_val_ch.JxW(q);

                // (1,0): - (2 M + epsilon^2 A)
                data.local_matrix(i,j)
                    -=  2.0 * scratch.eta_val[i] * scratch.phi_val[j]
                        * scratch.fe_val_ch.JxW(q);

                data.local_matrix(i,j)
                    -=  pow(this->eps,2)
                        * scratch.eta_grad[i] * scratch.phi_grad[j]
                        * scratch.fe_val_ch.JxW(q); 

                // (1,1): M
                data.local_matrix(i,j)
                    +=  scratch.eta_val[i] * scratch.eta_val[j] 
                    * scratch.fe_val_ch.JxW(q);
            }
        }
    }

}

template<int dim>
void SCHSolver<dim>::copyCahnHilliardMatrixLocalToGlobal(
    const Assembly::CopyData::CahnHilliardMatrix<dim> &data
)
{
    this->constraints_ch.distribute_local_to_global(
        data.local_matrix,
        data.local_dof_indices,
        this->matrix_ch
    );
}

template<int dim>
void SCHSolver<dim>::assembleCahnHilliardMatrix()
{

    this->pcout << "Assembling Cahn-Hilliard matrix" << std::endl;

    this->matrix_ch = 0;

    auto worker = [this](
        const typename DoFHandler<dim>::active_cell_iterator    &cell,
        Assembly::Scratch::CahnHilliardMatrix<dim>              &scratch,
        Assembly::CopyData::CahnHilliardMatrix<dim>             &data)
    {
        this->assembleCahnHilliardMatrixLocal(cell, scratch, data);
    };

    auto copier = [this](
        const Assembly::CopyData::CahnHilliardMatrix<dim> &data
    ) {
        this->copyCahnHilliardMatrixLocalToGlobal(data);
    };

    using CellFilter = FilteredIterator<typename 
        DoFHandler<dim>::active_cell_iterator>;

    WorkStream::run(
        CellFilter(IteratorFilters::LocallyOwnedCell(),
                   dof_handler_ch.begin_active()),
        CellFilter(IteratorFilters::LocallyOwnedCell(),
                   dof_handler_ch.end()),
        worker,
        copier,
        Assembly::Scratch::CahnHilliardMatrix<dim>(
            this->fe_ch,
            this->quad_formula,
            update_values | update_JxW_values |
            update_gradients),
        Assembly::CopyData::CahnHilliardMatrix<dim>(
            this->fe_ch
        )
    );

    this->matrix_ch.compress(VectorOperation::add);
    this->pcout << "Completed" << std::endl;

}

template<int dim>
void SCHSolver<dim>::assembleCahnHilliardRHSLocal(
    const typename DoFHandler<dim>::active_cell_iterator &cell,
    Assembly::Scratch::CahnHilliardRHS<dim>              &scratch,
    Assembly::CopyData::CahnHilliardRHS<dim>             &data
)
{
    data.local_rhs = 0;
    scratch.fe_val_ch.reinit(cell);

    typename DoFHandler<dim>::active_cell_iterator cell_stokes(
        &this->triangulation, cell->level(),
        cell->index(), &this->dof_handler_stokes
    );
    scratch.fe_val_stokes.reinit(cell_stokes);

    cell->get_dof_indices(data.local_dof_indices);

    double timestep_ratio = timestep / timestep_old;

    const uint dofs_per_cell = scratch.fe_val_ch.get_fe().n_dofs_per_cell();
    const uint n_q_points    = scratch.fe_val_ch.get_quadrature().size();

    FEValuesExtractors::Vector  velocities(0);
    FEValuesExtractors::Scalar  phi(0);

    scratch.fe_val_ch[phi].get_function_values(
        this->solution_old_ch,
        scratch.phi_old_q
    );
    scratch.fe_val_ch[phi].get_function_values(
        this->solution_old_old_ch,
        scratch.phi_old_old_q
    );


    scratch.fe_val_ch[phi].get_function_gradients(
        this->solution_old_ch,
        scratch.phi_grad_old_q
    );
    scratch.fe_val_ch[phi].get_function_gradients(
        this->solution_old_old_ch,
        scratch.phi_grad_old_old_q
    );

    scratch.fe_val_stokes[velocities].get_function_values(
        this->solution_stokes,
        scratch.vel_old_q
    );

    for(uint q = 0; q < n_q_points; q++)
    {
        for(uint i = 0; i < dofs_per_cell; i++)
        { 
            // <\varphi_i, phi_old>
            data.local_rhs(i) +=  (1 + timestep_ratio) 
                              *  scratch.fe_val_ch[phi].value(i,q)
                              *  scratch.phi_old_q[q] 
                              *  scratch.fe_val_ch.JxW(q);

            data.local_rhs(i) -= std::pow(timestep_ratio,2) / (1 + timestep_ratio)
                              * scratch.fe_val_ch[phi].value(i,q)
                              * scratch.phi_old_old_q[q]
                              * scratch.fe_val_ch.JxW(q);

            // 3 k <\nabla\varphi_i, \nabla\phi_old>
            data.local_rhs(i)    +=  3.0 * this->timestep * (1 + timestep_ratio)
                            *   scratch.fe_val_ch[phi].gradient(i, q)
                            *   scratch.phi_grad_old_q[q]
                            *   scratch.fe_val_ch.JxW(q);
            
            data.local_rhs(i)    -=  3.0 * this->timestep * timestep_ratio
                            *   scratch.fe_val_ch[phi].gradient(i, q)
                            *   scratch.phi_grad_old_old_q[q]
                            *   scratch.fe_val_ch.JxW(q);

            // - k <\nabla\varphi_i, 3(\phi_old)^2 \nabla\phi_old>
            data.local_rhs(i)    -=  this->timestep * (1 + timestep_ratio) * (
                                    scratch.fe_val_ch[phi].gradient(i,q)
                                    * 3.0 * pow(scratch.phi_old_q[q],2) 
                                    * scratch.phi_grad_old_q[q]
                                 ) * scratch.fe_val_ch.JxW(q);

            data.local_rhs(i)    +=  this->timestep * timestep_ratio * (
                                    scratch.fe_val_ch[phi].gradient(i,q)
                                    * 3.0 * pow(scratch.phi_old_old_q[q],2) 
                                    * scratch.phi_grad_old_old_q[q]
                                 ) * scratch.fe_val_ch.JxW(q);
            
            // Advection
            data.local_rhs(i)    += this->timestep * (1 + timestep_ratio) * (
                                    scratch.fe_val_ch[phi].value(i,q) 
                                    * scratch.vel_q[q] 
                                    * scratch.phi_grad_old_q[q]
                                 ) * scratch.fe_val_ch.JxW(q);

            data.local_rhs(i)    -= this->timestep * timestep_ratio * (
                                    scratch.fe_val_ch[phi].value(i,q) 
                                    * scratch.vel_old_q[q] 
                                    * scratch.phi_grad_old_old_q[q]
                                 ) * scratch.fe_val_ch.JxW(q);
        }
    }

}

template<int dim>
void SCHSolver<dim>::copyCahnHilliardRHSLocalToGlobal(
    const Assembly::CopyData::CahnHilliardRHS<dim> &data
)
{
    this->constraints_ch.distribute_local_to_global(
        data.local_rhs,
        data.local_dof_indices,
        this->rhs_ch
    );
}

template<int dim>
void SCHSolver<dim>::assembleCahnHilliardRHS()
{
    this->pcout << "Assembling Cahn-Hilliard RHS" << std::endl;
    
    this->rhs_ch = 0;

    auto worker = [this](
        const typename DoFHandler<dim>::active_cell_iterator    &cell,
        Assembly::Scratch::CahnHilliardRHS<dim>              &scratch,
        Assembly::CopyData::CahnHilliardRHS<dim>             &data)
    {
        this->assembleCahnHilliardRHSLocal(cell, scratch, data);
    };

    auto copier = [this](
        const Assembly::CopyData::CahnHilliardRHS<dim> &data
    ) {
        this->copyCahnHilliardRHSLocalToGlobal(data);
    };

    using CellFilter = FilteredIterator<typename 
        DoFHandler<dim>::active_cell_iterator>;

    WorkStream::run(
        CellFilter(IteratorFilters::LocallyOwnedCell(),
                   dof_handler_ch.begin_active()),
        CellFilter(IteratorFilters::LocallyOwnedCell(),
                   dof_handler_ch.end()),
        worker,
        copier,
        Assembly::Scratch::CahnHilliardRHS<dim>(
            this->fe_ch,
            this->quad_formula,
            update_values | update_JxW_values |
            update_gradients,
            this->fe_stokes,
            update_values | update_JxW_values),
        Assembly::CopyData::CahnHilliardRHS<dim>(
            this->fe_ch
        )
    );

    this->rhs_ch.compress(VectorOperation::add);
    this->pcout << "Completed" << std::endl;

}

template<int dim>
void SCHSolver<dim>::solveStokes()
{

    this->pcout << "Solving Stokes system... " << std::endl;

    TrilinosWrappers::MPI::BlockVector  locally_owned_solution;

    locally_owned_solution.reinit(this->rhs_stokes);
    locally_owned_solution = this->solution_stokes;

    TrilinosWrappers::PreconditionBlockwiseDirect prec_A;
    prec_A.initialize(this->matrix_stokes.block(0,0));

    TrilinosWrappers::PreconditionBlockwiseDirect prec_Mp;
    prec_Mp.initialize(this->precon_stokes.block(1,1));

    LinearSolvers::BlockDiagonalPreconditioner<
        TrilinosWrappers::PreconditionBlockwiseDirect,
        TrilinosWrappers::PreconditionBlockwiseDirect> 
            stokes_precon(prec_A, prec_Mp);

    SolverControl sc(100000, 
                     std::max(1e-6,
                              1e-6 * this->rhs_stokes.l2_norm())
                     );

    SolverFGMRES<TrilinosWrappers::MPI::BlockVector> solver(sc);

    solver.solve(this->matrix_stokes,
                 locally_owned_solution,
                 this->rhs_stokes,
                 stokes_precon);

    this->constraints_stokes.distribute(locally_owned_solution);
    this->solution_stokes = locally_owned_solution;

    this->pcout << "Solution norms:\n"
                << "\tBlock 0: "
                    << this->solution_stokes.block(0).linfty_norm()
                << "\n\tBlock 1: " 
                    << this->solution_stokes.block(1).linfty_norm()
                << std::endl;

    this->pcout << "Completed" << std::endl;

}

template<int dim>
void SCHSolver<dim>::computeTimestep()
{

}

template<int dim>
void SCHSolver<dim>::solveCahnHilliard()
{

    this->pcout << "Solving Cahn-Hilliard system... " << std::endl;

    TrilinosWrappers::MPI::BlockVector  locally_owned_solution;

    locally_owned_solution.reinit(this->rhs_ch);
    locally_owned_solution = this->solution_ch;

    TrilinosWrappers::PreconditionSSOR prec_A;
    TrilinosWrappers::PreconditionSSOR prec_D;

    prec_A.initialize(this->matrix_ch.block(0,0));
    prec_D.initialize(this->matrix_ch.block(1,1));
    
    LinearSolvers::BlockDiagonalPreconditioner<
        TrilinosWrappers::PreconditionSSOR,
        TrilinosWrappers::PreconditionSSOR> ch_precon(prec_A, prec_D);

    SolverControl sc(100000,
                     std::max(1e-8,
                              1e-8 * this->rhs_ch.l2_norm()));

    SolverGMRES<TrilinosWrappers::MPI::BlockVector> solver(sc);
    solver.solve(this->matrix_ch,
                 locally_owned_solution,
                 this->rhs_ch,
                 ch_precon);

    this->constraints_ch.distribute(locally_owned_solution);
    this->solution_ch = locally_owned_solution;

    this->pcout << "Completed" << std::endl;

}

template<int dim>
void SCHSolver<dim>::refineGrid()
{
    parallel::distributed::SolutionTransfer<
        dim, TrilinosWrappers::MPI::BlockVector> trans_sol_stokes(
            this->dof_handler_stokes);
    
    parallel::distributed::SolutionTransfer<
        dim, TrilinosWrappers::MPI::BlockVector> trans_sol_ch(
            this->dof_handler_ch);

    {
    Vector<float>               estimated_errors_stokes(
                                    triangulation.n_active_cells());
    Vector<float>               estimated_errors_stokes_rhs(
                                    triangulation.n_active_cells());
    Vector<float>               estimated_errors_ch(
                                    triangulation.n_active_cells());

    KellyErrorEstimator<dim>::estimate(
        this->dof_handler_stokes,
        QGauss<dim - 1>(degree + 1),
        std::map<types::boundary_id, const Function<dim> *>(),
        this->solution_stokes,
        estimated_errors_stokes,
        ComponentMask(),
        nullptr,
        0,
        triangulation.locally_owned_subdomain()
    );

    KellyErrorEstimator<dim>::estimate(
        this->dof_handler_ch,
        QGauss<dim - 1>(degree + 1),
        std::map<types::boundary_id, const Function<dim> *>(),
        this->solution_ch,
        estimated_errors_ch,
        ComponentMask(),
        nullptr,
        0,
        triangulation.locally_owned_subdomain()
    );

    if(estimated_errors_stokes.l2_norm() > 0)
        estimated_errors_stokes /= estimated_errors_stokes.l2_norm();
    if(estimated_errors_ch.l2_norm() > 0)
        estimated_errors_ch /= estimated_errors_ch.l2_norm();

    Vector<float> max_err(triangulation.n_active_cells());
    for(uint i = 0; i < max_err.size(); i++)
        max_err[i] = std::max(estimated_errors_stokes[i],
                              estimated_errors_ch[i]);

    this->pcout << "Performing refinement..." << std::endl;

    parallel::distributed::GridRefinement::refine_and_coarsen_fixed_fraction(
        this->triangulation, max_err, 0.8, 0.1);

    this->pcout << "Lock 1" << std::endl;

    std::vector<const TrilinosWrappers::MPI::BlockVector *> x_ch(3);
    x_ch[0] = &this->solution_ch;
    x_ch[1] = &this->solution_old_ch;
    x_ch[2] = &this->solution_old_old_ch;
    
    std::vector<const TrilinosWrappers::MPI::BlockVector *> x_stokes(2);
    x_stokes[0] = &this->solution_stokes;
    x_stokes[1] = &this->solution_old_stokes;

    this->pcout << "Lock 2" << std::endl;

    this->triangulation.prepare_coarsening_and_refinement();

    trans_sol_ch.prepare_for_coarsening_and_refinement(x_ch);
    trans_sol_stokes.prepare_for_coarsening_and_refinement(x_stokes);

    this->pcout << "Lock Lomond" << std::endl;

    this->triangulation.execute_coarsening_and_refinement();
    }

    this->setupDoFs();
    
    {
        TrilinosWrappers::MPI::BlockVector distributed_tmp1(this->rhs_ch);
        TrilinosWrappers::MPI::BlockVector distributed_tmp2(this->rhs_ch);
        TrilinosWrappers::MPI::BlockVector distributed_tmp3(this->rhs_ch);

        std::vector<TrilinosWrappers::MPI::BlockVector *> tmp(3);
        tmp[0] = &(distributed_tmp1);
        tmp[1] = &(distributed_tmp2);
        tmp[2] = &(distributed_tmp3);
        
        trans_sol_ch.interpolate(tmp);

        this->constraints_ch.distribute(distributed_tmp1);
        this->constraints_ch.distribute(distributed_tmp2);
        this->constraints_ch.distribute(distributed_tmp3);

        this->solution_ch = distributed_tmp1;
        this->solution_old_ch = distributed_tmp2;
        this->solution_old_old_ch = distributed_tmp3;
    }

    {
        TrilinosWrappers::MPI::BlockVector distributed_tmp1(this->rhs_stokes);
        TrilinosWrappers::MPI::BlockVector distributed_tmp2(this->rhs_stokes);

        std::vector<TrilinosWrappers::MPI::BlockVector *> tmp(2);
        tmp[0] = &(distributed_tmp1);
        tmp[1] = &(distributed_tmp2);
        
        trans_sol_stokes.interpolate(tmp);

        this->constraints_stokes.distribute(distributed_tmp1);
        this->constraints_stokes.distribute(distributed_tmp2);

        this->solution_stokes = distributed_tmp1;
        this->solution_old_stokes = distributed_tmp2;
    }

}

template<int dim>
void SCHSolver<dim>::outputStokes()
{
    
    this->pcout << "Generating stokes RHS output" << std::endl;
    std::vector<std::string> component_names_stokes_rhs(dim, "surface_tension");
    component_names_stokes_rhs.emplace_back("pressure_rhs");

    std::vector<DataComponentInterpretation::DataComponentInterpretation>
        data_interp(dim, DataComponentInterpretation::component_is_part_of_vector);
    data_interp.push_back(DataComponentInterpretation::component_is_scalar);
    
    std::vector<std::string> component_names_stokes_sol(dim, "velocity");
    component_names_stokes_sol.emplace_back("pressure");

    std::vector<uint> stokes_sub_blocks(dim+1,0);
    stokes_sub_blocks[dim] = 1;

    const std::vector<types::global_dof_index> dofs_per_block =
        DoFTools::count_dofs_per_fe_block(this->dof_handler_stokes,
                                          stokes_sub_blocks);

    const types::global_dof_index n_u = dofs_per_block[0],
                                  n_p = dofs_per_block[1];

    std::vector<IndexSet> stokes_relevant_partitioning;
    
    IndexSet stokes_relevant_set;
    DoFTools::extract_locally_relevant_dofs(this->dof_handler_stokes,
                                            stokes_relevant_set);
    stokes_relevant_partitioning.push_back(
        stokes_relevant_set.get_view(0, n_u));
    stokes_relevant_partitioning.push_back(
        stokes_relevant_set.get_view(n_u, n_u+n_p));
    
    TrilinosWrappers::MPI::BlockVector locally_rel_stokes_rhs;
    TrilinosWrappers::MPI::BlockVector locally_rel_stokes_sol;
    locally_rel_stokes_rhs.reinit(stokes_relevant_partitioning,
                                  mpi_communicator);
    locally_rel_stokes_sol.reinit(stokes_relevant_partitioning,
                                  mpi_communicator);

    locally_rel_stokes_rhs = this->rhs_stokes;
    locally_rel_stokes_sol = this->solution_stokes;

    DataOut<dim> data_out;
    data_out.add_data_vector(this->dof_handler_stokes,
                             locally_rel_stokes_rhs,
                             component_names_stokes_rhs,
                             data_interp);
    data_out.add_data_vector(this->dof_handler_stokes,
                             locally_rel_stokes_sol,
                             component_names_stokes_sol,
                             data_interp);

    Vector<float> subdomain(triangulation.n_active_cells());
    for (unsigned int i = 0; i < subdomain.size(); ++i)
        subdomain(i) = triangulation.locally_owned_subdomain();
    data_out.add_data_vector(subdomain, "subdomain");

    data_out.build_patches();

    data_out.write_vtu_with_pvtu_record("data/", "surface_tension", 0, 
                                        mpi_communicator, 2);


}

template<int dim>
void SCHSolver<dim>::outputTimestep(const uint timestep_number)
{
        TrilinosWrappers::MPI::BlockVector locally_rel_stokes_rhs;
        TrilinosWrappers::MPI::BlockVector locally_rel_stokes_sol;
        TrilinosWrappers::MPI::BlockVector locally_rel_ch_sol;

    {
        std::vector<uint> stokes_sub_blocks(dim+1,0);
        stokes_sub_blocks[dim] = 1;

        const std::vector<types::global_dof_index> dofs_per_block =
            DoFTools::count_dofs_per_fe_block(this->dof_handler_stokes,
                                              stokes_sub_blocks);

        const types::global_dof_index n_u = dofs_per_block[0],
                                      n_p = dofs_per_block[1];

        std::vector<IndexSet> stokes_relevant_partitioning;
        
        IndexSet stokes_relevant_set;
        DoFTools::extract_locally_relevant_dofs(this->dof_handler_stokes,
                                                stokes_relevant_set);
        stokes_relevant_partitioning.push_back(
            stokes_relevant_set.get_view(0, n_u));
        stokes_relevant_partitioning.push_back(
            stokes_relevant_set.get_view(n_u, n_u+n_p));
        
        locally_rel_stokes_rhs.reinit(stokes_relevant_partitioning,
                                      mpi_communicator);
        locally_rel_stokes_sol.reinit(stokes_relevant_partitioning,
                                      mpi_communicator);

        locally_rel_stokes_rhs = this->rhs_stokes;
        locally_rel_stokes_sol = this->solution_stokes;
    }
    
    {
        std::vector<uint> ch_sub_blocks = {0,1};

        const std::vector<types::global_dof_index> dofs_per_block =
            DoFTools::count_dofs_per_fe_block(this->dof_handler_ch,
                                              ch_sub_blocks);

        const types::global_dof_index n_phi = dofs_per_block[0],
                                      n_eta = dofs_per_block[1];

        std::vector<IndexSet> ch_relevant_partitioning;
        
        IndexSet ch_relevant_set;
        DoFTools::extract_locally_relevant_dofs(this->dof_handler_ch,
                                                ch_relevant_set);
        ch_relevant_partitioning.push_back(
            ch_relevant_set.get_view(0, n_phi));
        ch_relevant_partitioning.push_back(
            ch_relevant_set.get_view(n_phi, n_phi+n_eta));
        
        locally_rel_ch_sol.reinit(ch_relevant_partitioning,
                                      mpi_communicator);

        locally_rel_ch_sol = this->solution_ch;
    }

    this->pcout << "Saving solution at timestep number: " 
                << timestep_number
                << std::endl;
    std::vector<std::string> component_names_stokes_rhs(dim, "surface_tension");
    component_names_stokes_rhs.emplace_back("pressure_rhs");

    std::vector<DataComponentInterpretation::DataComponentInterpretation>
        data_interp_stokes(
            dim, DataComponentInterpretation::component_is_part_of_vector
        );
    data_interp_stokes.push_back(
        DataComponentInterpretation::component_is_scalar
    );
    
    std::vector<std::string> component_names_stokes_sol(dim, "velocity");
    component_names_stokes_sol.emplace_back("pressure");

    std::vector<std::string> component_names_ch_sol = {"phi", "eta"};
    std::vector<DataComponentInterpretation::DataComponentInterpretation>
        data_interp_ch(2, DataComponentInterpretation::component_is_scalar);

    DataOut<dim> data_out;
    data_out.add_data_vector(this->dof_handler_stokes,
                             locally_rel_stokes_rhs,
                             component_names_stokes_rhs,
                             data_interp_stokes);
    data_out.add_data_vector(this->dof_handler_stokes,
                             locally_rel_stokes_sol,
                             component_names_stokes_sol,
                             data_interp_stokes);
    data_out.add_data_vector(this->dof_handler_ch,
                             locally_rel_ch_sol,
                             component_names_ch_sol,
                             data_interp_ch);

    Vector<float> subdomain(triangulation.n_active_cells());
    for (unsigned int i = 0; i < subdomain.size(); ++i)
        subdomain(i) = triangulation.locally_owned_subdomain();
    data_out.add_data_vector(subdomain, "subdomain");

    data_out.build_patches();

    data_out.write_vtu_with_pvtu_record("data/", "solution", timestep_number, 
                                        mpi_communicator, 2);

}

template<int dim>
void SCHSolver<dim>::run(bool debug)
{
   
    this->pcout << "Running" << std::endl;
    this->setupTriang();
    this->setupDoFs();
    for(uint i = 0; i < 6; i++)
    {
        this->initializeValues();
        this->assembleStokesPrecon();
        this->assembleStokesMatrix();
        this->assembleStokesRHS();
        this->solveStokes();

        if(debug) this->outputStokes();

        this->assembleCahnHilliardMatrix();
        this->assembleCahnHilliardRHS();
        this->solveCahnHilliard();

        this->outputTimestep(i);

        this->refineGrid();

    }
}

} // stokesCahnHilliard

int main(int argc, char *argv[]){
    try{

        dealii::Utilities::MPI::MPI_InitFinalize mpi_initialize(
            argc, argv, 2
        );

        stokesCahnHilliard::SCHSolver<2> stokesCahnHilliard;
        stokesCahnHilliard.run(true);

    }
    catch (std::exception &exc)
    {
        std::cerr << std::endl
                  << std::endl
                  << "----------------------------------------------------"
                  << std::endl;
        std::cerr << "Exception on processing: " << std::endl
                  << exc.what() << std::endl
                  << "Aborting!" << std::endl
                  << "----------------------------------------------------"
                  << std::endl;

        return 1;
    }
    catch (...)
    {
        std::cerr << std::endl
                  << std::endl
                  << "----------------------------------------------------"
                  << std::endl;
        std::cerr << "Unknown exception!" << std::endl
                  << "Aborting!" << std::endl
                  << "----------------------------------------------------"
                  << std::endl;
        return 1;
    }
    return 0;
}
