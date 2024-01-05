// Deal.II Libraries
#include <algorithm>
#include <boost/iostreams/categories.hpp>

#include <deal.II/base/data_out_base.h>
#include <deal.II/base/quadrature.h>
#include <deal.II/base/subscriptor.h>
#include <deal.II/base/symmetric_tensor.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/types.h>
#include <deal.II/base/function.h>
#include <deal.II/base/work_stream.h>

#include <deal.II/base/utilities.h>
#include <deal.II/fe/fe.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>

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

#include <ostream>
#include <random>
#include <unordered_map>
#include <fstream>

namespace stokesCahnHilliard {
    using namespace dealii;

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

        shifted_p1[0] -= 0.25;
        shifted_p1[1] -= 0.25;

        shifted_p2[0] += 0.25;
        shifted_p2[1] += 0.25;

        double droplet_1 = std::tanh(
            (shifted_p1.norm()-0.25) / (std::sqrt(2) * this->eps)
        );

        double droplet_2 = std::tanh(
                (shifted_p2.norm()-0.25) / (std::sqrt(2) * this->eps)
        );

        return droplet_1 * droplet_2;

    } else {
        return 0;
    }
}

} // EquationData

namespace LinearSolver
{
    template<int dim>
    struct InnerPreconditioner
    {
        using type = SparseILU<double>;
    };

    template<class MatrixType, class PreconditionerType>
    class InverseMatrix : public Subscriptor
    {
    public:
        InverseMatrix(
            const MatrixType            &m,
            const PreconditionerType    &preconditioner
        );

        void vmult(Vector<double> &dst, const Vector<double> &src) const;

    private:
        const SmartPointer<const MatrixType>            matrix;
        const SmartPointer<const PreconditionerType>    preconditioner;
    };

    template<class MatrixType, class PreconditionerType>
    InverseMatrix<MatrixType, PreconditionerType>::InverseMatrix(
    const MatrixType            &m,
    const PreconditionerType    &preconditioner
    )
    : matrix(&m)
    , preconditioner(&preconditioner)
    {}

    template<class MatrixType, class PreconditionerType>
    void InverseMatrix<MatrixType, PreconditionerType>::vmult(
        Vector<double>          &dst,
        const Vector<double>    &src
    ) const
    {
        SolverControl               solver_control(src.size(),
                                                   1e-6 * src.l2_norm());
        SolverCG<Vector<double>>    cg(solver_control);

        dst = 0;

        cg.solve(*matrix, dst, src, *preconditioner);
    }

    template <class PreconditionerType>
    class SchurComplement : public Subscriptor
    {
    public:
        SchurComplement(
            const BlockSparseMatrix<double> &system_matrix,
            const InverseMatrix<
                SparseMatrix<double>,
                PreconditionerType> &A_inverse
        );

        void vmult(Vector<double> &dst, const Vector<double> &src) const;

    private:
        const SmartPointer<const BlockSparseMatrix<double>> system_matrix;
        const SmartPointer<
        const InverseMatrix<SparseMatrix<double>, PreconditionerType>>
        A_inverse;

        mutable Vector<double> tmp1, tmp2;
    };
  
    template <class PreconditionerType>
    SchurComplement<PreconditionerType>::SchurComplement(
        const BlockSparseMatrix<double> &system_matrix,
        const InverseMatrix<SparseMatrix<double>, PreconditionerType> &A_inverse
    ) : system_matrix(&system_matrix)
    , A_inverse(&A_inverse)
    , tmp1(system_matrix.block(0, 0).m())
    , tmp2(system_matrix.block(0, 0).m())
    {}

    template <class PreconditionerType>
    void
    SchurComplement<PreconditionerType>::vmult(Vector<double> &      dst,
                                               const Vector<double> &src) const
    {
        // B * A * B^T * src
        system_matrix->block(0, 1).vmult(tmp1, src);
        A_inverse->vmult(tmp2, tmp1);
        system_matrix->block(1, 0).vmult(dst, tmp2);
    }
} // LinearSolver

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
        std::vector<Tensor<1,dim>>  phi_grad_old_q;

        std::vector<Tensor<1,dim>>   vel_old_q;
 
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
    , phi_grad_old_q(quad_ch.size())
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
    , phi_grad_old_q(scratch.phi_grad_old_q)
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
        SCHSolver(const uint degree, const bool debug=false,
                  const uint max_grid_level = 9,
                  const uint min_grid_level = 3);
        void run(const std::unordered_map<std::string, double> params,
                 const double                                  total_sim_time);

    private:
        void setupParameters(
            const std::unordered_map<std::string, double> params
        );
        void setupSystem(
            const std::unordered_map<std::string, double> params,
            const double                                  total_sim_time
        );
        void setupTriang();
        void setupDoFs();
        void setupLinearSystems();
        void initializeValues();
        void refineGrid();

        // Local assembly
        void assembleStokesPreconLocal(
            const typename DoFHandler<dim>::active_cell_iterator &cell,
            Assembly::Scratch::StokesPreconditioner<dim>&        scratch,
            Assembly::CopyData::StokesPreconditioner<dim>&       data
        );
        void assembleStokesMatrixLocal(
            const typename DoFHandler<dim>::active_cell_iterator &cell,
            Assembly::Scratch::StokesMatrix<dim>&        scratch,
            Assembly::CopyData::StokesMatrix<dim>&       data
        );
        void assembleStokesRHSLocal(
            const typename DoFHandler<dim>::active_cell_iterator &cell,
            Assembly::Scratch::StokesRHS<dim>&        scratch,
            Assembly::CopyData::StokesRHS<dim>&       data
        );

        // Copiers
        void copyStokesPreconLocalToGlobal(
            const Assembly::CopyData::StokesPreconditioner<dim> & data
        );
        void copyStokesMatrixLocalToGlobal(
            const Assembly::CopyData::StokesMatrix<dim> & data
        );
        void copyStokesRHSLocalToGlobal(
            const Assembly::CopyData::StokesRHS<dim> &data
        );

        // Assembly
        void assembleStokesPrecon();
        void assembleStokesMatrix();
        void assembleStokesRHS();
        void assembleCahnHilliard();

        void assembleCahnHilliardMatrixLocal(
            const typename DoFHandler<dim>::active_cell_iterator &cell,
            Assembly::Scratch::CahnHilliardMatrix<dim>           &scratch,
            Assembly::CopyData::CahnHilliardMatrix<dim>          &data
        );
        void assembleCahnHilliardRHSLocal(
            const typename DoFHandler<dim>::active_cell_iterator &cell,
            Assembly::Scratch::CahnHilliardRHS<dim>              &scratch,
            Assembly::CopyData::CahnHilliardRHS<dim>             &data
        );

        void copyCahnHilliardMatrixLocalToGlobal(
            const Assembly::CopyData::CahnHilliardMatrix<dim> &data
        );
        void copyCahnHilliardRHSLocalToGlobal(
            const Assembly::CopyData::CahnHilliardRHS<dim> &data
        );

        void assembleCahnHilliardMatrix();
        void assembleCahnHilliardRHS();
        
        void solveStokes();
        void solveCahnHilliard();

        void outputSurfaceTension();
        void outputStokes(const uint timestep_number);
        void outputCahnHilliard(const uint timestep_number);
        void outputTimestep(const uint timepstep_number);

        uint        degree;
        Triangulation<dim>  triangulation;
        FESystem<dim>       fe_stokes;
        FESystem<dim>       fe_ch;
        QGauss<dim>         quad_formula;

        DoFHandler<dim>     dof_handler_stokes;
        DoFHandler<dim>     dof_handler_ch;

        AffineConstraints<double>   constraints_stokes;
        AffineConstraints<double>   constraints_ch;
        AffineConstraints<double>   constraints_pressure;

        BlockSparsityPattern        sparsity_pattern_stokes;
        BlockSparseMatrix<double>   system_matrix_stokes;
        BlockVector<double>         solution_stokes;
        BlockVector<double>         rhs_stokes;
        BlockSparseMatrix<double>   system_matrix_precon;

        BlockSparsityPattern         sparsity_pattern_ch;
        BlockSparseMatrix<double>    system_matrix_ch;
        BlockVector<double>          solution_ch;
        BlockVector<double>          solution_old_ch;
        BlockVector<double>          rhs_ch;
        
        double      timestep;
        double      time;
        uint        timestep_number;
        double      total_simulation_time;

        double gamma;
        double rho_0;
        double rho_1;
        double nu_0;
        double nu_1;

        double eps;
        bool   debug;

        const uint max_grid_level;
        const uint min_grid_level;

    };

template<int dim>
SCHSolver<dim>::SCHSolver(const uint degree, const bool debug, 
                          const uint max_grid_level,
                          const uint min_grid_level)
: degree(degree)
, triangulation(Triangulation<dim>::maximum_smoothing)
, fe_stokes(FE_Q<dim>(degree+1), dim, FE_Q<dim>(degree), 1)
, fe_ch(FE_Q<dim>(degree), 2)
, quad_formula(degree+2)
, dof_handler_stokes(triangulation)
, dof_handler_ch(triangulation)
, timestep(1e-4)
, time(timestep)
, timestep_number(1)
, debug(debug)
, max_grid_level(max_grid_level)
, min_grid_level(min_grid_level)
{}

template<int dim>
void SCHSolver<dim>::setupParameters(
    std::unordered_map<std::string, double> params
)
{

    std::cout << "Passing parameters:" << std::endl;
    for(auto it=params.begin(); it!=params.end(); it++){
        std::cout   << "    "   << it->first
                    << ": "     << it->second
                    << std::endl;
    }

    this->eps   = params.at("eps");
    this->gamma = params.at("gamma");

}

template<int dim>
void SCHSolver<dim>::setupTriang()
{ 
    std::cout << "Generating triangulation... " << std::endl;
    
    GridGenerator::hyper_cube(
        this->triangulation,
        -1, 1,
        true
    );

    std::cout   << "\tConnecting nodes to neighbours due to periodic boundary"
                << " conditions."
                << std::endl;
    
    if(dim == 2){

        std::vector<GridTools::PeriodicFacePair<
            typename Triangulation<dim>::cell_iterator>
        > matched_pairs_X;

        std::vector<GridTools::PeriodicFacePair<
            typename Triangulation<dim>::cell_iterator>
        > matched_pairs_Y;

        GridTools::collect_periodic_faces(this->triangulation,
                                          0, 1, 0, matched_pairs_X);
        
        GridTools::collect_periodic_faces(this->triangulation,
                                          2, 3, 1, matched_pairs_Y);

        triangulation.add_periodicity(matched_pairs_X);
        triangulation.add_periodicity(matched_pairs_Y);

    } else if (dim == 3) {

        std::vector<GridTools::PeriodicFacePair<
            typename Triangulation<dim>::cell_iterator>
        > matched_pairs_X;

        std::vector<GridTools::PeriodicFacePair<
            typename Triangulation<dim>::cell_iterator>
        > matched_pairs_Y;
        
        std::vector<GridTools::PeriodicFacePair<
            typename Triangulation<dim>::cell_iterator>
        > matched_pairs_Z;

        GridTools::collect_periodic_faces(this->triangulation,
                                          0, 1, 0, matched_pairs_X);
        
        GridTools::collect_periodic_faces(this->triangulation,
                                          2, 3, 1, matched_pairs_Y);

        GridTools::collect_periodic_faces(this->triangulation,
                                          4, 5, 2, matched_pairs_Z);

        triangulation.add_periodicity(matched_pairs_X);
        triangulation.add_periodicity(matched_pairs_Y);
        triangulation.add_periodicity(matched_pairs_Z);
    }

    std::cout << "\tNeighbours updated to reflect periodicity" << std::endl;

    std::cout << "\tRefining grid" << std::endl;
    triangulation.refine_global(6);

    std::cout   << "\tActive cells: " << triangulation.n_active_cells()
                << std::endl;

    std::cout << "Completed." << std::endl;
}

template<int dim>
void SCHSolver<dim>::setupDoFs()
{
    std::cout << "Setting up DoFs for Stokes portion" << std::endl;

    {
    std::vector<uint> block_component(dim+1,0);
    block_component[dim] = 1;

    std::cout << "\tDistributing..." << std::endl;
    this->dof_handler_stokes.distribute_dofs(this->fe_stokes);

    std::cout << "\tRenumbering..." << std::endl;
    DoFRenumbering::Cuthill_McKee(this->dof_handler_stokes);
    DoFRenumbering::component_wise(
        this->dof_handler_stokes,
        block_component
    );
    
    const std::vector<types::global_dof_index> dofs_per_block =
        DoFTools::count_dofs_per_fe_block(
            this->dof_handler_stokes,
            block_component
        );
    
    const std::vector<types::global_dof_index> block_sizes = 
        {dofs_per_block[0],
         dofs_per_block[1]};
    
    std::cout   << "\tNumber of degrees of freedom: "
                << dof_handler_stokes.n_dofs()
                << std::endl;
    std::cout   << "\tPer block:\n"
                << "\t\tBlock 0: " << dofs_per_block[0] << std::endl
                << "\t\tBlock 1: " << dofs_per_block[1] << std::endl;

    std::cout << "\tUpdating constraints" << std::endl;
    this->constraints_stokes.clear();
    this->constraints_pressure.clear();

    FEValuesExtractors::Scalar pressure(dim);
    ComponentMask pressure_mask = this->fe_stokes.component_mask(pressure);

    // To make the pressure unique, we add a constraint that the mean vanishes
    const IndexSet dofs = DoFTools::extract_dofs(this->dof_handler_stokes,
                                                 pressure_mask);
    const types::global_dof_index first_pressure_dof 
            = dofs.nth_index_in_set(0);
    this->constraints_pressure.add_line(first_pressure_dof);

    // Add the hanging node constraints
    DoFTools::make_hanging_node_constraints(
        this->dof_handler_stokes,
        this->constraints_stokes
    );
    
    // Add the periodic constraints
    std::vector<GridTools::PeriodicFacePair<
        typename DoFHandler<dim>::cell_iterator>
    > periodicity_vectorX;

    std::vector<GridTools::PeriodicFacePair<
        typename DoFHandler<dim>::cell_iterator>
    > periodicity_vectorY;

    GridTools::collect_periodic_faces(this->dof_handler_stokes,
                                      0,1,0,periodicity_vectorX);
    GridTools::collect_periodic_faces(this->dof_handler_stokes,
                                      2,3,1,periodicity_vectorY);

    DoFTools::make_periodicity_constraints<dim,dim>(periodicity_vectorX,
                                                    this->constraints_stokes);
    DoFTools::make_periodicity_constraints<dim,dim>(periodicity_vectorY,
                                                    this->constraints_stokes);

    this->constraints_stokes.merge(this->constraints_pressure,
                                   AffineConstraints<double>::no_conflicts_allowed);

    std::cout << "\tClosing constraints" << std::endl;
    this->constraints_pressure.close();
    this->constraints_stokes.close();
    
    std::cout   << "\tBuilding sparsity pattern..."
                << std::endl;

    BlockDynamicSparsityPattern dsp(block_sizes, block_sizes);
    DoFTools::make_sparsity_pattern(
        this->dof_handler_stokes,
        dsp,
        this->constraints_stokes,
        true);

    sparsity_pattern_stokes.copy_from(dsp);

    std::cout << "Completed Stokes portion" << std::endl;
    }

    std::cout << "Setting up DoFs for Cahn-Hilliard potion" << std::endl;
    {
    std::vector<uint> block_component(2,0);
    block_component[1] = 1;
    
    std::cout << "\tDistributing..." << std::endl;
    this->dof_handler_ch.distribute_dofs(this->fe_ch);
    
    std::cout << "\tRenumbering..." << std::endl;
    DoFRenumbering::Cuthill_McKee(this->dof_handler_ch);
    DoFRenumbering::component_wise(
            this->dof_handler_ch,
            block_component
    );
    
    const std::vector<types::global_dof_index> dofs_per_component =
        DoFTools::count_dofs_per_fe_component(this->dof_handler_ch);
    
    const std::vector<types::global_dof_index> block_sizes = 
        {dofs_per_component[0],
         dofs_per_component[1]};

    std::cout   << "Number of degrees of freedom: "
                << this->dof_handler_ch.n_dofs()
                << std::endl;
    std::cout   << "Per block:\n"
                << "    Block 0: " << dofs_per_component[0] << std::endl
                << "    Block 1: " << dofs_per_component[1] << std::endl;

    std::cout   << "\tUpdating constraints" << std::endl;
    this->constraints_ch.clear();

    DoFTools::make_hanging_node_constraints(
        this->dof_handler_ch,
        this->constraints_ch
    );
    
    std::vector<GridTools::PeriodicFacePair<
        typename DoFHandler<dim>::cell_iterator>
    > periodicity_vectorX;

    std::vector<GridTools::PeriodicFacePair<
        typename DoFHandler<dim>::cell_iterator>
    > periodicity_vectorY;

    GridTools::collect_periodic_faces(this->dof_handler_ch,
                                      0,1,0,periodicity_vectorX);
    GridTools::collect_periodic_faces(this->dof_handler_ch,
                                      2,3,1,periodicity_vectorY);

    DoFTools::make_periodicity_constraints<dim,dim>(periodicity_vectorX,
                                                    this->constraints_ch);
    DoFTools::make_periodicity_constraints<dim,dim>(periodicity_vectorY,
                                                    this->constraints_ch);
    
    std::cout << "\tClosing constraints" << std::endl;
    this->constraints_ch.close();

    std::cout   << "\tBuilding sparsity pattern..."
                << std::endl;
    BlockDynamicSparsityPattern dsp(block_sizes, block_sizes);

    DoFTools::make_sparsity_pattern(this->dof_handler_ch,
                                    dsp,
                                    this->constraints_ch,
                                    true);

    this->sparsity_pattern_ch.copy_from(dsp);

    std::cout << "Completed Cahn-Hillaird portion" << std::endl;
    }

    std::cout   << "Total degrees of freedom: "
                << (this->dof_handler_stokes.n_dofs() 
                    + this->dof_handler_ch.n_dofs())
                << std::endl;
}

template<int dim>
void SCHSolver<dim>::setupLinearSystems()
{
    
    std::cout   << "Setting up matrix and vector objects for Stokes portion"
                << std::endl;
    { 
    std::vector<uint> block_component(dim+1, 0);
    block_component[dim] = 1;

    std::vector<types::global_dof_index> dofs_per_block 
        = DoFTools::count_dofs_per_fe_block(this->dof_handler_stokes,
                                            block_component);

    const std::vector<types::global_dof_index> block_sizes =
        {dofs_per_block[0],
         dofs_per_block[1]};

    this->system_matrix_stokes.reinit(this->sparsity_pattern_stokes);
    this->system_matrix_precon.reinit(this->sparsity_pattern_stokes);
    this->solution_stokes.reinit(block_sizes);
    this->rhs_stokes.reinit(block_sizes);
    }
    std::cout << "Completed." << std::endl;

    std::cout   << "Setting up matrix and vector objects for Cahn-Hilliard "
                << "portion" << std::endl;
    {

    const std::vector<types::global_dof_index> dofs_per_component =
        DoFTools::count_dofs_per_fe_component(this->dof_handler_ch);
    
    const std::vector<types::global_dof_index> block_sizes = 
        {dofs_per_component[0],
         dofs_per_component[1]};

    this->system_matrix_ch.reinit(this->sparsity_pattern_ch);
    this->solution_ch.reinit(block_sizes);
    this->solution_old_ch.reinit(block_sizes);
    this->rhs_ch.reinit(block_sizes);

    }
    std::cout << "Completed" << std::endl;

}

template<int dim>
void SCHSolver<dim>::initializeValues()
{
    std::cout << "Initializing values for phi" << std::endl;
    
    VectorTools::interpolate(this->dof_handler_ch,
                             EquationData::InitialValuesPhi<dim>(this->eps),
                             this->solution_old_ch);

    this->constraints_ch.distribute(this->solution_old_ch);
    this->solution_ch = this->solution_old_ch;

    auto phi_range = std::minmax_element(
        this->solution_old_ch.block(0).begin(),
        this->solution_old_ch.block(0).end());
    auto eta_range = std::minmax_element(
        this->solution_old_ch.block(1).begin(),
        this->solution_old_ch.block(1).end());


    std::cout   << "Initial values propagated:\n"
                << "    Phi Range: (" 
                    << *phi_range.first << ", " 
                    << *phi_range.second
                << ")" 
                << std::endl;
    std::cout   << "    Eta Range: (" 
                    << *eta_range.first << ", " 
                    << *eta_range.second
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
        this->system_matrix_precon
    );
}

template<int dim>
void SCHSolver<dim>::assembleStokesPrecon()
{

    std::cout << "Assembling Stokes preconditioner..." << std::endl;

    this->system_matrix_precon = 0;

    auto worker = [this](
        const typename DoFHandler<dim>::active_cell_iterator &cell,
        Assembly::Scratch::StokesPreconditioner<dim>&        scratch,
        Assembly::CopyData::StokesPreconditioner<dim>&       data)
    {
        this->assembleStokesPreconLocal(cell, scratch, data);
    };

    auto copier = [this](const Assembly::CopyData::StokesPreconditioner<dim> &data)
    {
        this->copyStokesPreconLocalToGlobal(data);
    };

    WorkStream::run(
        dof_handler_stokes.begin_active(),
        dof_handler_stokes.end(),
        worker,
        copier,
        Assembly::Scratch::StokesPreconditioner<dim>(
            this->fe_stokes,
            this->quad_formula,
            update_values | update_JxW_values),
        Assembly::CopyData::StokesPreconditioner<dim>(
            this->fe_stokes
        )
    );

    std::cout << "Compeleted" << std::endl;

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
        this->system_matrix_stokes
    );
}

template<int dim>
void SCHSolver<dim>::assembleStokesMatrix()
{
    std::cout << "Assembling Stokes matrix" << std::endl;

    this->system_matrix_stokes = 0;

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
    
    WorkStream::run(
        dof_handler_stokes.begin_active(),
        dof_handler_stokes.end(),
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

    std::cout << "Completed" << std::endl;
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
    std::cout << "Assembling Stokes right hand side" << std::endl;

    this->rhs_stokes = 0;

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
    
    WorkStream::run(
        dof_handler_stokes.begin_active(),
        dof_handler_stokes.end(),
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

    std::cout << "Completed" << std::endl;
}

template<int dim>
void SCHSolver<dim>::solveStokes()
{

    std::cout << "Solving Stokes system... " << std::endl;

    auto A      = linear_operator(this->system_matrix_stokes.block(0,0));
    auto B_T    = linear_operator(this->system_matrix_stokes.block(0,1));
    auto B      = linear_operator(this->system_matrix_stokes.block(1,0));
    auto Mp     = linear_operator(this->system_matrix_precon.block(1,1));

    std::cout << "Block norms: " << std::endl;
    std::cout << "\t" << this->system_matrix_stokes.block(0,0).frobenius_norm()
              << std::endl;
    std::cout << "\t" << this->system_matrix_stokes.block(0,1).frobenius_norm()
              << std::endl;
    std::cout << "\t" << this->system_matrix_stokes.block(1,0).frobenius_norm()
              << std::endl;
    std::cout << "\t" << this->system_matrix_precon.block(1,1).frobenius_norm()
              << std::endl;

    SolverControl                           solver_control_A(
                                                2000, 
                                                1e-14
                                            );
    SolverCG<Vector<double>>                solver_A(solver_control_A);
    SparseDirectUMFPACK                     preconditioner_A;
    preconditioner_A.initialize(this->system_matrix_stokes.block(0,0));

    SolverControl                           solver_control_Mp(
                                                2000,
                                                std::max(1e-8 * this->rhs_stokes.l2_norm(),
                                                         1e-8)
                                            );
    SolverGMRES<Vector<double>>             solver_Mp(solver_control_Mp);
    SparseILU<double>                       preconditioner_Mp;
    preconditioner_Mp.initialize(this->system_matrix_precon.block(1,1));

    const auto op_A_inv     = inverse_operator(A, solver_A, preconditioner_A);
    const auto op_Mp_inv    = inverse_operator(Mp, solver_Mp, preconditioner_Mp);
    const auto op_S         = B * op_A_inv * B_T;

    std::cout << "\tComputing right hand side" << std::endl;

    Vector<double> schur_rhs = B * op_A_inv * this->rhs_stokes.block(0);

    SolverControl                   schur_solver_control(
                                        2000, 
                                        1e-10);
    SolverGMRES<Vector<double>>     solver_schur(schur_solver_control);

    const auto op_S_inv = inverse_operator(op_S, solver_schur,
                                           op_Mp_inv);

    std::cout << "\tSolving pressure" << std::endl;

    this->solution_stokes.block(1) = op_S_inv * schur_rhs;

    std::cout << "\tFinding velocity field" << std::endl;
    
    this->solution_stokes.block(0) 
        = op_A_inv * (this->rhs_stokes.block(0) 
                        - B_T * this->solution_stokes.block(1));

    this->constraints_stokes.distribute(this->solution_stokes);

    std::cout << "Finished solving Stokes portion" << std::endl;

}

template<int dim>
void SCHSolver<dim>::assembleCahnHilliardMatrixLocal(
    const typename DoFHandler<dim>::active_cell_iterator &cell,
    Assembly::Scratch::CahnHilliardMatrix<dim>           &scratch,
    Assembly::CopyData::CahnHilliardMatrix<dim>          &data
)
{

    data.local_matrix = 0;

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
                    +=  scratch.phi_val[i] * scratch.phi_val[j]
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
        this->system_matrix_ch
    );
}

template<int dim>
void SCHSolver<dim>::assembleCahnHilliardMatrix()
{

    std::cout << "Assembling Cahn-Hilliard matrix" << std::endl;

    this->system_matrix_ch = 0;

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

    std::cout << "Beginning work stream" << std::endl;
    
    WorkStream::run(
        dof_handler_ch.begin_active(),
        dof_handler_ch.end(),
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

    std::cout << "Completed" << std::endl;
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


    const uint dofs_per_cell = scratch.fe_val_ch.get_fe().n_dofs_per_cell();
    const uint n_q_points    = scratch.fe_val_ch.get_quadrature().size();

    FEValuesExtractors::Vector  velocities(0);
    FEValuesExtractors::Scalar  phi(0);

    scratch.fe_val_ch[phi].get_function_values(
        this->solution_old_ch,
        scratch.phi_old_q
    ); 
    scratch.fe_val_ch[phi].get_function_gradients(
        this->solution_old_ch,
        scratch.phi_grad_old_q
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
            data.local_rhs(i)    +=  scratch.fe_val_ch[phi].value(i,q)
                            *   scratch.phi_old_q[q]
                            *   scratch.fe_val_ch.JxW(q);

            // 3 k <\nabla\varphi_i, \nabla\phi_old>
            data.local_rhs(i)    +=  3.0 * this->timestep
                            *   scratch.fe_val_ch[phi].gradient(i, q)
                            *   scratch.phi_grad_old_q[q]
                            *   scratch.fe_val_ch.JxW(q);

            // - k <\nabla\varphi_i, 3(\phi_old)^2 \nabla\phi_old>
            data.local_rhs(i)    -=  this->timestep * (
                                    scratch.fe_val_ch[phi].gradient(i,q)
                                    * 3.0 * pow(scratch.phi_old_q[q],2) 
                                    * scratch.phi_grad_old_q[q]
                                 ) * scratch.fe_val_ch.JxW(q);
            
            // Advection
            data.local_rhs(i)    += this->timestep * (
                                    scratch.fe_val_ch[phi].value(i,q) 
                                    * scratch.vel_old_q[q] 
                                    * scratch.phi_grad_old_q[q]
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
    std::cout << "Assembling Cahn-Hilliard right hand side" << std::endl;

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

    std::cout << "Beginning work stream" << std::endl;
    
    WorkStream::run(
        dof_handler_ch.begin_active(),
        dof_handler_ch.end(),
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

    std::cout << "Completed" << std::endl;
}

template<int dim>
void SCHSolver<dim>::assembleCahnHilliard()
{

    std::cout << "Assembling Cahn-Hilliard system" << std::endl;
    
    this->assembleCahnHilliardMatrix();
    this->assembleCahnHilliardRHS();
    
    // Decomposition of RHS vector
    auto phi_rhs = this->rhs_ch.block(0);
    auto eta_rhs = this->rhs_ch.block(1);
    
    auto phi_range = std::minmax_element(phi_rhs.begin(),
                                         phi_rhs.end());
    auto eta_range = std::minmax_element(eta_rhs.begin(),
                                         eta_rhs.end());

    std::cout   <<    "   Phi RHS range: (" 
                << *phi_range.first << ", "
                << *phi_range.second 
                << ")" << std::endl;
    std::cout   <<    "   Eta RHS range: (" 
                << *eta_range.first << ", "
                << *eta_range.second 
                << ")" << std::endl;
}

template<int dim>
void SCHSolver<dim>::solveCahnHilliard()
{

    std::cout << "Solving Cahn-Hilliard system" << std::endl;

    ReductionControl solverControlInner(2000, 1.0e-18, 1.0e-10);
    SolverCG<Vector<double>>    solverInner(solverControlInner);

    SolverControl               solverControlOuter(
                                    10000,
                                    1e-10 * this->rhs_ch.l2_norm()
                                );
    SolverGMRES<
        Vector<double>
        >    solverOuter(solverControlOuter);
    
    // Decomposition of tangent matrix
    const auto A = linear_operator(this->system_matrix_ch.block(0,0));
    const auto B = linear_operator(this->system_matrix_ch.block(0,1));
    const auto C = linear_operator(this->system_matrix_ch.block(1,0));
    const auto D = linear_operator(this->system_matrix_ch.block(1,1));
   
    // Decomposition of solution vector
    auto phi = this->solution_ch.block(0);
    auto eta = this->solution_ch.block(1);
    
    // Decomposition of RHS vector
    auto phi_rhs = this->rhs_ch.block(0);
    auto eta_rhs = this->rhs_ch.block(1);
    
    SparseDirectUMFPACK precon_A;
    precon_A.initialize(this->system_matrix_ch.block(0,0));

    // Construction of inverse of Schur complement
    const auto A_inv = inverse_operator(A, solverInner, precon_A);
    const auto S = schur_complement(A_inv,B,C,D);
     
    const auto S_inv = inverse_operator(S, solverOuter, 
                                        this->system_matrix_ch.block(1,1));
     
    // Solve reduced block system
    // PackagedOperation that represents the condensed form of g
    auto rhs = condense_schur_rhs(A_inv,C, phi_rhs, eta_rhs);
     
    // Solve for y
    eta = S_inv * rhs;

    std::cout << "\tSolved for eta..." << std::endl;
     
    // Compute x using resolved solution y
    phi = postprocess_schur_solution (A_inv, B, eta, phi_rhs);

    std::cout << "\tSolved for phi..." << std::endl;

    auto eta_range = std::minmax_element(eta.begin(),
                                         eta.end());
    auto phi_range = std::minmax_element(phi.begin(),
                                         phi.end());

    std::cout   <<    "   Phi range: (" 
                << *phi_range.first << ", "
                << *phi_range.second 
                << ")" << std::endl;
    std::cout   <<    "   Eta range: (" 
                << *eta_range.first << ", "
                << *eta_range.second 
                << ")" << std::endl;

    this->solution_ch.block(0) = phi;
    this->solution_ch.block(1) = eta;
    this->constraints_ch.distribute(this->solution_ch);

}

template<int dim>
void SCHSolver<dim>::refineGrid()
{
    std::cout << "Performing refinement..." << std::endl; 

    Vector<float> estimated_error_per_cell(triangulation.n_active_cells());

    FEValuesExtractors::Scalar phi(0);
    ComponentMask phi_mask = this->fe_ch.component_mask(phi);

    KellyErrorEstimator<dim>::estimate(
        this->dof_handler_ch,
        QGauss<dim-1>(degree+1),
        {},
        this->solution_ch,
        estimated_error_per_cell,
        phi_mask, {}, {}, {}, {},
        KellyErrorEstimator<dim>::face_diameter_over_twice_max_degree
    );
    
    GridRefinement::refine_and_coarsen_optimize(
        this->triangulation,
        estimated_error_per_cell
    );
    
    KellyErrorEstimator<dim>::estimate(
        this->dof_handler_stokes,
        QGauss<dim-1>(degree+1),
        {},
        this->solution_stokes,
        estimated_error_per_cell,
        {}, {}, {}, {}, {},
        KellyErrorEstimator<dim>::face_diameter_over_twice_max_degree
    );
    
    GridRefinement::refine_and_coarsen_optimize(
        this->triangulation,
        estimated_error_per_cell
    );

    std::cout << "\tEstimated errors" << std::endl;

    // Ensure that we do not refine above or below the min and max refinement
    for(auto & cell : triangulation.active_cell_iterators_on_level(min_grid_level))
    {
        cell->clear_coarsen_flag();
    }
    if (triangulation.n_levels() > max_grid_level)
    {
        for(auto & cell : triangulation.active_cell_iterators_on_level(max_grid_level))
        {
            cell->clear_refine_flag();
        }
    }

    std::cout   << "\tSet refine and coarsen flags\n"
                << "\tPreparing solution transfer"
                << std::endl;

    SolutionTransfer<dim, BlockVector<double>> ch_trans(this->dof_handler_ch);
    SolutionTransfer<dim, BlockVector<double>> ch_trans_old(this->dof_handler_ch);

    BlockVector<double> pre_refine_sol_ch;
    BlockVector<double> pre_refine_sol_old_ch;
    pre_refine_sol_ch       = this->solution_ch;
    pre_refine_sol_old_ch   = this->solution_old_ch;

    triangulation.prepare_coarsening_and_refinement();
    ch_trans.prepare_for_coarsening_and_refinement(pre_refine_sol_ch);
    ch_trans_old.prepare_for_coarsening_and_refinement(pre_refine_sol_old_ch);

    triangulation.execute_coarsening_and_refinement();

    std::cout << "\tExecuted coarsening and refinement" << std::endl;

    this->setupDoFs();
    this->setupLinearSystems();

    ch_trans.interpolate(pre_refine_sol_ch, this->solution_ch);
    ch_trans_old.interpolate(pre_refine_sol_old_ch, this->solution_old_ch);

    this->constraints_ch.distribute(this->solution_ch);
    this->constraints_ch.distribute(this->solution_old_ch);

    std::cout << "Completed." << std::endl;
    
}

template<int dim>
void SCHSolver<dim>::outputSurfaceTension()
{
    DataOut<dim>    data_out;
    data_out.attach_dof_handler(this->dof_handler_stokes);
    data_out.add_data_vector(this->rhs_stokes, "surface_tension");
    data_out.build_patches();

    DataOutBase::VtkFlags vtk_flags;
    vtk_flags.compression_level = DataOutBase::VtkFlags::ZlibCompressionLevel::best_speed;
    data_out.set_flags(vtk_flags);
    std::ofstream output("surface_tension.vtu");
    data_out.write_vtu(output);
}

template<int dim>
void SCHSolver<dim>::outputStokes(const uint timestep_number)
{
    std::vector<std::string> solution_names(dim, "velocity");
    solution_names.emplace_back("pressure");

    std::vector<DataComponentInterpretation::DataComponentInterpretation>
        data_component_interpretation(
            dim, DataComponentInterpretation::component_is_part_of_vector);

    data_component_interpretation.push_back(
      DataComponentInterpretation::component_is_scalar);

    DataOut<dim> data_out;
    data_out.attach_dof_handler(this->dof_handler_stokes);
    data_out.add_data_vector(this->solution_stokes,
                             solution_names,
                             DataOut<dim>::type_dof_data,
                             data_component_interpretation);
    data_out.build_patches();

    std::ofstream output(
      "solution-" + Utilities::int_to_string(timestep_number, 2) + ".vtk");
    data_out.write_vtk(output);
}

template<int dim>
void SCHSolver<dim>::outputCahnHilliard(const uint timestep_number)
{
    DataOut<dim>    data_out;
    std::vector<DataComponentInterpretation::DataComponentInterpretation>
        interpretation(2,DataComponentInterpretation::component_is_scalar);
    std::vector<std::string> solution_names = {"phi", "eta"};
    std::vector<std::string> rhs_names = {"phi_rhs", "eta_rhs"};
    std::vector<std::string> old_names = {"phi_old", "eta_old"};

    data_out.add_data_vector(this->dof_handler_ch,
                            this->solution_ch,
                            solution_names,
                            interpretation);
    data_out.add_data_vector(this->dof_handler_ch,
                            this->solution_old_ch,
                            old_names,
                            interpretation);
    data_out.add_data_vector(this->dof_handler_ch,
                            this->rhs_ch,
                            rhs_names,
                            interpretation);
    data_out.build_patches(this->degree+1);

    const std::string filename = ("ch-solution-" 
                                 + std::to_string(timestep_number) 
                                 + ".vtu");

    DataOutBase::VtkFlags vtk_flags;
    vtk_flags.compression_level = DataOutBase
        ::VtkFlags
        ::ZlibCompressionLevel
        ::best_speed;
    data_out.set_flags(vtk_flags);

    std::ofstream output(filename);
    data_out.write_vtu(output);
}

template<int dim>
void SCHSolver<dim>::outputTimestep(const uint timestep_number)
{
    std::vector<std::string> stokes_names(dim, "velocity");
    stokes_names.emplace_back("pressure");

    std::vector<
        DataComponentInterpretation::DataComponentInterpretation
    > stokes_component_interpretation(
        dim + 1, DataComponentInterpretation::component_is_scalar
    );
    for(uint i = 0; i < dim; i++)
        stokes_component_interpretation[i] =
            DataComponentInterpretation::component_is_part_of_vector;

    std::vector<std::string> ch_names = {"phi", "eta"};
    std::vector<
        DataComponentInterpretation::DataComponentInterpretation
    > ch_component_interpretation(
        2, DataComponentInterpretation::component_is_scalar
    );

    DataOut<dim> data_out;
    data_out.add_data_vector(this->dof_handler_stokes,
                             this->solution_stokes,
                             stokes_names,
                             stokes_component_interpretation);
    data_out.add_data_vector(this->dof_handler_ch,
                             this->solution_ch,
                             ch_names,
                             ch_component_interpretation);

    data_out.build_patches(
        std::min(this->fe_stokes.degree, this->fe_ch.degree)
    );

    std::ofstream output("data/solution-" +
                         Utilities::int_to_string(timestep_number, 4)
                         + ".vtk");
    data_out.write_vtk(output);
}

template<int dim>
void SCHSolver<dim>::run(
    std::unordered_map<std::string, double> params,
    double                                  total_sim_time)
{
    this->total_simulation_time = total_sim_time;

    this->setupParameters(params);
    this->setupTriang();
    this->setupDoFs();
    this->setupLinearSystems();
    
    this->initializeValues();
    this->assembleStokesPrecon();
    this->assembleStokesMatrix();
    this->assembleStokesRHS();
    
    if(debug) this->outputSurfaceTension();
    
    this->solveStokes();

    for(uint i = 0; i < 4; i++)
    {

        this->refineGrid();
        this->initializeValues();
        this->assembleStokesPrecon();
        this->assembleStokesMatrix();
        this->assembleStokesRHS();
        
        if(debug) this->outputSurfaceTension();
        
        this->solveStokes();

    }

    this->assembleCahnHilliardMatrix();
   
    for(uint i = 0; i < 1000000; i++)
    {
        this->assembleStokesRHS();
        this->solveStokes();

        this->assembleCahnHilliardRHS();
        this->solveCahnHilliard();
 
        if (i % 100 == 0){
            this->outputTimestep(this->timestep_number);
            
            std::cout   << "Completed timestep number: " 
                        << timestep_number
                        << std::endl;

            timestep_number++;
        }
        
        this->solution_old_ch = this->solution_ch;
    }

}

} // stokesCahnHilliard


int main(){
    try{
        std::cout   << "Running" << std::endl << std::endl;

        std::unordered_map<std::string, double> params;

        params["eps"]       = 1e-2;
        params["gamma"]     = 1;

        double total_sim_time = 10;

        stokesCahnHilliard::SCHSolver<2> stokesCahnHilliard(1, true);
        stokesCahnHilliard.run(params, total_sim_time);

        std::cout << "Finished running." << std::endl;
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
