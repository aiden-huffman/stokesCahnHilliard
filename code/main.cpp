// Deal.II Libraries
#include <algorithm>
#include <boost/iostreams/categories.hpp>

#include <deal.II/base/data_out_base.h>
#include <deal.II/base/quadrature.h>
#include <deal.II/base/subscriptor.h>
#include <deal.II/base/types.h>
#include <deal.II/base/function.h>

#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>

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
    : Function<dim>(1)
    , eps(eps)
{}

template<int dim>
double InitialValuesPhi<dim>::value(
    const Point<dim> &p,
    const uint       component) const
{
    (void) component;
    return std::tanh(
        (p.norm()-0.25) / (std::sqrt(2) * this->eps)
    );
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

        void assembleCurvature();
        void solveCurvature();
        void outputCurvature();

        void assembleStokes();
        void outputSurfaceTension();
        void solveStokes();
        void outputStokes(const uint timestep_number);

        uint        degree;
        Triangulation<dim>  triangulation;
        FESystem<dim>       fe_stokes;
        FE_Q<dim>           fe_ch;
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

        SparsityPattern         sparsity_pattern_ch;
        SparseMatrix<double>    system_matrix_ch;
        Vector<double>          solution_ch;
        Vector<double>          solution_old_ch;
        Vector<double>          rhs_ch;
        
        SparseMatrix<double>    system_matrix_curve;
        Vector<double>          solution_curve;
        Vector<double>          rhs_curve;

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
, fe_ch(2)
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
    
    std::cout << "\tDistributing..." << std::endl;
    this->dof_handler_ch.distribute_dofs(this->fe_ch);

    std::cout << "\tRenumbering..." << std::endl;
    DoFRenumbering::Cuthill_McKee(this->dof_handler_ch);

    std::cout   << "\tNumber of degrees of freedom: "
                << dof_handler_ch.n_dofs()
                << std::endl;

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
    DynamicSparsityPattern dsp(this->dof_handler_ch.n_dofs(),
                               this->dof_handler_ch.n_dofs());

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

    this->system_matrix_ch.reinit(this->sparsity_pattern_ch);
    this->solution_ch.reinit(this->dof_handler_ch.n_dofs());
    this->solution_old_ch.reinit(this->dof_handler_ch.n_dofs());
    this->rhs_ch.reinit(this->dof_handler_ch.n_dofs());
    
    this->system_matrix_curve.reinit(this->sparsity_pattern_ch);
    this->solution_curve.reinit(this->dof_handler_ch.n_dofs());
    this->rhs_curve.reinit(this->dof_handler_ch.n_dofs());
    
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
        this->solution_old_ch.begin(),
        this->solution_old_ch.end()
    );
    
    std::cout   << "Initial values propagated:\n"
                << "    Phi Range: (" 
                    << *phi_range.first << ", " 
                    << *phi_range.second
                << ")" << std::endl;
}

template<int dim>
void SCHSolver<dim>::assembleCurvature()
{
    std::cout << "Assembling curvature system... ";

    this->system_matrix_curve   = 0;
    this->rhs_curve             = 0;

    FEValues<dim> fe_values_ch(this->fe_ch,
                               this->quad_formula,
                               update_values |
                               update_gradients |
                               update_JxW_values);
    const uint dofs_per_cell = this->fe_ch.n_dofs_per_cell();
    
    FullMatrix<double>  local_matrix(dofs_per_cell,
                                     dofs_per_cell);
    Vector<double>      local_rhs(dofs_per_cell);

    std::vector<Tensor<1,dim>>   cell_grad_ch(this->quad_formula.size());

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    for(const auto &cell : this->dof_handler_ch.active_cell_iterators())
    {
        fe_values_ch.reinit(cell);
        local_matrix    = 0;
        local_rhs       = 0;
        
        cell->get_dof_indices(local_dof_indices);

        fe_values_ch.get_function_gradients(
            this->solution_old_ch,
            cell_grad_ch
        );

        for(uint q_index = 0; q_index < this->quad_formula.size(); q_index++)
        {
            Tensor<1,dim> grad_ch = cell_grad_ch[q_index];

            for(uint i = 0; i < dofs_per_cell; i++)
            {
                for(uint j = 0; j < dofs_per_cell; j++)
                {
                    local_matrix(i,j) += fe_values_ch.shape_value(i,q_index)
                        * fe_values_ch.shape_value(j,q_index)
                        * fe_values_ch.JxW(q_index);
                }

                double norm = grad_ch.norm();
                 
                local_rhs(i) += -fe_values_ch.shape_grad(i,q_index) * grad_ch / (norm + 1e-4);
            }
        }

        this->constraints_ch.distribute_local_to_global(
            local_matrix,
            local_rhs,
            local_dof_indices,
            this->system_matrix_curve,
            this->rhs_curve);
    }
    
    std::cout << "Completed." << std::endl;
}

template<int dim>
void SCHSolver<dim>::assembleStokes()
{
    std::cout << "Constructing Stokes system" << std::endl;

    this->system_matrix_stokes  = 0;
    this->rhs_stokes            = 0;

    FEValues fe_val_stokes(
        this->fe_stokes,
        this->quad_formula,
        update_values |
        update_JxW_values |
        update_gradients
    );
    FEValues fe_val_ch(
        this->fe_ch,
        this->quad_formula,
        update_values |
        update_gradients |
        update_JxW_values
    );

    const uint dofs_per_cell    = this->fe_stokes.n_dofs_per_cell();
    const uint n_q_points       = this->quad_formula.size();

    FullMatrix<double> local_matrix(
        dofs_per_cell,
        dofs_per_cell
    );
    
    FullMatrix<double> local_precon_matrix(
        dofs_per_cell,
        dofs_per_cell
    );

    Vector<double> local_rhs(dofs_per_cell);

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
    
    std::vector<SymmetricTensor<2, dim>> symgrad_phi_u(dofs_per_cell);
    std::vector<double>                  div_phi_u(dofs_per_cell);
    std::vector<Tensor<1, dim>>          phi_u(dofs_per_cell);
    std::vector<double>                  phi_p(dofs_per_cell);
    
    std::vector<double>         kappa_values(n_q_points);
    std::vector<Tensor<1,dim>>  grad_phi_cell(n_q_points);
    
    auto cell       = this->dof_handler_stokes.begin_active();
    auto cell_ch    = this->dof_handler_ch.begin_active();
    const auto endc = this->dof_handler_stokes.end();
    
    const FEValuesExtractors::Vector velocities(0);
    const FEValuesExtractors::Scalar pressure(dim);

    for(; cell != endc; cell++, cell_ch++)
    {
        fe_val_stokes.reinit(cell);
        fe_val_ch.reinit(cell_ch);

        local_precon_matrix = 0;
        local_matrix    = 0;
        local_rhs       = 0;

        cell->get_dof_indices(local_dof_indices);

        fe_val_ch.get_function_gradients(this->solution_old_ch,
                                         grad_phi_cell);
        fe_val_ch.get_function_values(this->solution_curve,
                                      kappa_values);

        for(uint q = 0; q < n_q_points; q++)
        {
            const Tensor<1,dim> grad_phi    = grad_phi_cell[q];
            const double        kappa       = kappa_values[q];
        
            for(uint k = 0; k < dofs_per_cell; k++)
            {
                symgrad_phi_u[k] =
                    fe_val_stokes[velocities].symmetric_gradient(k,q);
                div_phi_u[k]    = fe_val_stokes[velocities].divergence(k,q);
                phi_u[k]        = fe_val_stokes[velocities].value(k,q);
                phi_p[k]        = fe_val_stokes[pressure].value(k,q);
            }

            for(uint i = 0; i < dofs_per_cell; i++)
            {
                for(uint j = 0; j < dofs_per_cell; j++)
                {
                    local_matrix(i,j) +=
                        (2 * (symgrad_phi_u[i] * symgrad_phi_u[j])
                         - div_phi_u[i] * phi_p[j]
                         - phi_p[i] * div_phi_u[j])
                        * fe_val_stokes.JxW(q);

                    local_precon_matrix(i,j) += 
                        (phi_p[i] * phi_p[j]) * fe_val_stokes.JxW(q);
                }

                local_rhs(i) += -24 * std::sqrt(2) * this->eps * 1.0e-7
                    * kappa * grad_phi.norm() * grad_phi
                    * phi_u[i] * fe_val_stokes.JxW(q);
            }
        }

        this->constraints_stokes.distribute_local_to_global(
            local_matrix,
            local_rhs,
            local_dof_indices,
            this->system_matrix_stokes,
            this->rhs_stokes
        );

        this->constraints_stokes.distribute_local_to_global(
            local_precon_matrix,
            local_dof_indices,
            this->system_matrix_precon
        );
    }
}

template<int dim>
void SCHSolver<dim>::solveCurvature()
{
    std::cout << "Solving for curvature" << std::endl;

    SolverControl               solver_control(this->rhs_curve.size(),
                                               1e-12);
    SolverCG<Vector<double>>    cg(solver_control);

    PreconditionSSOR<SparseMatrix<double>> precon;
    precon.initialize(this->system_matrix_curve, 1.2);

    cg.solve(this->system_matrix_curve, this->solution_curve,
             this->rhs_curve, precon);

    std::cout   << "\tSolved curvature for current state of system\n"
                << "\tNumber of CG iterations: " << solver_control.last_step()
                << std::endl;
            
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
void SCHSolver<dim>::refineGrid()
{
    std::cout << "Performing refinement..." << std::endl; 

    Vector<float> estimated_error_per_cell(triangulation.n_active_cells());

    KellyErrorEstimator<dim>::estimate(
        this->dof_handler_ch,
        QGauss<dim-1>(degree+1),
        {},
        this->solution_curve,
        estimated_error_per_cell,
        {}, {}, {}, {}, {},
        KellyErrorEstimator<dim>::face_diameter_over_twice_max_degree
    );
    
    GridRefinement::refine_and_coarsen_optimize(
        this->triangulation,
        estimated_error_per_cell
    );
    
    KellyErrorEstimator<dim>::estimate(
        this->dof_handler_ch,
        QGauss<dim-1>(degree+1),
        {},
        this->solution_ch,
        estimated_error_per_cell,
        {}, {}, {}, {}, {},
        KellyErrorEstimator<dim>::face_diameter_over_twice_max_degree
    );

    GridRefinement::refine_and_coarsen_optimize(
        this->triangulation,
        estimated_error_per_cell
    );

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

    SolutionTransfer<dim, Vector<double>> ch_trans(this->dof_handler_ch);

    std::vector<Vector<double>> pre_refine_sol(3);
    pre_refine_sol[0] = this->solution_curve;
    pre_refine_sol[1] = this->solution_ch;
    pre_refine_sol[2] = this->solution_old_ch;

    triangulation.prepare_coarsening_and_refinement();
    ch_trans.prepare_for_coarsening_and_refinement(pre_refine_sol);

    triangulation.execute_coarsening_and_refinement();

    this->setupDoFs();
    this->setupLinearSystems();

    std::vector<Vector<double>> tmp(3);
    tmp[0].reinit(this->solution_curve);
    tmp[1].reinit(this->solution_ch);
    tmp[2].reinit(this->solution_old_ch);

    ch_trans.interpolate(pre_refine_sol, tmp);

    this->solution_curve    = tmp[0];
    this->solution_ch       = tmp[1];
    this->solution_old_ch   = tmp[2];

    this->constraints_ch.distribute(this->solution_curve);
    this->constraints_ch.distribute(this->solution_ch);
    this->constraints_ch.distribute(this->solution_old_ch);

    std::cout << "Completed." << std::endl;
    
}

template<int dim>
void SCHSolver<dim>::outputCurvature()
{
    DataOut<dim>    data_out;
    data_out.attach_dof_handler(this->dof_handler_ch);
    data_out.add_data_vector(this->solution_curve, "kappa");
    data_out.add_data_vector(this->solution_old_ch, "phi");
    data_out.build_patches();

    DataOutBase::VtkFlags vtk_flags;
    vtk_flags.compression_level = DataOutBase::VtkFlags::ZlibCompressionLevel::best_speed;
    data_out.set_flags(vtk_flags);
    std::ofstream output("curvature.vtu");
    data_out.write_vtu(output);
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
    this->assembleCurvature();
    this->solveCurvature();

    for(uint i = 0; i < 6; i++)
    {
        this->refineGrid();
        this->initializeValues();
        this->assembleCurvature();
        this->solveCurvature();
    

    }
    
    this->initializeValues();
    this->assembleCurvature();
    this->solveCurvature();

    if(debug) this->outputCurvature();

    this->assembleStokes();
    if(debug) this->outputSurfaceTension();

    this->solveStokes();
    this->outputStokes(this->timestep_number);
    timestep_number++;

}

} // stokesCahnHilliard


int main(){ 
    std::cout   << "Running" << std::endl << std::endl;

    std::unordered_map<std::string, double> params;

    params["eps"]       = 1e-2;
    params["gamma"]     = 1;

    double total_sim_time = 10;

    stokesCahnHilliard::SCHSolver<2> stokesCahnHilliard(1, true);
    stokesCahnHilliard.run(params, total_sim_time);

    std::cout << "Finished running." << std::endl;

    return 0;
}
