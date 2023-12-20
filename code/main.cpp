// Deal.II Libraries
#include <algorithm>
#include <boost/iostreams/categories.hpp>
#include <deal.II/base/quadrature.h>
#include <deal.II/base/types.h>
#include <deal.II/base/function.h>

#include <deal.II/fe/fe_values.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/linear_operator.h>
#include <deal.II/lac/schur_complement.h>
#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/fe/fe_update_flags.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

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

#include <deal.II/dofs/dof_renumbering.h>

#include <ostream>
#include <random>
#include <unordered_map>
#include <fstream>

namespace cahnHilliard {
    using namespace dealii;

template<int dim>
class InitialValuesC : public Function<dim>
{
    public:
        InitialValuesC(double eps);
        virtual double value(
            const Point<dim> &p,
            const unsigned int component = 0
        ) const override;

    private:

        mutable std::default_random_engine  generator;
        mutable std::normal_distribution<double>    distribution;

        double eps;
};

template<int dim> 
InitialValuesC<dim>::InitialValuesC(double eps)
    : Function<dim>(2)
    , eps(eps)
{}
    
template<int dim>
double InitialValuesC<dim> :: value(
    const Point<dim> &p,
    const unsigned int component
) const
{
    if(component == 0)
    {
        return std::tanh(
            (p.norm()-0.25) / (std::sqrt(2)*this->eps)
        ); 
    } else {
         return 0.0;
    }
}

template<int dim>
class CahnHilliardEquation
{
public:
    CahnHilliardEquation();
    void run(
        const std::unordered_map<std::string, double> params,
        const double                                  totalSimTime
    );

private:
    void setupSystem(
            const std::unordered_map<std::string, double> params,
            const double                                  totalSimTime
    );
    void setupTriang();
    void setupDoFs();
    void reinitLinearSystem();
    void initializeValues();

    void assembleSystem();
    void updateRHS();
    void solveSystem();

    void outputResults() const;
    
    uint                degree;
    Triangulation<dim>  triangulation;
    FESystem<dim>       fe;
    QGauss<dim>         quad_formula;
    FEValues<dim>       fe_values;
    DoFHandler<dim>     dof_handler;

    AffineConstraints<double>   constraints;

    BlockSparsityPattern        sparsity_pattern;
    BlockSparseMatrix<double>   system_matrix;
    BlockVector<double>         solution;
    BlockVector<double>         solution_old;
    BlockVector<double>         system_rhs;

    double          timestep;
    double          time;
    unsigned int    timestep_number;
    double          totalSimTime;

    double eps;
};

template<int dim> 
CahnHilliardEquation<dim> :: CahnHilliardEquation()
    : degree(1)
    , fe(FE_Q<dim>(degree),2)
    , quad_formula(degree+1)
    , fe_values(this->fe,
                this->quad_formula,
                update_values |
                update_gradients |
                update_JxW_values)
    , dof_handler(triangulation)
    , timestep(1e-4)
    , time(timestep)
    , timestep_number(1)
{}

template<int dim>
void CahnHilliardEquation<dim> :: setupTriang(){

    std::cout << "Building mesh" << std::endl;

    GridGenerator::hyper_cube(
        this->triangulation,
        -1, 1,
        true
    );

    std::cout   << "Connecting nodes to neighbours due to periodic boundary"
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

    std::cout << "Neighbours updated to reflect periodicity" << std::endl;

    std::cout << "Refining grid" << std::endl;
    triangulation.refine_global(8);

    std::cout   << "Mesh generated...\n"
                << "Active cells: " << triangulation.n_active_cells()
                << std::endl;

}

template<int dim>
void CahnHilliardEquation<dim> :: setupDoFs()
{

    std::cout   << "Indexing degrees of freedom..."
                << std::endl;

    this->dof_handler.distribute_dofs(fe);
    DoFRenumbering::component_wise(this->dof_handler);
    
    const std::vector<types::global_dof_index> dofs_per_component =
        DoFTools::count_dofs_per_fe_component(this->dof_handler);
    
    const std::vector<types::global_dof_index> block_sizes = 
        {dofs_per_component[0],
         dofs_per_component[1]};

    std::cout   << "Number of degrees of freedom: "
                << dof_handler.n_dofs()
                << std::endl;
    std::cout   << "Per block:\n"
                << "    Block 0: " << dofs_per_component[0] << std::endl
                << "    Block 1: " << dofs_per_component[1] << std::endl;

    std::cout   << "Adding periodicity considerations to degrees of freedom"
                << " and constraints"
                << std::endl;

    std::vector<GridTools::PeriodicFacePair<
        typename DoFHandler<dim>::cell_iterator>
    > periodicity_vectorX;

    std::vector<GridTools::PeriodicFacePair<
        typename DoFHandler<dim>::cell_iterator>
    > periodicity_vectorY;

    GridTools::collect_periodic_faces(this->dof_handler,
                                      0,1,0,periodicity_vectorX);
    GridTools::collect_periodic_faces(this->dof_handler,
                                      2,3,1,periodicity_vectorY);

    DoFTools::make_periodicity_constraints<dim,dim>(periodicity_vectorX,
                                                    this->constraints);
    DoFTools::make_periodicity_constraints<dim,dim>(periodicity_vectorY,
                                                    this->constraints);

    std::cout   << "Closing constraints" << std::endl;
    this->constraints.close();

    std::cout   << "Building sparsity pattern..."
                << std::endl;

    BlockDynamicSparsityPattern dsp(block_sizes, block_sizes);
    DoFTools::make_sparsity_pattern(
        this->dof_handler,
        dsp,
        this->constraints);

    sparsity_pattern.copy_from(dsp);
    sparsity_pattern.compress();
    
}

template<int dim>
void CahnHilliardEquation<dim> :: setupSystem(
    const std::unordered_map<std::string, double> params,
    const double                                  totalSimTime
)
{

    std::cout << "Passing parameters:" << std::endl;
    for(auto it=params.begin(); it!=params.end(); it++){
        std::cout   << "    "   << it->first
                    << ": "     << it->second
                    << std::endl;
    }

    this->eps = params.at("eps");
    this->totalSimTime = totalSimTime;
    
    this->setupTriang();
    this->setupDoFs();
    this->reinitLinearSystem();
}

template<int dim>
void CahnHilliardEquation<dim> :: reinitLinearSystem()
{
    std::cout   << "Reinitializing the objects for the linear system"
                << std::endl;

    const std::vector<types::global_dof_index> dofs_per_component =
        DoFTools::count_dofs_per_fe_component(this->dof_handler);
    
    const std::vector<types::global_dof_index> block_sizes = 
        {dofs_per_component[0],
         dofs_per_component[1]};

    this->system_matrix.reinit(sparsity_pattern);
    this->solution.reinit(block_sizes);
    this->solution_old.reinit(block_sizes);
    this->system_rhs.reinit(block_sizes);
}

template<int dim>
void CahnHilliardEquation<dim> :: initializeValues()
{   
   
    std::cout   << "Initializing values for phi" << std::endl;

    VectorTools::project(this->dof_handler,
                         this->constraints,
                         QGauss<dim>(fe.degree + 1),
                         InitialValuesC<dim>(this->eps),
                         this->solution_old);

    this->constraints.distribute(this->solution_old);
    
    auto phi_range = std::minmax_element(this->solution_old.block(0).begin(),
                                       this->solution_old.block(0).end());
    auto eta_range = std::minmax_element(this->solution_old.block(1).begin(),
                                       this->solution_old.block(1).end());


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
void CahnHilliardEquation<dim> :: assembleSystem()
{

    std::cout << "Assembling system" << std::endl;

    this->system_matrix = 0;
    this->system_rhs    = 0;

    const unsigned int dofs_per_cell = this->fe.n_dofs_per_cell();

    FullMatrix<double>  local_matrix(
        dofs_per_cell, 
        dofs_per_cell
    );
    Vector<double>      local_rhs(dofs_per_cell);

    std::vector<double>          cell_old_phi_values(this->quad_formula.size());
    std::vector<Tensor<1,dim>>   cell_old_phi_grad(this->quad_formula.size());

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    const FEValuesExtractors::Scalar    phi(0);
    const FEValuesExtractors::Scalar    eta(1);

    for(const auto &cell : dof_handler.active_cell_iterators())
    {

        this->fe_values.reinit(cell);
        local_matrix    = 0;
        local_rhs       = 0;

        cell->get_dof_indices(local_dof_indices);

       this->fe_values[phi].get_function_values(
            this->solution_old,
            cell_old_phi_values
        ); 
       this->fe_values[phi].get_function_gradients(
            this->solution_old,
            cell_old_phi_grad
        ); 

        for(uint q_index = 0 ;  q_index < this->quad_formula.size(); q_index++)
        {   

            double          phi_old_x       = cell_old_phi_values[q_index];
            Tensor<1,dim>   phi_old_x_grad  = cell_old_phi_grad[q_index];

            for(uint i = 0; i < dofs_per_cell; i++)
            {
                
                for(uint j = 0; j < dofs_per_cell; j++)
                {
                    // (0,0): M
                    local_matrix(i,j)
                        +=  this->fe_values[phi].value(i,q_index)
                        *   this->fe_values[phi].value(j,q_index)
                        *   this->fe_values.JxW(q_index);
                    
                    // (0,1): kA
                    local_matrix(i,j)
                        +=  this->timestep 
                        *   this->fe_values[phi].gradient(i,q_index)
                        *   this->fe_values[eta].gradient(j,q_index)
                        *   this->fe_values.JxW(q_index);

                    // (1,0): - (2 M + epsilon^2 A)
                    local_matrix(i,j)
                        -=  2.0 * this->fe_values[eta].value(i,q_index)
                            * this->fe_values[phi].value(j,q_index)
                            * this->fe_values.JxW(q_index);

                    local_matrix(i,j)
                        -=  pow(this->eps,2)
                            * this->fe_values[eta].gradient(i,q_index)
                            * this->fe_values[phi].gradient(j,q_index)
                            * this->fe_values.JxW(q_index); 
 
                    // (1,1): M
                    local_matrix(i,j)
                        +=  this->fe_values[eta].value(i,q_index)
                            * this->fe_values[eta].value(j,q_index)
                            * this->fe_values.JxW(q_index);
                }
                
                // <\varphi_i, phi_old>
                local_rhs(i)    +=  this->fe_values[phi].value(i,q_index)
                                *   phi_old_x
                                *   this->fe_values.JxW(q_index);

                // 3 k <\nabla\varphi_i, \nabla\phi_old>
                local_rhs(i)    +=  3.0 * this->timestep
                                *   this->fe_values[phi].gradient(i,q_index)
                                *   phi_old_x_grad 
                                *   this->fe_values.JxW(q_index);

                // - k <\nabla\varphi_i, 3(\phi_old)^2 \nabla\phi_old>
                local_rhs(i)    -=  this->timestep 
                                *   (this->fe_values[phi].gradient(i,q_index)
                                *   3.0 * pow(phi_old_x,2) * phi_old_x_grad)
                                *   this->fe_values.JxW(q_index);

            }
        }

        this->constraints.distribute_local_to_global(
            local_matrix,
            local_rhs,
            local_dof_indices,
            this->system_matrix,
            this->system_rhs
        );
    }

    std::cout << "Assembly completed" << std::endl;
}

template<int dim>
void CahnHilliardEquation<dim> :: updateRHS()
{

    std::cout << "Updating RHS..." << std::endl;

    const unsigned int dofs_per_cell = this->fe.n_dofs_per_cell();

    this->system_rhs = 0;

    FullMatrix<double>  local_matrix(
        dofs_per_cell, 
        dofs_per_cell
    );
    Vector<double>      local_rhs(dofs_per_cell);

    std::vector<double>          cell_old_phi_values(this->quad_formula.size());
    std::vector<Tensor<1,dim>>   cell_old_phi_grad(this->quad_formula.size());

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    const FEValuesExtractors::Scalar    phi(0);
    const FEValuesExtractors::Scalar    eta(1);

    for(const auto &cell : dof_handler.active_cell_iterators())
    {

        this->fe_values.reinit(cell);
        local_matrix    = 0;
        local_rhs       = 0;

        cell->get_dof_indices(local_dof_indices);

       this->fe_values[phi].get_function_values(
            this->solution_old,
            cell_old_phi_values
        ); 
       this->fe_values[phi].get_function_gradients(
            this->solution_old,
            cell_old_phi_grad
        ); 

        for(uint q_index = 0 ;  q_index < this->quad_formula.size(); q_index++)
        {   

            double          phi_old_x       = cell_old_phi_values[q_index];
            Tensor<1,dim>   phi_old_x_grad  = cell_old_phi_grad[q_index];

            for(uint i = 0; i < dofs_per_cell; i++)
            {
                
                // <\varphi_i, phi_old>
                local_rhs(i)    +=  this->fe_values[phi].value(i,q_index)
                                *   phi_old_x
                                *   this->fe_values.JxW(q_index);

                // 3 k <\nabla\varphi_i, \nabla\phi_old>
                local_rhs(i)    +=  3.0 * this->timestep
                                *   this->fe_values[phi].gradient(i,q_index)
                                *   phi_old_x_grad 
                                *   this->fe_values.JxW(q_index);

                // - k <\nabla\varphi_i, 3(\phi_old)^2 \nabla\phi_old>
                local_rhs(i)    -=  this->timestep 
                                *   (this->fe_values[phi].gradient(i,q_index)
                                *   3.0 * pow(phi_old_x,2) * phi_old_x_grad)
                                *   this->fe_values.JxW(q_index);

            }
        }

        this->constraints.distribute_local_to_global(
            local_rhs,
            local_dof_indices,
            this->system_rhs
        );

    }

    std::cout << "Update completed" << std::endl;
}

template<int dim>
void CahnHilliardEquation<dim> :: solveSystem()
{

    std::cout << "Solving system" << std::endl;

    ReductionControl solverControlInner(2000, 1.0e-18, 1.0e-10);
    SolverCG<Vector<double>>    solverInner(solverControlInner);

    SolverControl               solverControlOuter(
                                    10000,
                                    1e-8 * this->system_rhs.l2_norm()
                                );
    SolverGMRES<
        Vector<double>
        >    solverOuter(solverControlOuter);
    
    // Decomposition of tangent matrix
    const auto A = linear_operator(system_matrix.block(0,0));
    const auto B = linear_operator(system_matrix.block(0,1));
    const auto C = linear_operator(system_matrix.block(1,0));
    const auto D = linear_operator(system_matrix.block(1,1));
   
    // Decomposition of solution vector
    auto phi = solution.block(0);
    auto eta = solution.block(1);
    
    // Decomposition of RHS vector
    auto phi_rhs = system_rhs.block(0);
    auto eta_rhs = system_rhs.block(1);
    
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
    
    SparseILU<double> precon_A;
    precon_A.initialize(system_matrix.block(0,0));

    // Construction of inverse of Schur complement
    const auto A_inv = inverse_operator(A, solverInner, precon_A);
    const auto S = schur_complement(A_inv,B,C,D);
     
    const auto S_inv = inverse_operator(S, solverOuter, 
                                        system_matrix.block(1,1));
     
    // Solve reduced block system
    // PackagedOperation that represents the condensed form of g
    auto rhs = condense_schur_rhs(A_inv,C, phi_rhs, eta_rhs);
     
    // Solve for y
    eta = S_inv * rhs;

    std::cout << "Solved inner problem..." << std::endl;
     
    // Compute x using resolved solution y
    phi = postprocess_schur_solution (A_inv, B, eta, phi_rhs);

    std::cout << "Solved outer problem..." << std::endl;

    eta_range = std::minmax_element(eta.begin(),
                                    eta.end());
    phi_range = std::minmax_element(phi.begin(),
                                    phi.end());

    std::cout   <<    "   Phi range: (" 
                << *phi_range.first << ", "
                << *phi_range.second 
                << ")" << std::endl;
    std::cout   <<    "   Eta range: (" 
                << *eta_range.first << ", "
                << *eta_range.second 
                << ")" << std::endl;

    this->solution.block(0) = phi;
    this->solution.block(1) = eta;
    this->constraints.distribute(this->solution);
    this->solution_old = this->solution;

}

template<int dim>
void CahnHilliardEquation<dim> :: outputResults() const
{

    DataOut<dim> dataOut;

    std::vector<DataComponentInterpretation::DataComponentInterpretation>
        interpretation(2,DataComponentInterpretation::component_is_scalar);
    std::vector<std::string> solution_names = {"phi", "eta"};
    std::vector<std::string> rhs_names = {"phi_rhs", "eta_rhs"};
    std::vector<std::string> old_names = {"phi_old", "eta_old"};

    dataOut.add_data_vector(this->dof_handler,
                            this->solution,
                            solution_names,
                            interpretation);
    dataOut.add_data_vector(this->dof_handler,
                            this->solution_old,
                            old_names,
                            interpretation);
    dataOut.add_data_vector(this->dof_handler,
                            this->system_rhs,
                            rhs_names,
                            interpretation);
    dataOut.build_patches(this->degree+1);

    const std::string filename = ("data/solution-" 
                                 + std::to_string(this->timestep_number) 
                                 + ".vtu");

    DataOutBase::VtkFlags vtk_flags;
    vtk_flags.compression_level = DataOutBase
        ::VtkFlags
        ::ZlibCompressionLevel
        ::best_speed;
    dataOut.set_flags(vtk_flags);

    std::ofstream output(filename);
    dataOut.write_vtu(output);

};

template<int dim> 
void CahnHilliardEquation<dim> :: run(
    const std::unordered_map<std::string, double> params,
    const double                                  totalSimTime
)
{
    this->setupSystem(params, totalSimTime);
    this->initializeValues();

    this->assembleSystem();
    this->solveSystem();
    this->outputResults();
    
    for(uint i = 0; i < 10000; i++)
    {   
        this->timestep_number++;
        this->updateRHS();
        this->solveSystem();
        this->outputResults();
    }
}

}


int main(){ 
    std::cout   << "Running" << std::endl << std::endl;

    std::unordered_map<std::string, double> params;

    params["eps"] = 1e-2;

    double totalSimTime = 10;

    cahnHilliard::CahnHilliardEquation<2> cahnHilliard;
    cahnHilliard.run(params, totalSimTime);

    std::cout << "Completed" << std::endl;

    return 0;
}
