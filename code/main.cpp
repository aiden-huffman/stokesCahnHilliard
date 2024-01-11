// Deal.II Libraries
#include <algorithm>
#include <boost/iostreams/categories.hpp>

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
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_generator.h>

#include <deal.II/lac/block_sparsity_pattern.h>
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

#include <ostream>
#include <random>
#include <unordered_map>
#include <fstream>

namespace stokesCahnHilliard {
    using namespace dealii;

template<int dim>
class SCHSolver
    {
    public:
        SCHSolver();
        void run();

    private:

        MPI_Comm            mpi_communicator;
        ConditionalOStream  pcout;

        uint                                        degree;
        parallel::distributed::Triangulation<dim>   triangulation;
        FESystem<dim>                               fe_stokes;
        FESystem<dim>                               fe_ch;
        QGauss<dim>                                 quad_formula;

        DoFHandler<dim> dof_handler_stokes;
        DoFHandler<dim> dof_handler_ch;

        AffineConstraints<double>   constraints_stokes;
        AffineConstraints<double>   constraints_pressure;
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

    };

template<int dim>
SCHSolver<dim>::SCHSolver()
: mpi_communicator(MPI_COMM_WORLD)
, pcout(std::cout, (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0))
, degree(1)
, triangulation(mpi_communicator,
                typename Triangulation<dim>::MeshSmoothing(
                Triangulation<dim>::smoothing_on_refinement |
                Triangulation<dim>::smoothing_on_coarsening)
                )
, fe_stokes(FE_Q<dim>(degree+1), dim, FE_DGP<dim>(degree), 1)
, fe_ch(FE_Q<dim>(degree), 2)
, quad_formula(degree+2)
, dof_handler_stokes(this->triangulation)
, dof_handler_ch(this->triangulation)
{}

template<int dim>
void SCHSolver<dim>::setupTriang()
{ 
    this->pcout << "Generating triangulation... " << std::endl;
 
    GridGenerator::hyper_cube(
      triangulation, -1, 1, true);
 
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

    this->pcout << "\tNeighbours updated to reflect periodicity" << std::endl;

    this->pcout << "\tRefining grid" << std::endl;
    triangulation.refine_global(8);

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

    DoFRenumbering::Cuthill_McKee(this->dof_handler_stokes);
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
                this->constraints_stokes
            );
            DoFTools::make_periodicity_constraints<dim,dim>(
                periodicity_vectorY,
                this->constraints_stokes
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
                this->constraints_stokes
            );

            DoFTools::make_periodicity_constraints<dim,dim>(
                periodicity_vectorY,
                this->constraints_stokes
            );

            DoFTools::make_periodicity_constraints<dim,dim>(
                periodicity_vectorZ,
                this->constraints_stokes
            );

        }

        this->constraints_stokes.close();
    }

    reinitStokesMatrix(
        stokes_partitioning,
        stokes_relevant_partitioning
    );

    this->solution_stokes.reinit(
        stokes_partitioning,
        stokes_relevant_partitioning,
        mpi_communicator,
        true
    );

    this->solution_old_stokes.reinit(
        stokes_partitioning,
        stokes_relevant_partitioning,
        mpi_communicator,
        true
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

    Table<2, DoFTools::Coupling> coupling(dim+1, dim+1);
    for(uint i = 0; i < dim+1; i++)
    {
        for(uint j = 0; j < dim+1; j++)
        {
            if(!((i == dim) && (j == dim)))
            {
                coupling[i][j] = DoFTools::always;
            } else {
                coupling[i][j] = DoFTools::none;
            }
        }
    }

    DoFTools::make_sparsity_pattern(
        this->dof_handler_stokes,
        coupling,
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

    Table<2, DoFTools::Coupling> coupling(dim+1, dim+1);
    for(uint i = 0; i < dim+1; i++)
    {
        for(uint j = 0; j < dim+1; j++)
        {
            if(i == j)
            {
                coupling[i][j] = DoFTools::always;
            } else {
                coupling[i][j] = DoFTools::none;
            }
        }
    }

    DoFTools::make_sparsity_pattern(
        this->dof_handler_stokes,
        coupling,
        sp, this->constraints_stokes,
        false,
        Utilities::MPI::this_mpi_process(mpi_communicator)
    );
    }
}

template<int dim>
void SCHSolver<dim>::setupDoFsCahnHilliard()
{

    this->dof_handler_ch.distribute_dofs(this->fe_ch);

    std::vector<uint> ch_sub_blocks = {0,1};

    DoFRenumbering::Cuthill_McKee(this->dof_handler_ch);
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
        ch_partitioning,
        ch_relevant_partitioning,
        mpi_communicator,
        true
    );
    this->solution_old_ch.reinit(
        ch_partitioning,
        ch_relevant_partitioning,
        mpi_communicator,
        true
    );
    this->solution_old_old_ch.reinit(
        ch_partitioning,
        ch_relevant_partitioning,
        mpi_communicator,
        true
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

    this->pcout << "Completed." << std::endl;
}

template<int dim>
void SCHSolver<dim>::run()
{
   
    this->pcout << "Running" << std::endl;
    this->setupTriang();
    this->setupDoFs();

}

} // stokesCahnHilliard

int main(int argc, char *argv[]){
    try{

        dealii::Utilities::MPI::MPI_InitFinalize mpi_initialize(
            argc, argv, 2
        );

        stokesCahnHilliard::SCHSolver<2> stokesCahnHilliard;
        stokesCahnHilliard.run();

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
