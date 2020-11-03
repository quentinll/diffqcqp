#include <Eigen/Dense>

using namespace Eigen;

class Solver
{
    public:
    Solver();
    double power_iteration(MatrixXd A, double epsilon, int maxStep);
    VectorXd iterative_refinement(MatrixXd A,VectorXd b);
    VectorXd solveQP(MatrixXd P, VectorXd q, VectorXd warm_start ,double epsilon, double mu_prox, int max_iter);
    VectorXd dualFromPrimalQP(MatrixXd P, VectorXd q, VectorXd l, double epsilon);
    VectorXd solveDerivativesQP(MatrixXd P, VectorXd q, VectorXd l, VectorXd gamma, double epsilon);
    VectorXd prox_circle(VectorXd l, VectorXd l_n);
    VectorXd solveQCQP(MatrixXd P, VectorXd q, VectorXd l_n,VectorXd warm_start, double epsilon, double mu_prox, int max_iter);
    VectorXd dualFromPrimalQCQP(MatrixXd P, VectorXd q, VectorXd l_n, VectorXd l, double epsilon);
    VectorXd solveDerivativesQCQP(MatrixXd P, VectorXd q, VectorXd l, VectorXd gamma, double epsilon);
    int test();

    private:
    int prob_id;
};