#include <Eigen/Dense>

using namespace Eigen;
using namespace std;

class Solver
{
    public:
    Solver();
    double power_iteration(MatrixXd A, double epsilon, int maxStep);
    VectorXd iterative_refinement(MatrixXd A,VectorXd b, double mu_ir, double epsilon, int max_iter);
    VectorXd solveQP(MatrixXd P, VectorXd q, VectorXd warm_start ,double epsilon, double mu_prox, int max_iter);
    VectorXd dualFromPrimalQP(MatrixXd P, VectorXd q, VectorXd l, double epsilon);
    VectorXd solveDerivativesQP(MatrixXd P, VectorXd q, VectorXd l, VectorXd gamma,VectorXd grad_l, double epsilon);
    VectorXd prox_circle(VectorXd l, VectorXd l_n);
    VectorXd solveQCQP(MatrixXd P, VectorXd q, VectorXd l_n,VectorXd warm_start, double epsilon, double mu_prox, int max_iter);
    VectorXd dualFromPrimalQCQP(MatrixXd P, VectorXd q, VectorXd l_n, VectorXd l, double epsilon);
    VectorXd solveDerivativesQCQP(MatrixXd P, VectorXd q, VectorXd l_n, VectorXd l, VectorXd gamma,VectorXd grad_l, double epsilon);
    std::tuple<MatrixXd,MatrixXd> makeE12QCQP(VectorXd l_n, VectorXd mu, VectorXd gamma);
    int test();

    private:
    int prob_id;
};