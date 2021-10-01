#include <Eigen/Dense>

using namespace Eigen;
using namespace std;

class Solver
{
    public:
    Solver();
    double power_iteration(const MatrixXd &A, const double epsilon,const int maxStep);
    VectorXd iterative_refinement(const Ref<const MatrixXd> A,const VectorXd &b, const double mu_ir,const double epsilon,const int max_iter);
    VectorXd solveQP( MatrixXd P, const VectorXd &q, const VectorXd &warm_start , const double epsilon, const double mu_prox, const int max_iter, const bool adaptative_rho);
    VectorXd dualFromPrimalQP(const MatrixXd &P,const VectorXd &q,const VectorXd &l, const double &epsilon);
    VectorXd solveDerivativesQP(const MatrixXd &P, const VectorXd &q, const VectorXd &l, const VectorXd &gamma, const VectorXd &grad_l, const double &epsilon);
    VectorXd solveBoxQP( MatrixXd P, const VectorXd &q, const VectorXd &l_min, const VectorXd &l_max, const VectorXd &warm_start , const double epsilon, const double mu_prox, const int max_iter, const bool adaptative_rho);
    VectorXd dualFromPrimalBoxQP(const MatrixXd &P,const VectorXd &q, const VectorXd &l_min, const VectorXd &l_max,const VectorXd &l, const double &epsilon);
    VectorXd solveDerivativesBoxQP(const MatrixXd &P, const VectorXd &q, const VectorXd &l_min, const VectorXd &l_max, const VectorXd &l, const VectorXd &gamma, const VectorXd &grad_l, const double &epsilon);
    VectorXd solveSignedBoxQP( MatrixXd P, const VectorXd &q, const VectorXd &l_min, const VectorXd &l_max, VectorXd v, const VectorXd &warm_start , const double epsilon, const double mu_prox, const int max_iter, const bool adaptative_rho);
    VectorXd dualFromPrimalSignedBoxQP(const MatrixXd &P,const VectorXd &q, const VectorXd &l_min, const VectorXd &l_max, VectorXd v,const VectorXd &l, const double &epsilon);
    //VectorXd prox_circle(VectorXd l, const VectorXd &l_n);
    void prox_circle(VectorXd &l, const VectorXd &l_n);
    VectorXd solveQCQP( MatrixXd P, const VectorXd &q, const VectorXd &l_n, const VectorXd &warm_start, const double epsilon, const double mu_prox, const int max_iter,const bool adaptative_rho);
    VectorXd dualFromPrimalQCQP(const MatrixXd &P, const VectorXd &q, const VectorXd &l_n, const VectorXd &l, const double &epsilon);
    VectorXd solveDerivativesQCQP(const MatrixXd &P, const VectorXd &q, const VectorXd &l_n, const VectorXd &l, const VectorXd &gamma, const VectorXd &grad_l, const double epsilon);
    std::tuple<MatrixXd,MatrixXd> getE12QCQP(const VectorXd &l_n, const VectorXd &mu, const VectorXd &gamma);
    int test();

    private:
    int prob_id;
};
