
#include <Eigen/Dense>

using namespace Eigen;

class Solver
{
    public:
    Solver();
    void say_hello();
    void print_matrix();
    double power_iteration(MatrixXd A, double epsilon, int maxStep);
    int test();
    VectorXd solveQP(MatrixXd P, VectorXd q, VectorXd warm_start ,double epsilon, double mu_prox, int max_iter);
    VectorXd solveQCQP(MatrixXd P, VectorXd q, VectorXd l_n,VectorXd warm_start, double epsilon, double mu_prox, int max_iter);

    private:
    int prob_id;
};