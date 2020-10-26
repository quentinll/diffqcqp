#include <iostream>
#include <Eigen/Dense>
#include <chrono>
#include "Solver.hpp"

using namespace std;
using namespace Eigen;

Solver::Solver(){
    prob_id = 1;
}


void Solver::say_hello(){
    std::cout << "Hello, from operations_tests!" << "my id is "<< prob_id<<"\n";
}

void Solver::print_matrix(){
    MatrixXd m(2,2);
    m(0,0) = 3;
    m(1,0) = 2.5;
    m(0,1) = -1;
    m(1,1) = m(1,0) + m(0,1);
    std::cout << m << std::endl;
}

double Solver::power_iteration(MatrixXd A,  double epsilon, int max_iter){
    VectorXd v(A.cols());
    v = VectorXd::Random(A.cols());
    for (int i =0; i< max_iter; i++){
        v = A*v;
        v = v/v.norm();
    };
    double l_max;
    l_max = v.dot(A*v)/v.dot(v);
    return l_max;
}

VectorXd Solver::solveQP(MatrixXd P, VectorXd q, VectorXd warm_start, double epsilon, double mu_prox, int max_iter){
    typedef std::chrono::high_resolution_clock Time;
    typedef std::chrono::duration<float> fsec;
    auto t0 = Time::now();

    double L, rho, res;
    MatrixXd Pinv(P.rows(), P.cols()),P_prox(P.rows(), P.cols());
    VectorXd q_prox(q.size()),l(q.size()),l_2(q.size()), u(q.size());
    L = Solver::power_iteration(P,epsilon, 10);
    rho = std::sqrt(mu_prox*L)*std::pow(L/mu_prox,.4);
    q_prox = q;
    P_prox = P+(rho+mu_prox)*MatrixXd::Identity(P.rows(), P.cols());
    Pinv = P_prox.inverse();
    for(int i = 0; i< max_iter; i++){
        l = Pinv*(rho*l_2-u-q_prox);
        q_prox = q - mu_prox*l;
        l_2 = (l+u/rho).cwiseMax(0);
        u += rho*(l-l_2);
        res = (P*l+q + u).lpNorm<Infinity>();
        if(res < epsilon){
            break;
        }
    };
    auto t1 = Time::now();
    fsec fs = t1 - t0;
    std::cout << fs.count() << "s\n";

    return l_2;
}

VectorXd Solver::solveQCQP(MatrixXd P, VectorXd q, VectorXd l_n,VectorXd warm_start, double epsilon, double mu_prox, int max_iter){
    double L, rho, res;
    MatrixXd Pinv(P.rows(), P.cols()),P_prox(P.rows(), P.cols());
    VectorXd q_prox(q.size()),l(q.size()),l_2(q.size()), u(q.size());
    L = Solver::power_iteration(P,epsilon, 10);
    rho = std::sqrt(mu_prox*L)*std::pow(L/mu_prox,.4);
    q_prox = q;
    P_prox = P+(rho+mu_prox)*MatrixXd::Identity(P.rows(), P.cols());
    Pinv = P_prox.inverse();
    for(int i = 0; i< max_iter; i++){
        l = Pinv*(rho*l_2-u-q_prox);
        q_prox = q - mu_prox*l;
        l_2 = (l+u/rho).cwiseMax(0);
        u += rho*(l-l_2);
        res = (P*l+q + u).lpNorm<Infinity>();
        std::cout <<"res: "<< res << std::endl;
        if(res < epsilon){
            std::cout <<"iter: "<< i << std::endl;
            break;
        }
    };
    return l_2;
}

int Solver::test(){
    MatrixXd m(2,2);
    m(0,0) = 4;
    m(1,0) = 0;
    m(0,1) = 0;
    m(1,1) = 2;
    int max_iter(100);
    double mu_prox(1e-7), epsilon(1e-9);
    VectorXd q(m.cols()), warm_start(m.cols());
    q(0) = -2;
    q(1) = -3;
    VectorXd sol(q.size());
    sol = Solver::solveQP(m,q,warm_start,epsilon,mu_prox,max_iter);
    std::cout << sol << std::endl;
    return 0;
}