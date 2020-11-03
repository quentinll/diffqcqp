#include <iostream>
#include <Eigen/Dense>
#include <chrono>
#include <vector>
#include "Solver.hpp"

using namespace std;
using namespace Eigen;

Solver::Solver(){
    prob_id = 1;
}

VectorXd iterative_refinement(MatrixXd A, VectorXd b, double mu_ir = 1e-7, double epsilon = 1e-10, int max_iter = 100){
    VectorXd x(A.cols()),Ab(A.cols()), delta(A.cols());
    MatrixXd AA_tild(A.cols(), A.cols()),AA_tild_inv(A.cols(), A.cols());
    Ab = A.transpose()*b;
    AA_tild = A.transpose()*A;
    AA_tild_inv = AA_tild+mu_ir*MatrixXd::Identity(AA_tild.rows(),AA_tild.cols());
    AA_tild_inv = AA_tild_inv.inverse();
    for(int i = 0; i<max_iter; i++){
        x = AA_tild_inv * (mu_ir*x + Ab);
        delta = AA_tild*x - Ab;
        if (delta.lpNorm<Infinity>()<epsilon){
            break;
        }
    }
    return x;
}

double Solver::power_iteration(MatrixXd A,  double epsilon = 1e-10, int max_iter = 100){
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

VectorXd Solver::solveQP(MatrixXd P, VectorXd q, VectorXd warm_start, double epsilon =1e-10, double mu_prox = 1e-7, int max_iter=1000){
    //typedef std::chrono::high_resolution_clock Time;
    //typedef std::chrono::duration<float> fsec;
    //auto t0 = Time::now();
    //std::cout << P << std::endl;
    //std::cout << q << std::endl;
    //std::cout << warm_start << std::endl;
    double L, rho, res;
    MatrixXd Pinv(P.rows(), P.cols()),P_prox(P.rows(), P.cols());
    VectorXd q_prox(q.size()),l(q.size());
    VectorXd u = VectorXd::Zero(q.size());
    VectorXd l_2 = VectorXd::Zero(q.size());
    l = warm_start;
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
    //auto t1 = Time::now();
    //fsec fs = t1 - t0;
    //std::cout << fs.count() << "s\n";

    //std::cout << l_2 << std::endl;
    return l_2;
}

VectorXd Solver::dualFromPrimalQP(MatrixXd P, VectorXd q, VectorXd l, double epsilon=1e-10){
    VectorXd gamma(l.size());
    gamma = -(P*l + q);
    for (int i = 0; i<gamma.size();i++){
        if(l(i)>epsilon){
            gamma(i) = 0;
        }
    }
    return gamma;
}

VectorXd Solver::solveDerivativesQP(MatrixXd P, VectorXd q, VectorXd l,VectorXd gamma, double epsilon){
    VectorXd dl(l.size());
    return dl;
}

VectorXd Solver::prox_circle(VectorXd l, VectorXd l_n){
    int nb_contacts;
    double norm_l2d;
    VectorXd l_2d(2);
    nb_contacts = l_n.size();
    for(int i = 0; i<nb_contacts; i++){
        l_2d(0) = l(2*i);
        l_2d(1) = l(2*i+1);
        norm_l2d = l_2d.norm();
        if(norm_l2d> l_n(i)){
            l(2*i) = l_2d(0)*l_n(i)/norm_l2d;
            l(2*i+1) = l_2d(1)*l_n(i)/norm_l2d;
        }
    }
    return l;
}

VectorXd Solver::solveQCQP(MatrixXd P, VectorXd q, VectorXd l_n,VectorXd warm_start, double epsilon=1e-10, double mu_prox = 1e-7, int max_iter = 1000){
    //typedef std::chrono::high_resolution_clock Time;
    //typedef std::chrono::duration<float> fsec;
    //auto t0 = Time::now();

    double L, rho, res;
    MatrixXd Pinv(P.rows(), P.cols()),P_prox(P.rows(), P.cols());
    VectorXd q_prox(q.size()),l(q.size());
    VectorXd u = VectorXd::Zero(q.size());
    VectorXd l_2 = VectorXd::Zero(q.size());
    l = warm_start;
    L = Solver::power_iteration(P,epsilon, 10);
    rho = std::sqrt(mu_prox*L)*std::pow(L/mu_prox,.4);
    q_prox = q;
    P_prox = P+(rho+mu_prox)*MatrixXd::Identity(P.rows(), P.cols());
    Pinv = P_prox.inverse();
    for(int i = 0; i< max_iter; i++){
        l = Pinv*(rho*l_2-u-q_prox);
        q_prox = q - mu_prox*l;
        l_2 = Solver::prox_circle(l+u/rho,l_n);
        u += rho*(l-l_2);
        res = (P*l+q + u).lpNorm<Infinity>();
        if(res < epsilon){
            break;
        }
    };
    //auto t1 = Time::now();
    //fsec fs = t1 - t0;
    //std::cout << fs.count() << "s\n";

    return l_2;
}

VectorXd Solver::dualFromPrimalQCQP(MatrixXd P, VectorXd q, VectorXd l_n, VectorXd l, double epsilon=1e-10){
    VectorXd gamma(l_n.size()), slack(l_n.size()), l_2d(2), A(l.size(), l_n.size());
    std::vector<int> not_null,all;
    slack = l_n;
    for(int i = 0; i<l.size(); i++){
        A(2*i,i) = 2*l(2*i);
        A(2*i+1,i) = 2*l(2*i+1);
    }
    for (int i = 0; i<gamma.size();i++){
        l_2d(0) = l(2*i);
        l_2d(1) = l(2*i+1);
        slack(i) += -l_2d.norm();
        if(slack(i)>epsilon){
            gamma(i) = 0;
        }
        else
        {
            not_null.push_back(i);
        }
        all.push_back(i);
    }
    MatrixXd A_tild(A.rows(), not_null.size());
    for (int i = 0; i < not_null.size(); ++i) {
        A_tild.col(i) = A.col(not_null[i]);
    }
    //A_tild = A(all,not_null);
    A_tild = (A_tild.transpose()*A_tild).inverse()*A_tild.transpose();
    VectorXd gamma_not_null(not_null.size());
    gamma_not_null = -A_tild*(P*l+q);
    int idx;
    for(int i=0; i<not_null.size();i++){
        idx =not_null[i];
        gamma(idx) = gamma_not_null(i);
    }
    return gamma;
}

VectorXd solveDerivativesQCQP(MatrixXd P, VectorXd q, VectorXd l, VectorXd gamma, double epsilon){
    VectorXd dl(l.size());
    return dl;
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
    VectorXd sol(q.size()),l_n(q.size()/2);
    l_n(0) =100;
    sol = Solver::solveQCQP(m,q,l_n,warm_start,epsilon,mu_prox,max_iter);
    std::cout << sol << std::endl;

    MatrixXd m2(4,4);
    m2 << 4.45434,  1.11359, -2.22717,  1.11359,
         1.11359,  4.45434,  1.11359, -2.22717,
        -2.22717,  1.11359,  4.45434,  1.11359,
         1.11359, -2.22717,  1.11359,  4.45434;
    VectorXd sol2(4),q2(4), warm_start2(4);
    q2 << -0.0112815,-0.0083385,-0.0083385,-0.0112815;
    //q2 << -0.00981,-0.00981,-0.00981,-0.00981;
    warm_start2 << 0.00220234,0.00220234,0.00220234,0.00220234;
    sol2 = Solver::solveQP(m2,q2,warm_start2);
    return 0;
}