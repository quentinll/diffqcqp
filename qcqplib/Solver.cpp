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

VectorXd Solver::iterative_refinement(MatrixXd A, VectorXd b, double mu_ir = 1e-7, double epsilon = 1e-10, int max_iter = 10){
    VectorXd Ab(A.cols()), delta(A.cols());
    MatrixXd AA_tild(A.cols(), A.cols()),AA_tild_inv(A.cols(), A.cols());
    VectorXd x = VectorXd::Zero(A.cols());
    Ab = A.transpose()*b;
    //MatrixXd A_t = A.transpose();
    AA_tild = A.transpose()*A;
    AA_tild_inv = AA_tild+mu_ir*MatrixXd::Identity(AA_tild.rows(),AA_tild.cols());
    //SelfAdjointEigenSolver<MatrixXd> eigensolver(AA_tild);
    //if (eigensolver.info() != Success) abort();
    //std::cout << " A_t  : " << A_t << std::endl;
    //std::cout << " A  : " << A << std::endl;
    //std::cout << " dd  : " << b << std::endl;
    //cout << "The eigenvalues of AA are:\n" << eigensolver.eigenvalues() << endl;
    AA_tild_inv = AA_tild_inv.inverse();
    int not_improved = 0;
    double res;
    double res_pred = std::numeric_limits<double>::max();
    for(int i = 0; i<max_iter; i++){
        x = AA_tild_inv * (mu_ir*x + Ab);
        delta = AA_tild*x - Ab;
        //std::cout << "IR res: "<< delta.norm() << std::endl;
        res = delta.norm();
        if(res_pred - res < epsilon){
            not_improved++;
        }
        else{
            res_pred = res;
            not_improved = 0;
        }
        if (res<epsilon || not_improved ==2){
            break;
        }
    }
    //std::cout << " b  : " << x << std::endl;
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

VectorXd Solver::solveDerivativesQP(MatrixXd P, VectorXd q, VectorXd l,VectorXd gamma,VectorXd grad_l, double epsilon){
    std::vector<int> not_null;
    for (int i = 0; i<gamma.size(); i++){
        if(gamma(i)>1e-10){
            not_null.push_back(i);
        }
    }
    MatrixXd B = gamma.asDiagonal();
    MatrixXd C = MatrixXd::Identity(l.size(), l.size());
    MatrixXd A_tild(not_null.size(), not_null.size()), B_tild(not_null.size(),l.size()), C_tild(l.size(),not_null.size()), D_tild(l.size(),l.size());
    //need to initialize with zeros
    A_tild = MatrixXd::Zero(not_null.size(), not_null.size());
    for(int i = 0; i< not_null.size();i++){
        A_tild(i,i) = l(not_null[i]);
        B_tild.row(i) = B.row(not_null[i]);
        C_tild.col(i) = C.col(not_null[i]);
    }
    D_tild = P;
    MatrixXd A(l.size()+not_null.size(),l.size()+not_null.size());
    A.topLeftCorner(not_null.size(),not_null.size()) = A_tild;
    A.topRightCorner(not_null.size(),l.size()) = B_tild;
    A.bottomLeftCorner(l.size(),not_null.size()) = C_tild;
    A.bottomRightCorner(l.size(),l.size()) = D_tild;
    A.transposeInPlace();
    VectorXd dd(A.cols());
    for(int i = 0 ; i< dd.size(); i++){
        if(i<not_null.size()){
            dd(i) = 0.;
        }
        else{
            dd(i) = grad_l(i-not_null.size());
        }
    }
    VectorXd b(A.cols());
    b = Solver::iterative_refinement(A,dd);
    VectorXd bl(l.size());
    for(int i = 0; i <l.size();i++){
        bl(i) = b(not_null.size()+i);
    }
    return bl;
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
    VectorXd gamma(l_n.size()),slack(l_n.size()),l_2d(2) ;
    MatrixXd  A = MatrixXd::Zero(l.size(), l_n.size());
    std::vector<int> not_null;
    slack = l_n;
    for(int i = 0; i<l_n.size(); i++){
        //A(2*i,i) = 2*l(2*i);
        A(2*i,i) = 2*l(2*i);
        //A(2*i+1,i) = 2*l(2*i+1);
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

VectorXd Solver::solveDerivativesQCQP(MatrixXd P, VectorXd q, VectorXd l_n, VectorXd l, VectorXd gamma,VectorXd grad_l, double epsilon){
    int nb_contacts = l_n.size();
    VectorXd slack(nb_contacts);
    slack = -l_n.cwiseProduct(l_n);
    double norm_l2d;
    VectorXd l_2d(2);
    MatrixXd C(2*nb_contacts,nb_contacts);
    MatrixXd D_tild = MatrixXd::Zero(2*nb_contacts, 2*nb_contacts);
    for(int i = 0; i<nb_contacts; i++){
        l_2d(0) = l(2*i);
        l_2d(1) = l(2*i+1);
        norm_l2d = l_2d.squaredNorm();
        slack(i) = slack(i) + norm_l2d;
        C(2*i,i) = 2*l(2*i);
        C(2*i+1,i) = 2*l(2*i+1);
        D_tild(2*i,2*i) = 2*gamma(i);
        D_tild(2*i+1,2*i+1) = 2*gamma(i);
    }
    //std::cout << " l : " << l<< " gamma : " << gamma << std::endl;
    std::vector<int> not_null;
    for (int i = 0; i<nb_contacts; i++){
        if(slack(i)>-1e-10){
            not_null.push_back(i);
        }
    }
    MatrixXd A_tild = MatrixXd::Zero(not_null.size(),not_null.size());
    MatrixXd B = gamma.asDiagonal()*(C.transpose());
    MatrixXd B_tild(not_null.size(),B.cols()), C_tild(C.rows(), not_null.size());
    for(int i = 0; i< not_null.size(); i++){
        A_tild(i,i) = slack(not_null[i]);
        B_tild.row(i) = B.row(not_null[i]);
        C_tild.col(i) = C.col(not_null[i]);
    }
    D_tild = D_tild+P;
    MatrixXd A(l.size()+not_null.size(),l.size()+not_null.size());
    A.topLeftCorner(not_null.size(),not_null.size()) = A_tild;
    A.topRightCorner(not_null.size(),l.size()) = B_tild;
    A.bottomLeftCorner(l.size(),not_null.size()) = C_tild;
    A.bottomRightCorner(l.size(),l.size()) = D_tild;
    //std::cout << " not null size  : " << not_null.size() << "l size: "<< l.size()<< "A size: "<< A.cols()<< std::endl;
    //std::cout << " A  : " << A << std::endl;
    A.transposeInPlace();
    
    VectorXd dd(A.cols());
    for(int i = 0 ; i< dd.size(); i++){
        if(i<not_null.size()){
            dd(i) = 0.;
        }
        else{
            dd(i) = grad_l(i-not_null.size());
        }
    }
    VectorXd b(A.cols());
    b = Solver::iterative_refinement(A,dd);
    VectorXd blgamma = VectorXd::Zero(gamma.size()+l.size());
    for(int i = 0; i<b.size();i++){
        if(i<not_null.size()){
            blgamma(not_null[i]) = b(i);
        }
        else{
            blgamma(nb_contacts-not_null.size()+i) = b(i);
        }
    }
    return blgamma;
}

std::tuple<MatrixXd,MatrixXd> Solver::makeE12QCQP(VectorXd l_n, VectorXd mu, VectorXd gamma){
    MatrixXd E1 = MatrixXd::Zero(l_n.size(),l_n.size());
    MatrixXd E2 = MatrixXd::Zero(l_n.size(),l_n.size());
    for (int i = 0;i<l_n.size();i++){
        E1(i,i) = 2*gamma(i)*l_n(i)*l_n(i)*mu(i);
        E2(i,i) = 2*gamma(i)*l_n(i)*mu(i)*mu(i);
    }
    return std::make_tuple(E1,E2);
}

int Solver::test(){
    
    MatrixXd m2(4,4);
    m2 << 4.45434,  1.11359, -2.22717,  1.11359,
         1.11359,  4.45434,  1.11359, -2.22717,
        -2.22717,  1.11359,  4.45434,  1.11359,
         1.11359, -2.22717,  1.11359,  4.45434;
    VectorXd sol(4),sol2(4),q2(4),l_n(2), warm_start2(4);
    l_n(0) = 1;
    l_n(1) = 1;
    q2 << -0.0112815,-0.0083385,-0.0083385,-0.0112815;
    //q2 << -0.00981,-0.00981,-0.00981,-0.00981;
    warm_start2 << 0.00220234,0.00220234,0.00220234,0.00220234;

    sol = Solver::solveQCQP(m2,q2,l_n,warm_start2);
    std::cout << " solution  : " << sol << std::endl;
    VectorXd gamma(sol.size());
    gamma = Solver::dualFromPrimalQCQP(m2,q2,l_n,sol);
    MatrixXd  A(sol.size(), l_n.size());
    for(int i = 0; i<l_n.size(); i++){
        A(2*i,i) = 2*sol(2*i);
        A(2*i+1,i) = 2*sol(2*i+1);
    }
    std::cout<< "KKT : " << m2*sol + q2 + A*gamma << std::endl;
    std::cout << "solution : " << sol << std::endl;
    std::cout << "solution dual : " << gamma << std::endl;
    VectorXd grad_l(sol.size());
    VectorXd bl(sol.size());
    bl = Solver::solveDerivativesQCQP(m2,q2,l_n,sol,gamma,grad_l,1e-8);
    std::cout << " bl : " << bl << std::endl;

    sol2 = Solver::solveQP(m2,q2,warm_start2);
    VectorXd gamma2(sol2.size());
    gamma2 = Solver::dualFromPrimalQP(m2,q2,sol2);
    std::cout<< "KKT 2: " << m2*sol2 + q2 + gamma2 << std::endl;
    std::cout << "solution 2: " << sol2 << std::endl;
    std::cout << "solution dual 2 : " << gamma2 << std::endl;
    VectorXd grad_l2(sol2.size());
    VectorXd bl2(sol2.size());
    bl2 = Solver::solveDerivativesQP(m2,q2,sol2,gamma2,grad_l2,1e-8);
    std::cout << " bl2 : " << bl2 << std::endl;

    MatrixXd G(12,12);
    G <<  6.6174e-24,  0.0000e+00,  0.0000e+00,  0.0000e+00, -4.8452e-04, 0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00, 0.0000e+00,  0.0000e+00,
         0.0000e+00, -6.6174e-24,  0.0000e+00,  0.0000e+00,  0.0000e+00, 0.0000e+00, -3.9642e-04,  0.0000e+00,  0.0000e+00,  0.0000e+00, 0.0000e+00,  0.0000e+00,
         0.0000e+00,  0.0000e+00, -6.6174e-24,  0.0000e+00,  0.0000e+00, 0.0000e+00,  0.0000e+00,  0.0000e+00, -3.9642e-04, -7.1925e-20, 0.0000e+00,  0.0000e+00,
         0.0000e+00,  0.0000e+00,  0.0000e+00,  6.6174e-24,  0.0000e+00, 0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00, -4.8452e-04,  4.4048e-20,
        -1.0544e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  4.3570e+03, -1.6704e+00,  4.4543e+00,  1.6704e+00,  1.1136e+00,  1.6704e+00, 1.1136e+00, -1.6704e+00,
         0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00, -1.6704e+00, 4.3570e+03, -1.6704e+00,  1.1136e+00,  1.6704e+00,  1.1136e+00, 1.6704e+00,  4.4543e+00,
         0.0000e+00, -1.0544e+00,  0.0000e+00,  0.0000e+00,  4.4543e+00, -1.6704e+00,  5.3243e+03,  1.6704e+00,  1.1136e+00,  1.6704e+00, 1.1136e+00, -1.6704e+00,
         0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  1.6704e+00, 1.1136e+00,  1.6704e+00,  5.3243e+03, -1.6704e+00,  4.4543e+00, -1.6704e+00,  1.1136e+00,
        0.0000e+00,  0.0000e+00, -1.0544e+00,  0.0000e+00,  1.1136e+00, 1.6704e+00,  1.1136e+00, -1.6704e+00,  5.3243e+03, -1.6704e+00,4.4543e+00,  1.6704e+00,
        0.0000e+00,  0.0000e+00, -1.9131e-16,  0.0000e+00,  1.6704e+00,1.1136e+00,  1.6704e+00,  4.4543e+00, -1.6704e+00,  5.3243e+03,-1.6704e+00,  1.1136e+00,
        0.0000e+00,  0.0000e+00,  0.0000e+00, -1.0544e+00,  1.1136e+00,1.6704e+00,  1.1136e+00, -1.6704e+00,  4.4543e+00, -1.6704e+00,4.3570e+03,  1.6704e+00,
        0.0000e+00,  0.0000e+00,  0.0000e+00,  9.5861e-17, -1.6704e+00,4.4543e+00, -1.6704e+00,  1.1136e+00,  1.6704e+00,  1.1136e+00,1.6704e+00,  4.3570e+03;
    G(0,0) = 0; G(1,1) = 4e1;
    VectorXd g(12);
    g << 0.0000e+00,0.0000e+00,0.0000e+00,0.0000e+00,7.2829e-04,2.2609e-14,7.2829e-04,2.2609e-14,7.2829e-04,2.2609e-14,7.2829e-04,2.2609e-14;
    std::cout << " IR : " << Solver::iterative_refinement(G,g,1e-14) << std::endl;

    return 0;
}