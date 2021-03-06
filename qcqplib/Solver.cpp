#include <iostream>
#include <Eigen/Dense>
#include <vector>
#include "Solver.hpp"

#include <chrono>

using namespace std;
using namespace Eigen;

Solver::Solver(){
    prob_id = 1;
}

VectorXd Solver::iterative_refinement(const Ref<const MatrixXd> &A,const VectorXd &b,const double mu_ir = 1e-7,const double epsilon = 1e-10,const int max_iter = 10){ //solves the system Ax=b using iterative refinement
    VectorXd Ab(A.cols()), delta(A.cols());
    MatrixXd AA_tild(A.cols(), A.cols()),AA_tild_inv(A.cols(), A.cols());
    VectorXd x = VectorXd::Zero(A.cols());
    Ab = A.transpose()*b;
    AA_tild = A.transpose()*A;
    AA_tild += mu_ir*MatrixXd::Identity(AA_tild.rows(),AA_tild.cols());
    AA_tild_inv.setIdentity();
    AA_tild.llt().solveInPlace(AA_tild_inv);
    int not_improved = 0;
    double res;
    double res_pred = std::numeric_limits<double>::max();
    VectorXd AA_tild_invAb = AA_tild_inv*Ab;
    for(int i = 0; i<max_iter; i++){
        x = mu_ir*AA_tild_inv*x + AA_tild_invAb;
        delta.noalias() = AA_tild*x - Ab;
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
    return x;
}

double Solver::power_iteration(const MatrixXd &A,const  double epsilon = 1e-10, const int max_iter = 100){//computes the biggest eigenvalue of A
    VectorXd v(A.cols()), Av(A.cols());
    v = VectorXd::Random(A.cols());
    v.normalize();
    for (int i =0; i< max_iter; i++){
        Av.noalias() = A*v;
        v = Av;
        v.normalize();
    };
    double l_max;
    Av.noalias() = A*v;
    l_max = v.dot(Av);
    return l_max;
}

VectorXd Solver::solveQP( MatrixXd P, const VectorXd &q, const VectorXd &warm_start, const double epsilon =1e-10, const double mu_prox = 1e-7, const int max_iter=1000,const bool adaptative_rho=true){
    //solving a QP using ADMM algorithm
    double L, rho, res_dual, res_prim, mu_thresh, tau_inc, tau_dec, alpha_relax;
    mu_thresh = 10.; alpha_relax = 1.5;
    MatrixXd Pinv(P.rows(), P.cols());
    VectorXd q_prox(q.size()),l(q.size()),Plqu(q.size());
    VectorXd u = VectorXd::Zero(q.size());
    VectorXd l_2 = VectorXd::Zero(q.size());
    VectorXd l_2_pred = l_2;
    l = warm_start;
    L = Solver::power_iteration(P,epsilon, 10);
    rho = std::sqrt(mu_prox*L)*std::pow(L/mu_prox,.4);
    tau_inc = std::pow(L/mu_prox,.15); tau_dec = tau_inc;
    q_prox = q;
    P += (rho+mu_prox)* MatrixXd::Identity(P.rows(), P.cols());
    LLT<MatrixXd> chol = P.llt();
    Pinv.setIdentity(); chol.solveInPlace(Pinv);
    int rho_up = 0, cpt = 0;
    for(int i = 0; i< max_iter; i++){
        l.noalias() = Pinv*(rho*l_2-u-q_prox);
        q_prox.noalias() = q - mu_prox*l;
        l_2.noalias() = (alpha_relax*l + (1-alpha_relax)*l_2+u/rho).cwiseMax(0);
        u.noalias() += rho*(alpha_relax*l + (1-alpha_relax)*l_2_pred-l_2);
        Plqu.noalias() = rho*(l_2-l_2_pred);
        res_dual = Plqu.lpNorm<Infinity>();
        res_prim = (l_2-(alpha_relax*l + (1-alpha_relax)*l_2_pred)).lpNorm<Infinity>();
        l_2_pred = l_2;
        if(res_dual < epsilon){
            break;
        }
        if(adaptative_rho){
            if(res_prim > mu_thresh*res_dual){ //rho needs to be increased
                if( cpt% 5 == 0){ //limits the frequency of rho update to 1 every 5 iterations
                    if (rho_up ==-1){
                        tau_inc = 1+.8*(tau_inc-1);
                        tau_dec = 1+.8*(tau_dec-1);
                    }
                    P += rho*(tau_inc-1)*MatrixXd::Identity(P.rows(), P.cols());
                    rho *= tau_inc;
                    chol = P.llt();
                    Pinv.setIdentity(); chol.solveInPlace(Pinv);
                    rho_up= 1;
                }
                cpt++;
            }
            else if (res_dual > mu_thresh*res_prim){//rho needs to be decreased
                if( cpt% 5 == 0){ 
                    if (rho_up ==1){
                        tau_inc = 1+.8*(tau_inc-1);
                        tau_dec = 1+.8*(tau_dec-1);
                    }
                    P += rho*(1./tau_dec-1)*MatrixXd::Identity(P.rows(), P.cols());
                    rho /= tau_dec;
                    chol = P.llt();
                    Pinv.setIdentity(); chol.solveInPlace(Pinv);
                    rho_up=-1;
                }
                cpt++;
            }
        }
    };
    return l_2;
}

VectorXd Solver::dualFromPrimalQP(const MatrixXd &P, const VectorXd &q, const VectorXd &l, const double &epsilon=1e-10){//computes dual solutions from primal solution
    VectorXd gamma(l.size());
    gamma = -(P*l + q);
    for (int i = 0; i<gamma.size();i++){
        if(l(i)>epsilon){
            gamma(i) = 0;
        }
    }
    return gamma;
}

VectorXd Solver::solveDerivativesQP(const MatrixXd &P, const VectorXd &q, const VectorXd &l, const VectorXd &gamma, const VectorXd &grad_l, const double &epsilon){ //solves the system obtained from differentiating the KKT optimality conditions
    std::vector<int> not_null; //contains index of non null coordinates of gamma
    std::vector<int> null_idx; //contains index of null coordinates of gamma
    for (int i = 0; i<gamma.size(); i++){
        if(gamma(i)<-1e-10){
       // if(gamma(i)>1e-10){
            not_null.push_back(i);
        }
        else{
            null_idx.push_back(i);
        }
    }
    MatrixXd B = gamma.asDiagonal();
    MatrixXd C = MatrixXd::Identity(l.size(), l.size());
    //MatrixXd A_tild(not_null.size(), not_null.size()), B_tild(not_null.size(),l.size()), C_tild(l.size(),not_null.size()), D_tild(l.size(),l.size());
    MatrixXd A_tild(not_null.size(), not_null.size()), B_tild(not_null.size(),null_idx.size()), C_tild(null_idx.size(),not_null.size()), D_tild(null_idx.size(),null_idx.size());
    A_tild = MatrixXd::Zero(not_null.size(), not_null.size());
    for(int i = 0; i< not_null.size();i++){
        A_tild(i,i) = l(not_null[i]);
        for(int j = 0; j< null_idx.size();j++){
            B_tild(i,j) = B(not_null[i],null_idx[j]); 
            C_tild(j,i) = C(null_idx[j],not_null[i]);
        }
        //B_tild.row(i) = B.row(not_null[i]);
        //C_tild.col(i) = C.col(not_null[i]);
    }
    for(int i = 0; i< null_idx.size();i++){
        for(int j = 0; j< null_idx.size();j++){
            D_tild(i,j) = P(null_idx[i],null_idx[j]);
        }
    }
    //D_tild = P;
    //MatrixXd A(l.size()+not_null.size(),l.size()+not_null.size());
    MatrixXd A(l.size(),l.size());
    A.topLeftCorner(not_null.size(),not_null.size()) = A_tild;
    //A.topRightCorner(not_null.size(),l.size()) = B_tild;
    A.topRightCorner(not_null.size(),null_idx.size()) = B_tild;
    //A.bottomLeftCorner(l.size(),not_null.size()) = C_tild;
    A.bottomLeftCorner(null_idx.size(),not_null.size()) = C_tild;
    //A.bottomRightCorner(l.size(),l.size()) = D_tild;
    A.bottomRightCorner(null_idx.size(),null_idx.size()) = D_tild;
    A.transposeInPlace();
    VectorXd dd(A.cols());
    for(int i = 0 ; i< dd.size(); i++){
        if(i<not_null.size()){
            dd(i) = 0.;
        }
        else{
            //dd(i) = grad_l(i-not_null.size());
            dd(i) = grad_l(null_idx[i-not_null.size()]);
        }
    }
    VectorXd b(A.cols());
    b = Solver::iterative_refinement(A,dd);
    VectorXd bl = VectorXd::Zero(l.size());
    //for(int i = 0; i <l.size();i++){
    for(int i = 0; i <null_idx.size();i++){
        bl(null_idx[i]) = b(not_null.size() +i );
    }
    return bl;
}

void Solver::prox_circle(VectorXd &l, const VectorXd &l_n){//projection of l on the disk of radius l_n
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
}

VectorXd Solver::solveQCQP( MatrixXd P, const VectorXd &q, const VectorXd &l_n, const VectorXd &warm_start, const double epsilon=1e-10, const double mu_prox = 1e-7, const int max_iter = 1000, const bool adaptative_rho=true){
    double L, rho, res_dual, res_prim, eps_rel, tau_dec, tau_inc, mu_thresh, alpha_relax;
    mu_thresh = 10.; alpha_relax = 1.5;
    eps_rel = 1e-4;
    MatrixXd Pinv(P.rows(), P.cols());
    VectorXd q_prox(q.size()),l(q.size()), Plqu (q.size());
    VectorXd u = VectorXd::Zero(q.size());
    VectorXd l_2 = VectorXd::Zero(q.size());VectorXd l_2_pred = VectorXd::Zero(q.size());
    l = warm_start;
    L = Solver::power_iteration(P,epsilon, 100);
    rho = std::sqrt(mu_prox*L)*std::pow(L/mu_prox,.4);
    tau_dec = std::pow(L/mu_prox,.15); tau_inc = tau_dec;
    q_prox = q;
    P +=(rho+mu_prox)*MatrixXd::Identity(P.rows(), P.cols());
    LLT<MatrixXd> chol = P.llt();
    Pinv.setIdentity(); chol.solveInPlace(Pinv);
    int rho_up = 0, cpt=0;
    for(int i = 0; i< max_iter; i++){
        l = Pinv*(rho*l_2-u-q_prox);
        q_prox = q - mu_prox*l;
        l_2 = alpha_relax*l + (1-alpha_relax)*l_2+u/rho;
        Solver::prox_circle(l_2,l_n);
        u += rho*(alpha_relax*l + (1-alpha_relax)*l_2_pred-l_2);
        Plqu = l_2-l_2_pred;
        res_dual = rho*Plqu.lpNorm<Infinity>();
        res_prim = (l_2-(alpha_relax*l + (1-alpha_relax)*l_2_pred)).lpNorm<Infinity>();
        l_2_pred = l_2;
        if( res_prim < epsilon + eps_rel*l.norm() && res_dual < epsilon ){
            break;
        }
        if (adaptative_rho){
            if( res_prim > mu_thresh*res_dual){//rho needs to be increased
                if(cpt%5 ==0){// limits the frequency of rho update to every 5 iterations
                    if (rho_up ==-1){
                        tau_inc = 1+.8*(tau_inc-1);
                    }
                    P += rho*(tau_inc-1)*MatrixXd::Identity(P.rows(), P.cols());
                    rho *= tau_inc;
                    chol = P.llt();
                    Pinv.setIdentity();chol.solveInPlace(Pinv);
                    rho_up= 1;
                }
                cpt++;
                
            }
            else if ( res_dual > mu_thresh*res_prim){
                if(cpt%5 ==0){
                    if (rho_up ==1){
                        tau_dec = 1+.8*(tau_dec-1);
                    }
                    P +=rho*(1./tau_dec-1)*MatrixXd::Identity(P.rows(), P.cols());
                    rho /= tau_dec;
                    chol = P.llt();
                    Pinv.setIdentity(); chol.solveInPlace(Pinv);
                    rho_up=-1;
                }
                cpt++;
            }
        }
    };
    return l_2;
}

VectorXd Solver::dualFromPrimalQCQP(const MatrixXd &P, const VectorXd &q, const VectorXd &l_n, const VectorXd &l, const double &epsilon=1e-10){
    VectorXd gamma(l_n.size()),slack(l_n.size()),l_2d(2) ;
    MatrixXd  A = MatrixXd::Zero(l.size(), l_n.size());
    std::vector<int> not_null;
    slack = l_n;
    for(int i = 0; i<l_n.size(); i++){
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
    }
    MatrixXd A_tild(A.rows(), not_null.size());
    for (int i = 0; i < not_null.size(); ++i) {
        A_tild.col(i) = A.col(not_null[i]);
    }
    VectorXd gamma_not_null(not_null.size());
    gamma_not_null = -(A_tild.transpose()*A_tild).llt().solve(A_tild.transpose()*(P*l+q));
    int idx;
    for(int i=0; i<not_null.size();i++){
        idx =not_null[i];
        gamma(idx) = gamma_not_null(i);
    }
    return gamma;
}

VectorXd Solver::solveDerivativesQCQP(const MatrixXd &P, const VectorXd &q, const VectorXd &l_n, const VectorXd &l, const VectorXd &gamma, const VectorXd &grad_l, const double &epsilon){
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

std::tuple<MatrixXd,MatrixXd> Solver::getE12QCQP(const VectorXd &l_n, const VectorXd &mu, const VectorXd &gamma){
    MatrixXd E1 = MatrixXd::Zero(l_n.size(),l_n.size());
    MatrixXd E2 = MatrixXd::Zero(l_n.size(),l_n.size());
    for (int i = 0;i<l_n.size();i++){
        E1(i,i) = 2*gamma(i)*l_n(i)*l_n(i)*mu(i);
        E2(i,i) = 2*gamma(i)*l_n(i)*mu(i)*mu(i);
    }
    return std::make_tuple(E1,E2);
}

int Solver::test(){
    typedef std::chrono::high_resolution_clock Time;
    typedef std::chrono::duration<float> fsec;
    MatrixXd m2(4,4);
    m2 << 4.45434,  1.11359, -2.22717,  1.11359,
         1.11359,  4.45434,  1.11359, -2.22717,
        -2.22717,  1.11359,  4.45434,  1.11359,
         1.11359, -2.22717,  1.11359,  4.45434;
    VectorXd sol(4),sol2(4),q2(4),l_n(2), warm_start2(4);
    l_n(0) = 1;
    l_n(1) = 1;
    l_n = l_n*10000;
    q2 << -0.0112815,-0.0083385,-0.0083385,-0.0112815;
    //q2 << -0.00981,-0.00981,-0.00981,-0.00981;
    warm_start2 << 0.00220234,0.00220234,0.00220234,0.00220234;
    m2.setZero();m2(0,0) = .0005;m2(1,1) = 3.;
    q2.setZero(); q2(0) = -8000; q2(1) = 0.;
    warm_start2.setZero();
    //sol = Solver::solveQCQP(m2,q2,l_n,warm_start2, 1e-10,1e-7,1000);
    sol = Solver::solveQP(m2,q2,warm_start2, 1e-10,1e-7,1000);
    std::cout << " solution  : " << sol << std::endl;
    VectorXd gamma(sol.size());
    gamma = Solver::dualFromPrimalQCQP(m2,q2,l_n,sol);
    MatrixXd  A(sol.size(), l_n.size());
    for(int i = 0; i<l_n.size(); i++){
        A(2*i,i) = 2*sol(2*i);
        A(2*i+1,i) = 2*sol(2*i+1);
    }
    //std::cout<< "KKT : " << m2*sol + q2 + A*gamma << std::endl;
    //std::cout << "solution : " << sol << std::endl;
    //std::cout << "solution dual : " << gamma << std::endl;
    VectorXd grad_l(sol.size());
    VectorXd bl(sol.size());
    bl = Solver::solveDerivativesQCQP(m2,q2,l_n,sol,gamma,grad_l,1e-8);
    //std::cout << " bl : " << bl << std::endl;

    sol2 = Solver::solveQP(m2,q2,warm_start2,1e-10,1e-7,1);
    VectorXd gamma2(sol2.size());
    gamma2 = Solver::dualFromPrimalQP(m2,q2,sol2);
    //std::cout<< "KKT 2: " << m2*sol2 + q2 + gamma2 << std::endl;
    //std::cout << "solution 2: " << sol2 << std::endl;
    //std::cout << "solution dual 2 : " << gamma2 << std::endl;
    VectorXd grad_l2(sol2.size());
    VectorXd bl2(sol2.size());
    bl2 = Solver::solveDerivativesQP(m2,q2,sol2,gamma2,grad_l2,1e-8);
    //std::cout << " bl2 : " << bl2 << std::endl;

    
    MatrixXd G2(12,12);
    G2 <<  6.6174e-24,  0.0000e+00,  0.0000e+00,  0.0000e+00, -4.8452e-04, 0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00, 0.0000e+00,  0.0000e+00,
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
    G2(0,0) = 0; G2(1,1) = 4e1;
    G2 = G2*G2.transpose();
    VectorXd g2(12), l_ng(6), l_ng2(6);
    g2 << 0.0000e+00,0.0000e+00,0.0000e+00,0.0000e+00,7.2829e-04,2.2609e-14,7.2829e-04,2.2609e-14,7.2829e-04,2.2609e-14,7.2829e-04,2.2609e-14;
    VectorXd sol3(12);
    //VectorXd warm_start3  = VectorXd::Zero(12);
    VectorXd warm_start3  = VectorXd::Random(12);

    double mean,mean2, mean3;
    mean = 0;mean2 = 0.; mean3 = 0.;
    int ntest = 1;
    std::srand((unsigned int) time(0));
    int test_dimension = 1;
    MatrixXd G(2*test_dimension,2*test_dimension),delt_G(2*test_dimension,2*test_dimension);
    VectorXd g(2*test_dimension),delt_g(2*test_dimension);
    VectorXd grad_l3 = VectorXd::Zero(2*test_dimension);
    grad_l3[1] = 1.; 
    VectorXd gamma3(2*test_dimension);
    for (int i = 0; i< ntest; i++){
        //std::cout<< "prob id : " << i << std::endl;
        g = (VectorXd::Random(2*test_dimension)+VectorXd::Ones(2*test_dimension));
        g = g.array();
        delt_g = VectorXd::Zero(2*test_dimension);
        delt_g[0] = 1e-5;
        //G = g.asDiagonal();
        G = MatrixXd::Random(2*test_dimension,2*test_dimension);
        G = G*G.transpose();
        g = VectorXd::Random(2*test_dimension);
        G(0,0) = 0.4979; G(0,1) = 0.3295; G(1,0) = 0.3295; G(1,1) = 0.2432;
        g[0] =-0.3661;g[1]=-0.9514;
        std::cout<< "G: " << G << "\n";
        std::cout<< "g: " << g << "\n";
        SelfAdjointEigenSolver<MatrixXd> eigensolver(G);
        if (eigensolver.info() != Success) abort();
        //cout << "The eigenvalues of G are:\n" << eigensolver.eigenvalues() << endl;
        l_ng = VectorXd::Random(test_dimension)+VectorXd::Ones(test_dimension);
        l_ng = l_ng*.1;
        l_ng2 = l_ng*100000;
        //std::cout<< "lng: " << l_ng << "\n";
        auto t0 = Time::now();
        //sol3 = Solver::solveQP(G,g,warm_start3,1e-10,1e-7,10000);
        auto t1 = Time::now();
        //sol3 = Solver::solveQCQP(G,g,l_ng,warm_start3,1e-10,1e-7,100000);
        VectorXd sol3(2*test_dimension);
        sol3 = Solver::solveQP(G,g,warm_start3,1e-10,1e-7,100000, true );
        std::cout<< "sol: " << sol3 << "\n";
        VectorXd sol3_bis(2*test_dimension);
        auto t2 = Time::now();
        gamma3 = Solver::dualFromPrimalQP(G,g,sol3,1e-10);
        std::cout<< "gamma: " << gamma3 << "\n";
        std::cout<< "KKT: " << G*sol3+g + gamma3 << "\n";
        VectorXd bl(sol3.size());
        bl = Solver::solveDerivativesQP(G,g,sol3,gamma3,grad_l3,1e-10);
        std::cout<< "grad q " << -bl<< "\n";
        for (int j = 0; j< 2*test_dimension; j++){
            delt_g = VectorXd::Zero(2*test_dimension);
            delt_g[j] = 1e-5;
            sol3_bis = Solver::solveQP(G,g+delt_g,warm_start3,1e-10,1e-7,100000, true );
            std::cout<< "num diff q "<< j << ": " << (sol3_bis-sol3)[1]/1e-5 << "\n";
        }
        std::cout<< "grad P " << -bl*sol3.transpose()<< "\n";
        for (int j = 0; j< 2*test_dimension; j++){
            for (int k = 0; k< 2*test_dimension; k++){
                delt_G = MatrixXd::Zero(2*test_dimension,2*test_dimension);
                delt_G(j,k) = 1e-5;
                sol3_bis = Solver::solveQP(G+delt_G,g,warm_start3,1e-10,1e-7,100000, true );
                std::cout<< "num diff P "<< j << k << ": " << (sol3_bis-sol3)[1]/1e-5 << "\n";
            }
        }
        auto t3 = Time::now();
        sol3 = Solver::solveQCQP(G,g,l_ng2,warm_start3,1e-10,1e-7,100000);
        auto t33 = Time::now();
        gamma3 = Solver::dualFromPrimalQCQP(G,g,l_ng2,sol3,1e-10);
        Solver::solveDerivativesQCQP(G,g,l_ng2,sol3,gamma3,grad_l3,1e-10);
        auto t4 = Time::now();
        fsec fs = t1 - t0;
        fsec fs2 = t3 - t2;
        fsec fs3 = t4 - t33;
        mean += fs.count();
        mean2 += fs2.count();
        mean3 += fs3.count();
        //std::cout << "grad to sol: " << (G*sol3 +g ) << "\n";

    }
    
    //fsec fs = t1 - t0;
    //std::cout<< "solving QP: " << mean/ntest << "s\n";
    std::cout<< "diff QP1: " << mean2/ntest << "s\n";
    std::cout<< "diff QCQP2: " << mean3/ntest << "s\n";

    //std::cout << " IR : " << Solver::iterative_refinement(G,g,1e-14) << std::endl;
    return 0;
}