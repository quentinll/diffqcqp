#include <iostream>
#include "qcqplib/Solver.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

#include <Eigen/Dense>

namespace py = pybind11;

using namespace Eigen;
using namespace std;


VectorXd solveQP( MatrixXd P, VectorXd q,VectorXd warm_start, double epsilon =1e-10, double mu_prox = 1e-7, int max_iter=1000){
    Solver solver;
    VectorXd solution(q.size());
    solution = solver.solveQP(P,q,warm_start,epsilon,mu_prox,max_iter);
    return solution;
}

VectorXd solveDerivativesQP( MatrixXd P, VectorXd q, VectorXd l, VectorXd grad_l, double epsilon =1e-10){
    Solver solver;
    VectorXd gamma(l.size()),bl(l.size());
    gamma = solver.dualFromPrimalQP(P,q,l,epsilon);
    bl = solver.solveDerivativesQP(P,q,l,gamma,grad_l,epsilon);
    return bl;
}

VectorXd solveQCQP(MatrixXd P, VectorXd q, VectorXd l_n, VectorXd mu,VectorXd warm_start, double epsilon=1e-10, double mu_prox = 1e-7, int max_iter = 1000){
    Solver solver;
    VectorXd solution(q.size());
    l_n = l_n.cwiseProduct(mu);
    solution = solver.solveQCQP(P,q,l_n,warm_start,epsilon,mu_prox,max_iter);
    return solution;
}

std::tuple<MatrixXd,MatrixXd,VectorXd> solveDerivativesQCQP( MatrixXd P, VectorXd q, VectorXd l_n, VectorXd mu, VectorXd l, VectorXd grad_l, double epsilon =1e-10){
    Solver solver;
    MatrixXd E1(l_n.size(),l_n.size()),E2(l_n.size(),l_n.size());
    VectorXd mul_n(l_n.size()),gamma(l.size()),blgamma(l.size());
    mul_n = l_n.cwiseProduct(mu);
    gamma = solver.dualFromPrimalQCQP(P,q,mul_n,l,epsilon);
    //std::cout << " gamma : " << gamma << std::endl;
    std::tie(E1,E2) = solver.makeE12QCQP(l_n, mu, gamma);
    blgamma = solver.solveDerivativesQCQP(P,q,mul_n,l,gamma,grad_l,epsilon);
    return std::make_tuple(E1,E2,blgamma);
}


PYBIND11_MODULE(pybindings, m) {
    m.doc() = "module solving QCQP and QP with ADMM, and computing the derivatives of the solution using implicit differentiation of KKT optimality conditions";
    m.def("solveQP", &solveQP, "A function which solves a QP problem with a regularized ADMM algorithm",py::arg("P"), py::arg("q"),py::arg("warm_start"),py::arg("epsilon") = 1e-10,py::arg("mu_prox")= 1e-7,py::arg("max_iter")= 1000 );
    m.def("solveQCQP", &solveQCQP, "A function which solves a QCQP problem with a regularized ADMM algorithm",py::arg("P"), py::arg("q"), py::arg("l_n"),py::arg("mu"), py::arg("warm_start"),py::arg("epsilon")= 1e-10,py::arg("mu_prox")= 1e-7,py::arg("max_iter")= 1000);
    m.def("solveDerivativesQP", &solveDerivativesQP, "A function which solves the differentiated KKT system of a QP",py::arg("P"), py::arg("q"), py::arg("l"), py::arg("grad_l"), py::arg("epsilon")=1e-10 );
    m.def("solveDerivativesQCQP", &solveDerivativesQCQP, "A function which solves the differentiated KKT system of a QCQP",py::arg("P"), py::arg("q"), py::arg("l_n"),py::arg("mu"), py::arg("l"), py::arg("grad_l"), py::arg("epsilon")=1e-10 );
}