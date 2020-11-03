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

VectorXd solveQCQP(MatrixXd P, VectorXd q, VectorXd l_n, VectorXd mu,VectorXd warm_start, double epsilon=1e-10, double mu_prox = 1e-7, int max_iter = 1000){
    Solver solver;
    VectorXd solution(q.size());
    l_n = l_n.cwiseProduct(mu);
    solution = solver.solveQCQP(P,q,l_n,warm_start,epsilon,mu_prox,max_iter);
    return solution;
}


PYBIND11_MODULE(pybindings, m) {
    m.doc() = "module solving QCQP and QP with ADMM, and computing the derivatives of the solution using implicit differentiation of KKT optimality conditions";
    m.def("solveQP", &solveQP, "A function which solves a QP problem with a regularized ADMM algorithm",py::arg("P"), py::arg("q"),py::arg("warm_start"),py::arg("epsilon") = 1e-10,py::arg("mu_prox")= 1e-7,py::arg("max_iter")= 1000 );
    m.def("solveQCQP", &solveQCQP, "A function which solves a QCQP problem with a regularized ADMM algorithm",py::arg("P"), py::arg("q"), py::arg("l_n"),py::arg("mu"), py::arg("warm_start"),py::arg("epsilon")= 1e-10,py::arg("mu_prox")= 1e-7,py::arg("max_iter")= 1000);
}