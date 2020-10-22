#include <iostream>
#include <Eigen/Dense>
#include "Solver.hpp"

using namespace std;
using Eigen::MatrixXd;

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