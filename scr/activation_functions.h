#pragma once
#ifndef ACTIVATION_FUNCTIONS_H
#define ACTIVATION_FUNCTIONS_H

#include "Eigen/Eigen"
#include <functional> // Für std::function
#include <string>

#include <memory>

class ActivationFunction {
public:
    virtual Eigen::VectorXd activate(const Eigen::VectorXd& x) const = 0;
    virtual Eigen::VectorXd derivative(const Eigen::VectorXd& x) const = 0;
    virtual std::string name() const = 0;
    virtual ~ActivationFunction() = default;
};

class ReluActivation : public ActivationFunction {
public:
    Eigen::VectorXd activate(const Eigen::VectorXd& x) const override;
    Eigen::VectorXd derivative(const Eigen::VectorXd& x) const override;
    std::string name() const override { return "ReLU"; }
};

class SigmoidActivation : public ActivationFunction {
public:
    Eigen::VectorXd activate(const Eigen::VectorXd& x) const override;
    Eigen::VectorXd derivative(const Eigen::VectorXd& x) const override;
    std::string name() const override { return "Sigmoid"; }
};

class TanhActivation : public ActivationFunction {
public:
    Eigen::VectorXd activate(const Eigen::VectorXd& x) const override;
    Eigen::VectorXd derivative(const Eigen::VectorXd& x) const override;
    std::string name() const override { return "Tanh"; }
};

class SoftmaxActivation : public ActivationFunction {
public:
    Eigen::VectorXd activate(const Eigen::VectorXd& x) const override;
    Eigen::VectorXd derivative(const Eigen::VectorXd& x) const override;
    std::string name() const override { return "Softmax"; }
};

class IdentityActivation : public ActivationFunction {
public:
    Eigen::VectorXd activate(const Eigen::VectorXd& x) const override;
    Eigen::VectorXd derivative(const Eigen::VectorXd& x) const override;
    std::string name() const override { return "Identity"; }
};

class Activation {
public:
    static std::shared_ptr<ActivationFunction> get(const std::string& name);
};

#endif // ACTIVATION_FUNCTIONS_H

