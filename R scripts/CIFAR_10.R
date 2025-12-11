################################################################################
##    DL training for the CIFAR-10 dataset with the C++ program Dense_NN.exe
##                                
##
##
## Ad-Oculos-Projekt, https://www.faes.de/ad-oculos/
## GitHub: https://github.com/SuprenumDE/Dense_NN
##
## Günter Faes
## Version 0.0.4, 06.12.2025
## R-Version: 4.5.2  
## OS Windows 11
##
################################################################################
# 
# _________.__                 __________        __   
# /   _____/|  |__   ____      \______   \ _____/  |_ 
# \_____  \ |  |  \_/ __ \      |       _// __ \   __\
#  /        \|   Y  \  ___/     |    |   \  ___/|  |  
# /_______  /|___|  /\___  >    |____|_  /\___  >__|  
#         \/      \/     \/            \/     \/      
#  
#  EIGENnet - Neural networks in C++ with Eigen,   Version 0.1.20, 14.11.2025, License: MIT
#
# Usage:
#   ./Dense_NN [Optionen]
#
# Available options:
#   Train or test a neural network
# Usage:
#   Dense_NN [OPTION...]
#
# -m, --mode arg           Mode: 1=Train, 2=Test, 3=Exit (default: 1)
# -f, --dataset arg        Data set file (e.g., filename.csv)
# -c, --architecture arg   Layer sizes separated by commas, e.g. 784,128,64,10
# -x, --activations arg    Activation functions per layer, separated by 
#                          commas, e.g., relu, relu, tanh, softmax
# -w, --weights arg        Weights file (e.g., weights.txt) (default: weights.txt)
# -e, --epochs arg         Number of epochs (default: 100)
# -z, --loss arg           Loss function: CROSS_ENTROPY, MAE, or MSE (default: CROSS_ENTROPY)
# -s, --samples arg        Number of training samples (default: 1000)
# -b, --batch arg          Minibatch size (32, 64, or 128) (default: 64)
# -v, --val arg            Proportion of data for validation (e.g., 0.2) (default: 0.2) 
# -l, --learning_rate arg  Learning rate (default: 0.01)
# -r, --lr_mode arg        Learning rate mode (e.g., decay, step) (default: "")
# -i, --init arg           Initialization method (HE, XAVIER, ..., ITERATIVE) (default: HE)
# -d, --min_delta arg      Minimal improvement for early training stop (default: 0.0001)
# -o, --optimizer arg      Optimizer: SGD, RMSProp, Adam (default: SGD)
#     --beta1 arg          Adam: first moment decay (beta1) (default: 0.9)
#     --beta2 arg          Adam: second moment decay (beta2) (default: 0.999)
#     --eps arg            Stability term (Adam/RMSProp) (default: 1e-8)
#     --rho arg            RMSProp: Decay rate for squared gradients (default: 0.9)
# -p, --print_config       Outputs the current configuration
# -h, --help               Displays help
#
# Example:
#   ./Dense_NN --architecture=784,128,64,10 --activations=relu,relu,softmax --epochs=100 --lr=0.01 --optimizer=SGD --dataset=mnist.csv
#
# Further information:
#   Guenter Faes, Mail: eigennet@faes.de
# YouTube: https://www.youtube.com/@r-statistik
# GitHub:  https://github.com/SuprenumDE/EigenNET
# ################################################################################

system("./Dense_NN --help")

################################################################################

### Clear all variables:
rm(list = ls())

### Required packages:
library(keras3)
library(grid)
library(gridExtra)
library(ggplot2)     # Head map



# The path to my R working directory (the bin directory containing Dense_NN.exe):
setwd("F:/R-Projekte/Calling_C_NN/bin")

############################ Prepare data ##########################

# Load CIFAR-10 dataset: (see also https://www.cs.toronto.edu/~kriz/cifar.html)
cifar <- dataset_cifar10()

x_train <- cifar$train$x       # Images
y_train <- cifar$train$y       # Labels 0 - 9
x_test  <- cifar$test$x
y_test  <- cifar$test$y

# Only to check dimensions:
dim(x_train)  # 50000 x 32 x 32 x 3; 50,000 images, each 32×32 pixels, with 3 color channels (RGB)
dim(x_test)   # 10000 x 32 x 32 x 3

# CIFAR-10 Class labels:
cifar10_labels <- c(
  "airplane",   # 0
  "automobile", # 1
  "bird",       # 2
  "cat",        # 3
  "deer",       # 4
  "dog",        # 5
  "frog",       # 6
  "horse",      # 7
  "ship",       # 8
  "truck"       # 9
)

# Normalize images (pixel values from 0-255 -> 0-1):
x_train <- x_train / 255
x_test  <- x_test / 255

# Example: 10 color images with plain text titles:
par(mfrow = c(2,5), mar = c(1,1,2,1))
for (i in 1:10) {
  img <- x_train[i,,,]
  plot(as.raster(img))
  title(cifar10_labels[y_train[i] + 1])
}
par(mfrow = c(1,1))


# Example: 10 grayscale images with plain text titles:
gray_images <- list()
par(mfrow=c(2,5), mar=c(1,1,2,1))
for (i in 1:10) {
  img <- x_train[i,,,]
  gray <- 0.299*img[,,1] + 0.587*img[,,2] + 0.114*img[,,3] # Luminance formula
  gray_images[[i]] <- gray  # For later comparison
  plot(as.raster(gray))
  title(cifar10_labels[y_train[i] + 1])
}
par(mfrow = c(1,1))

# Images “flattened” for dense NN (32*32*3 = 3072 features):
x_train <- array_reshape(x_train, c(nrow(x_train), 32*32*3))
x_test  <- array_reshape(x_test,  c(nrow(x_test),  32*32*3))

# Only to check dimensions:
dim(x_train)  # 50000 x 3072
dim(x_test)   # 10000 x 3072

########################################################################
# The following R script section describes the training of a deep
# learning structure WITHOUT dimension reduction. Further down,
# network training with dimension reduction via SVD
# (Singular Value Decomposition) is described.
########################################################################

# Generate data frames for storage as training and test data sets:
# Training:
df_train <- data.frame(y_train, x_train, row.names = NULL)
# Test:
df_test <- data.frame(y_test, x_test, row.names = NULL)

# Export as csv file without header:
# (~ 2.5 GB)
write.table(na.omit(df_train), file = "CIFAR_train.csv", sep = ",", row.names = FALSE, col.names = FALSE)
write.table(na.omit(df_test), file = "CIFAR_test.csv", sep = ",", row.names = FALSE, col.names = FALSE)

######################## Perform training ##############################

### Setting the Dense_NN function parameters:
TrainingDataSet <- "CIFAR_train.csv"
architecture <- "3072,768,192,48,10"
ActivationFunctions <- "relu,relu,relu,softmax"
Optimizer <- "RMSProp"        # or SGD/RMSProp/Adam
Opti_RMSProp_rho = "0.9"      # paste0("--rho=", Opti_RMSProp_rho),
Opti_Adam_b1 = "0.9"          # paste0("--beta1=", Opti_Adam_b1),
Opti_Adam_b2 = "0.999"        # paste0("--beta2=", Opti_Adam_b2),
Loss <- "CROSS_ENTROPY"
epochs <- "150"
n_TrainingSample <- "10000"
BatchSize <- "128"
LearningRate <- "0.001"
LR_dynamic <- "decay"
MinDelta <- "0.001"
WeightInitialization <- "HE"

system2("Dense_NN.exe",
        args = c(
          "--mode=1",
          paste0("--architecture=", architecture), 
          paste0("--activations=", ActivationFunctions), 
          paste0("--optimizer=", Optimizer),
          paste0("--loss=", Loss), 
          paste0("--dataset=", TrainingDataSet), 
          paste0("--epochs=", epochs), 
          paste0("--samples=", n_TrainingSample),
          paste0("--batch=", BatchSize), 
          paste0("--learning_rate=", LearningRate),
          paste0("--lr_mode=", LR_dynamic),
          paste0("--min_delta=", MinDelta), 
          paste0("--init=", WeightInitialization)
        ),
        stdout = "",
        stderr = ""
)

#########################################################################
#
# To graphically display the training parameters and weight matrices,
# run the entire R script “Display_Training_Progress.R.”
#
#########################################################################








#########################################################################
#
#         Reduction of the number of parameters using SVD
#                 (Singular Value Decomposition)
#
#########################################################################

# SVD:
X_train_svd <- x_train        # Copy of the Feature Training Dataset
# Centering, average values per column:
mu <- colMeans(X_train_svd)   
X_train_svd <- sweep(X_train_svd, 2, mu, "-")

x_svd <- svd(X_train_svd)  # X=UDV', nu <= min(n, p) (economical variant of the SVD)
# The SVD will take a few minutes!

# Check dimensions:
dim(x_svd$u)  # 50000 x 3072, Basis in the data space (row vectors, “modes”)
dim(x_svd$v)  # 3072 x 3072, Feature space (column vectors, “eigenimages”)
length(x_svd$d) # 3072, Scaling

# Check SVD (X=UDV', see R documentation for svd):
# D <- diag(x_svd$d) # for X = x_svd$u %*% D %*% t(x_svd$v)
# X_train_svd_reconstructed <- x_svd$u %*% D %*% t(x_svd$v)
# dim(X_train_svd_reconstructed)


#########################################################################
#     Script section for visualization Feature reduction explanation
#########################################################################

# How large must the network training matrix be in order to represent
# reality with acceptable accuracy? The variance across the singular value
# matrix D helps to decide this.

# Variance components via singular values:
sing_vals <- x_svd$d
var_explained <- sing_vals^2 / sum(sing_vals^2)

# Cumulative Variance:
cum_var <- cumsum(var_explained)

# Plot Cumulative Variance:
plot(cum_var, type="l", lwd=2, col="blue",
     main="Cumulative Variance of the SVD Components",
     xlab="Number of Components", ylab="Cumulated Variance")

# Calculate threshold values:
k_90 <- which(cum_var >= 0.90)[1]
k_95 <- which(cum_var >= 0.95)[1]
k_99 <- which(cum_var >= 0.99)[1]

# Helplines and labels:
abline(h=0.9, col="red", lty=2)
abline(v=k_90, col="red", lty=2)
text(k_90, 0.9, paste0("90% at r=", k_90), pos=4, col="red")

abline(h=0.95, col="darkgreen", lty=2)
abline(v=k_95, col="darkgreen", lty=2)
text(k_95, 0.95, paste0("95% at r=", k_95), pos=4, col="darkgreen")

abline(h=0.99, col="purple", lty=2)
abline(v=k_99, col="purple", lty=2)
text(k_99, 0.99, paste0("99% at r=", k_99), pos=4, col="purple")

#
# The reduction factor r is selected based on the graphical analysis. 
#
# Dimension reduction for feature reduction:
r <- 660   # reduction factor
U_r <- x_svd$u[, 1:r]
S_r <- diag(x_svd$d[1:r])
V_r <- x_svd$v[, 1:r]

#
# Dimensions reduction verification:
# 
# Reconstruction of image data (flat structure) with the reduced-size matrices:
X_reduced <- U_r %*% S_r %*% t(V_r)

# What does the reduction in dimensions look like?
# Create sample image:
which_image <- 3    # 1..10!
img_original <- matrix(X_train_svd[3, ], nrow=32, ncol=32)
img_reconstructed <- matrix(X_reduced[3, ], nrow=32, ncol=32)

par(mfrow=c(1,3))
plot(as.raster(gray_images[[which_image]]))
image(img_original[, ncol(img_original):1], col=gray.colors(256), main="Initial Image Data (flat)")
image(img_reconstructed[, ncol(img_reconstructed):1], col=gray.colors(256), main=paste("Reconstruction with r =", r))

par(mfrow=c(1,1))

#########################################################################
#      Create SVD training data set for use in Dense_NN
#########################################################################


# Features:
X_features <- X_train_svd %*% V_r   # Projection: n_samples × r (r: reduction factor)
                                    # Each row of X_train_svd is projected into the reduced feature space.

df_train_svd <- data.frame(y_train, X_features)

# Export as csv file without header:
write.table(na.omit(df_train_svd), file = "CIFAR_train_SVD.csv", sep = ",", row.names = FALSE, col.names = FALSE)


######################## Perform training (SVD) ##############################

### Setting the Dense_NN function parameters:
TrainingDataSet <- "CIFAR_train_SVD.csv"
architecture <- "660,330,165,80,10"
ActivationFunctions <- "relu,relu,relu,softmax"
Optimizer <- "RMSProp"        # or SGD/RMSProp/Adam
Opti_RMSProp_rho = "0.9"      # paste0("--rho=", Opti_RMSProp_rho),
Opti_Adam_b1 = "0.9"          # paste0("--beta1=", Opti_Adam_b1),
Opti_Adam_b2 = "0.999"        # paste0("--beta2=", Opti_Adam_b2),
Loss <- "CROSS_ENTROPY"
epochs <- "200"
n_TrainingSample <- "-1"
BatchSize <- "128"
LearningRate <- "0.001"
LR_dynamic <- "decay"
MinDelta <- "0.001"
WeightInitialization <- "HE"

system2("Dense_NN.exe",
        args = c(
          "--mode=1",
          paste0("--architecture=", architecture), 
          paste0("--activations=", ActivationFunctions), 
          paste0("--optimizer=", Optimizer),
          paste0("--loss=", Loss), 
          paste0("--dataset=", TrainingDataSet), 
          paste0("--epochs=", epochs), 
          paste0("--samples=", n_TrainingSample),
          paste0("--batch=", BatchSize), 
          paste0("--learning_rate=", LearningRate),
          paste0("--lr_mode=", LR_dynamic),
          paste0("--min_delta=", MinDelta), 
          paste0("--init=", WeightInitialization)
        ),
        stdout = "",
        stderr = ""
)

#########################################################################
#
# To graphically display the training parameters and weight matrices,
# run the entire R script “Display_Training_Progress.R.”
#
#########################################################################

######################### Network Test ##################################

### Prepare test data:
# SVD:
x_test_svd <- x_test

# Centering, average values per column (with the mu from the training data):
x_test_svd <- sweep(x_test_svd, 2, mu, "-")


# Features:
X_features <- x_test_svd %*% V_r   # Projection: n_samples × r (r: reduction factor)
# Each row of X_train_svd is projected into the reduced feature space.

df_test_svd <- data.frame(y_test, X_features)

# Export as csv file without header:
write.table(na.omit(df_test_svd), file = "CIFAR_test_SVD.csv", sep = ",", row.names = FALSE, col.names = FALSE)

# Perform Test with Test Data:
system2("Dense_NN.exe",
        args = c(
          "--mode=2", 
          "--dataset=CIFAR_test_SVD.csv"
        ),
        stdout = "",
        stderr = ""
)

#
### Review of test results:
#

forecast <- read.csv("test_results.csv")
View(forecast)

# Overall prediction accuracy (VG):
average_forecast <- mean(forecast$true_label == forecast$predicted_label)
average_forecast


### Presentation as a tabular overview:
# Absolutely:
tabular_forecast <- table(forecast$true_label, forecast$predicted_label)
dimnames(tabular_forecast) <- list(
  True = cifar10_labels,
  Predicted = cifar10_labels
)
tabular_forecast

# Relative (%):
tabular_forecast_relative <- prop.table(tabular_forecast, margin = 1) * 100
round(tabular_forecast_relative, 1)

# Overall view based on 100:
tabular_forecast_total <- prop.table(tabular_forecast) * 100
round(tabular_forecast_total, 2)


### Confusion Matrix:

df_conf <- as.data.frame(as.table(tabular_forecast_relative))

# Plot
ggplot(df_conf, aes(x = Predicted, y = True, fill = Freq)) +
  geom_tile(color = "white") +
  scale_fill_gradient(low = "white", high = "steelblue") +
  geom_text(aes(label = round(Freq, 1)), size = 3) +
  theme_minimal() +
  labs(title = "CIFAR-10 Confusion Matrix (%)",
       x = "Predicted Label",
       y = "True Label")


#### EoS
