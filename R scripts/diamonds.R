################################################################################
##                    Calling the C++ program Dense_NN.exe
##                    Application example: regression model
## 
## Motivation:
## Calling the C/C++ program Dense_NN.exe for training and using an NN and
## return to R.
##
## The diamond dataset is used as an example dataset (Kaggle/OpenML).
## See also dataset Diamonds Description.txt (German/English). The dataset is
## not “clean” and contains heteroscedasticity.
##
## The last section of the script shows a brief analysis of the data set
## using multiple linear regression, which is quite insightful.
##
##
## Ad-Oculos-Projekt, https://www.faes.de/ad-oculos/
## GitHub: https://github.com/SuprenumDE/Dense_NN
##
## Günter Faes
## Version 0.0.04, 05.02.2026
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
#  EIGENnet - Neural networks in C++ with Eigen,   Version 0.1.30, 10.01.2026, License: MIT
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
# Guenter Faes, Mail: eigennet@faes.de
# YouTube: https://www.youtube.com/@r-statistik
# GitHub:  https://github.com/SuprenumDE/EigenNET
# ################################################################################

# For testing reasons:
# The path to bin must be set beforehand, see a few lines below!
system("./Dense_NN --help")

################################################################################

### Clear all variables:
rm(list = ls())

# 
library(rsample)      # to create a training and test data set

# The path to my R working directory (the bin directory containing Dense_NN.exe):
setwd("F:/R-Projekte/Calling_C_NN/bin")


### Reload functions:
source("F:/R-Projekte/Calling_C_NN/scale_features.R")  # scaling functions

################################################################################
#
#             Import, check, and edit data records
#
################################################################################

### Import data set:
diamonds <- read.csv("diamonds.csv")
str(diamonds)

### Check the data records:
View(diamonds)
summary(diamonds)

### The data set contains implausible 0 values for x, y, and z.
#   These are removed:
diamonds_clean <- subset(diamonds, x != 0 & y != 0 & z != 0)
summary(diamonds_clean)

### Overview of the distribution of carat characteristics and prices in order
#   to derive the 'necessary' logarithmic transformation:
par(mfrow = c(2,2))

# Carat:
hist(diamonds_clean$carat, main = "Carat original", col= "lightblue")
hist(log(diamonds_clean$carat), main = "Carat logarithmized", col= "blue")
# Price:
hist(diamonds_clean$price, main = "Price original", col = "lightgreen")
hist(log(diamonds_clean$price), main = "Price logarithmized", col = "green")

par(mfrow = c(1,1))

#-----------------------------------------------------------------------

# Transforming the char format into the factor type:
diamonds_clean$cut <- as.factor(diamonds_clean$cut)
diamonds_clean$color <- as.factor(diamonds_clean$color)
diamonds_clean$clarity <- as.factor(diamonds_clean$clarity)

### Transform factor values into purely numerical values for later storage as a csv file:
# Create codebook:
codebook <- lapply(diamonds_clean, function(x) {
  if (is.factor(x)) {
    data.frame(
      Variable = deparse(substitute(x)),
      Code = seq_along(levels(x)),
      Level = levels(x)
    )
  } else {
    NULL
  }
})
df_codebook <- do.call(rbind, codebook)    # Merge
df_codebook <- subset(df_codebook, select = - Variable)
View(df_codebook)

# Transformation Factor -> Perform num:
diamonds_num <- data.frame(lapply(diamonds_clean, function(x) {
  if (is.factor(x)) as.numeric(x) else x
}))

# Securing min & max Price for rescaling:
Price_min <- min(diamonds_num$price)
Price_max <- max(diamonds_num$price)

### Features scale:
scaling_method <- "minmax"  # Optionen: "minmax", "zscore", "none"
diamonds_scaled <- scale_features(diamonds_num, scaling_method)
View(diamonds_scaled)

# diamonds_scaled$price <- log(diamonds_scaled$price)
# diamonds_scaled$carat <- log(diamonds_scaled$carat)
# diamonds_scaled <- subset(diamonds_scaled, price != -Inf & carat != -Inf)

# Remove redundancies (described by depth and table):
diamonds_scaled <- subset(diamonds_scaled, select = -c(x, y, z))

# Last data check:
View(diamonds_scaled)
str(diamonds_scaled)



# -----  Separating into training and test data -----------

# Reproducibility setting:
set.seed(123)
# Initial split: 80% training, 10% testing:
split_obj <- initial_split(diamonds_scaled, prop = 0.9)
train_data <- training(split_obj)
test_data <- testing(split_obj)

# Separating labels and features:
train_y <- train_data$price
test_y <- test_data$price
train_x <- subset(train_data, select = -price)
test_x <- subset(test_data, select = -price)

# Build train & test data frame:
diamonds_train <- data.frame(price = train_y, train_x)
diamonds_test <- data.frame(price = test_y, test_x)

# -----        Export as CSV file -----

output_file_train <- "diamonds_train.csv"
output_file_test <- "diamonds_test.csv"

# Training dataset:
write.table(na.omit(diamonds_train), file = output_file_train, sep = ",", row.names = FALSE, col.names = FALSE)
# Test dataset:
write.table(na.omit(diamonds_test), file = output_file_test, sep = ",", row.names = FALSE, col.names = FALSE)


######################## Perform training ##############################

### Setting the Dense_NN function parameters:
TrainingDataSet <- output_file_train
architecture <- "6,24,8,1"
ActivationFunctions <- "relu,relu,none"
Optimizer <- "Adam"
Loss <- "MAE"   # Mean Absolute Error
epochs <- "500"
n_TrainingSample <- "-1"
ValidationPortion <- "0.1"
BatchSize <- "32"
LearningRate <- "0.001"
LR_dynamic <- ""
MinDelta <- "0.000001"
WeightInitialization <- "HE"

system2("Dense_NN.exe",
        args = c(
          "--mode=1", 
          paste0("--architecture=", architecture), 
          paste0("--activations=", ActivationFunctions), 
          paste0("--optimizer=", Optimizer),
          paste0("--loss=", Loss), 
          paste0("--dataset=", TrainingDataSet),
          paste0("--val=", ValidationPortion),
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


# cat("Exit-Status:", attr(res, "status"), "\n")
# cat(res, sep = "\n")


#########################################################################
#
# To graphically display the training parameters and weight matrices,
# run the entire R script “Display_Training_Progress.R.”
#
#########################################################################

#########################################################################
#
#                           NN-Test
#
#########################################################################

system2("Dense_NN.exe",
        args = c(
          "--mode=2", 
          "--dataset=diamonds_test.csv"
        ),
        stdout = "",
        stderr = ""
)


#########################################################################
#
#                      Review of test results
#
#########################################################################

prediction <- read.csv("test_results.csv")
View(prediction)

### Rescaling of the sales price:
# Observed values (real values):
rs_sp_real <- prediction$true_label * (Price_max - Price_min) + Price_min
# Model-based price prediction:
rs_sp_pred <- prediction$predicted_label * (Price_max - Price_min) + Price_min

par(mfrow = c(1,2))

plot(rs_sp_real, rs_sp_pred,
     xlab = "Sales Price", ylab = "Predicted Sales Price",
     main = "Prediction vs. Reality", pch = 19, col = "blue")
abline(0, 1, col = "red", lwd = 2)  # Diagonale

residuals <- rs_sp_real - rs_sp_pred
plot(rs_sp_real, residuals,
     xlab = "Sales Price", ylab = "Residuals",
     main = "Residual Analysis", pch = 19, col = "darkgreen")
abline(h = 0, col = "red", lwd = 2)

par(mfrow = c(1,1))


#########################################################################
#
#         Alternative model: multiple linear regression
#
#########################################################################

PriceFunction <- log(price) ~ log(carat) + cut + color + clarity + depth + table
Model <- lm(PriceFunction, diamonds_num)

summary(Model)

### Graphical model diagnosis:

rs_sp_real <- diamonds_num$price
# Logarithmic transformation followed by exponential back-transformation
# often leads to systematic overestimation at high values. Therefore,
# retransformation bias is corrected using the smearing estimator:
smearing_factor <- mean(exp(Model$residuals))
rs_sp_pred <- exp(Model$fitted.values) * smearing_factor

par(mfrow = c(2,1))

plot(rs_sp_real, rs_sp_pred,
     xlab = "Observed price",
     ylab = "Predicted price",
     main = "Observed vs Predicted with LOWESS smoother",
     col = rgb(0, 0, 1, 0.3),
     pch = 16)

# Ideal line:
abline(0, 1, col = "red", lwd = 2)

# LOWESS smoother
lines(lowess(rs_sp_real, rs_sp_pred), col = "#E69F00", lwd = 3)

# Residuals:
plot(Model$residuals, col = "lightgreen", ylab ="Residual model")
abline(h = 0, col = "red", lwd = 2)

par(mfrow = c(1,1))



#################### End of script ###########################################


