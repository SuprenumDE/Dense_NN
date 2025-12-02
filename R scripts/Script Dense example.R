################################################################################
##                    Calling the C++ program Dense_NN.exe
##                                 (PoC)
## 
## Motivation:
## Calling the C/C++ program Dense_NN.exe for training and using an NN and
## return to R.
##
##
## Ad-Oculos-Projekt, https://www.faes.de/ad-oculos/
## GitHub: https://github.com/SuprenumDE/Dense_NN
##
## Günter Faes
## Version 0.0.10, 30.11.2025
## R-Version: 4.5.1  
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
# Guenter Faes, Mail: eigennet@faes.de
# YouTube: https://www.youtube.com/@r-statistik
# GitHub:  https://github.com/SuprenumDE/EigenNET
# ################################################################################

system("./Dense_NN --help")

################################################################################

### Clear all variables:
rm(list = ls())

library(rsample)      # to create a training and test data set
library(AmesHousing)  # Dataset ames (Alternative to Boston Housing)

### Functions:
source("F:/R-Projekte/Calling_C_NN/scale_features.R")  # scaling function

# The path to my R working directory (the bin directory containing Dense_NN.exe):
setwd("F:/R-Projekte/Calling_C_NN/bin")


################################################################################
#
#                       Classification Model
#
#
################################################################################

############################ Prepare data ##########################
#
# It is assumed that the training data/test data file is in csv format.
# The challenge is to determine whether the label information is in the first or
# last column in the data set.
# The program Dense_NN.exe assumes that the label column is in the first
# position in the data set.

# ---- Configuration ----
input_file <- "mnist_train_raw.csv"
output_file <- "neu_mnist_train.csv"
label_column <- 1           # Index or name of the label column
scaling_method <- "minmax"  # Options: “minmax,” “zscore,” “none”

# ---- Import data ----
data <- read.csv(input_file)

# ---- Extract label (Y-values) ----
if (is.numeric(label_column)) {
  label <- data[[label_column]]
  features <- data[ , -label_column]
} else {
  label <- data[[label_column]]
  features <- data[ , !(names(data) == label_column)]
}

# Scaling the features (X-values):
scaled_features <- scale_features(features, scaling_method)

# ---- Required data structure: Label + Features: ----
data_ready <- data.frame(label, scaled_features)
data_size <- dim(data_ready) # Only for labeling!


# Export without header:
write.table(na.omit(data_ready), file = output_file, sep = ",", row.names = FALSE, col.names = FALSE)

cat("File successfully exported as", output_file, "\n")

# Diagnosis: readLines(output_file, n = 5)


######################## Perform training ##############################

### Setting the Dense_NN function parameters:
TrainingDataSet <- "neu_mnist_train.csv"
architecture <- "784,128,64,10"
ActivationFunctions <- "relu,relu,softmax"
Optimizer <- "SGD"            # or SGD/RMSProp/Adam
Opti_RMSProp_rho = "0.9"      # paste0("--rho=", Opti_RMSProp_rho),
Opti_Adam_b1 = "0.9"          # paste0("--beta1=", Opti_Adam_b1),
Opti_Adam_b2 = "0.999"        # paste0("--beta2=", Opti_Adam_b2),
Loss <- "CROSS_ENTROPY"
epochs <- "150"
n_TrainingSample <- "-1"
BatchSize <- "64"
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
#                           NN-Test
#
#########################################################################

system2("Dense_NN.exe",
        args = c(
          "--mode=2", 
          "--dataset=neu_mnist_test.csv"
        ),
        stdout = "",
        stderr = ""
)

#########################################################################
#
#                Review of test results
#
#########################################################################

forecast <- read.csv("test_results.csv")
View(forecast)

# Overall prediction accuracy (VG):
average_forecast <- mean(forecast$true_label == forecast$predicted_label)
average_forecast


### Presentation as a tabular overview:
# Absolutely:
tabular_forecast <- table(forecast$true_label, forecast$predicted_label)
tabular_forecast
# Relative (%):
tabular_forecast_relative <- prop.table(tabular_forecast, margin = 1) * 100
round(tabular_forecast_relative, 1)
# Overall view based on 100:
tabular_forecast_total <- prop.table(tabular_forecast) * 100
round(tabular_forecast_total, 2)


################################################################################
#
#                          Regressions Model
#
#
################################################################################

### Clear all variables:
rm(list = ls())

### Reload functions:
source("F:/R Projekte/Calling_C_NN/scale_features.R")  # scaling functions

# ----- Import, check, and edit data records ------

### Import data set:
ames <- make_ames()
str(ames)
View(ames)

# Remove outliers and categorical variables with low information content 
# (see publication Ames Iowa: Alternative to the Boston Housing Data Set):
ames_reduced <- ames[ames$Gr_Liv_Area < 4000,]
ames_reduced <- subset(ames_reduced, select = - Utilities)
ames_reduced <- subset(ames_reduced, select = - Land_Slope)
ames_reduced <- subset(ames_reduced, select = - Land_Contour)
ames_reduced <- subset(ames_reduced, select = - Bldg_Type)
ames_reduced <- subset(ames_reduced, select = - Roof_Style)
ames_reduced <- subset(ames_reduced, select = - Roof_Matl)
ames_reduced <- subset(ames_reduced, select = - Lot_Config)
ames_reduced <- subset(ames_reduced, select = - Bsmt_Full_Bath)
ames_reduced <- subset(ames_reduced, select = - Bsmt_Half_Bath)
ames_reduced <- subset(ames_reduced, select = - Full_Bath)
ames_reduced <- subset(ames_reduced, select = - Half_Bath)
ames_reduced <- subset(ames_reduced, select = - Low_Qual_Fin_SF)
ames_reduced <- subset(ames_reduced, select = - Kitchen_AbvGr)
ames_reduced <- subset(ames_reduced, select = - Fireplaces)
ames_reduced <- subset(ames_reduced, select = - Fireplace_Qu)
ames_reduced <- subset(ames_reduced, select = - Pool_Area)
ames_reduced <- subset(ames_reduced, select = - Pool_QC)

# Secure price range for house:
Price_min <- min(ames$Sale_Price)
Price_max <- max(ames$Sale_Price)

### Transform factor values into purely numerical values for later storage as a csv file:
# Create codebook:
codebook <- lapply(ames_reduced, function(x) {
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
ames_num <- data.frame(lapply(ames_reduced, function(x) {
  if (is.factor(x)) as.numeric(x) else x
}))
View(ames_num)
str(ames_num)

# Features scale:
scaling_method <- "minmax"  # Optionen: "minmax", "zscore", "none"
ames_num <- scale_features(ames_num, scaling_method)
View(ames_num)


# Separating labels and features:
y <- ames_num$Sale_Price
x <- subset(ames_num, select = -Sale_Price)

# Build data frame:
ames_ready <- data.frame(Sale_Price = y, x)

# Convert variable Sale_Price to a real float number, otherwise it will be
# recognized as an integer by train_nn):
ames_ready$Sale_Price <- ames_ready$Sale_Price + runif(nrow(ames_ready), min = 0, max = 0.01)

# Export without header:
# This CSV file is read in below and treated as if it were the analysis output file,
# i.e., a tutorial example!
output_file <- "ames_num.csv"
write.table(ames_ready, output_file, sep = ",", row.names = FALSE)

cat("File successfully exported as: ", output_file, "\n")


# ------------------- Generate training and test data ------------------

### Import data:
ames <- read.csv("ames_num.csv", colClasses = c(Sale_Price = "numeric"))


# ---- Split into training and test data sets ----
split <- initial_split(ames, prop = 0.9)
ames_train <- training(split)
ames_train_size <- dim(ames_train) # Only for labeling!
ames_test <- testing(split)

# ----- Export as CSV file -----
output_file_train <- "ames_train.csv"
output_file_test <- "ames_test.csv"
# training dataset:
write.table(na.omit(ames_train), file = output_file_train, sep = ",", row.names = FALSE, col.names = FALSE)
# test dataset:
write.table(na.omit(ames_test), file = output_file_test, sep = ",", row.names = FALSE, col.names = FALSE)


######################## Perform training ##############################

### Setting the Dense_NN function parameters:
TrainingDataSet <- output_file_train
architecture <- "63,48,32,16,1"
ActivationFunctions <- "relu,relu,sigmoid,none"
Optimizer <- "SGD"
Loss <- "MSE"
epochs <- "150"
n_TrainingSample <- "-1"
BatchSize <- "32"
LearningRate <- "0.03"
LR_dynamic <- ""
MinDelta <- "0.001"
WeightInitialization <- "XAVIER"

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
          "--dataset=ames_test.csv"
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
rs_sp_real <- prediction$true_label * (Price_max - Price_min) + Price_min
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
    
#################### End of script ###########################################
