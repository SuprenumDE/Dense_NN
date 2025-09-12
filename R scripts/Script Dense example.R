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
## Version 0.0.6, 03.09.2025
## R-Version: 4.5.1  
## OS Windows 10/11
##
################################################################################
# 
# _________.__                 __________        __   
# /   _____/|  |__   ____      \______   \ _____/  |_ 
# \_____  \ |  |  \_/ __ \      |       _// __ \   __\
# /        \|   Y  \  ___/      |    |   \  ___/|  |  
#   /_______  /|___|  /\___  >     |____|_  /\___  >__|  
#   \/      \/     \/             \/     \/      
#   
# EIGENnet - Neural networks in C++ with Eigen,   Version 0.1.09, 03.09.2025

# Usage:
#  ./Dense_NN [Optionen]
# 
# Available options:
# Train or test a neural network
# Usage:
#   Dense_NN [OPTION...]
# 
# -m, --mode arg           Mode: 1=Train, 2=Test, 3=Exit (default: 1)
# -f, --dataset arg        Data set file (e.g., filename.csv)
# -c, --architecture arg   Layer sizes separated by commas, e.g. 784,128,64,10
# -x, --activations arg    Activation functions per layer, separated by commas, e.g., relu, relu, tanh, softmax
# -w, --weights arg        Weights file (e.g., weights.txt) (default: weights.txt)
# -e, --epochs arg         Number of epochs (default: 100)
# -z, --loss arg           Loss function: CROSS_ENTROPY, MAE, or MSE (default: CROSS_ENTROPY)
# -s, --samples arg        Number of training samples (default: 1000)
# -b, --batch arg          Minibatch size (32, 64, or 128) (default: 64)
# -v, --val arg            Proportion of data for validation (e.g., 0.2) (default: 0.2)
# -l, --learning_rate arg  Learning rate (default: 0.01)
# -r, --lr_mode arg        Learning rate mode (e.g., decay, step) (default: "")
# -i, --init arg           Initialization method (HE, XAVIER, ...) (default: HE)
# -d, --min_delta arg      Minimal improvement for early training stop (default: 0.0001)
# -p, --print_config       Outputs the current configuration
# -h, --help               Displays help

# Example:
#  ./Dense_NN --architecture=784,128,64,10 --activations=relu,relu,softmax --epochs=100 --lr=0.01 --dataset=mnist.csv

#Further information:
#  Guenter Faes, Mail: spv@faes.de
#  YouTube: https://www.youtube.com/@r-statistik
#  GitHub:  https://github.com/dein-repo/EIGENnet
# ################################################################################

system("./Dense_NN --help")

################################################################################

### Required packages:
library(ggplot2)      # für heatmad
library(rsample)      # to create a training and test data set
library(AmesHousing)  # Datensatz ames (Alternative zu Boston Housing)

################################################################################
#
#                       Classification Model
#
#
################################################################################

### Clear all variables:
rm(list = ls())

### Functions:
source("scale_features.R")  # scaling function

############################ Prepare data ##########################
#
# It is assumed that the training data/test data file is in csv format.
# The challenge is to determine whether the label information is in the first or
# last column in the data set.
# The program train_nn.exe assumes that the label column is in the first
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


# Export without header:
write.table(na.omit(data_ready), file = output_file, sep = ",", row.names = FALSE, col.names = FALSE)

cat("✅ DFile successfully exported as", output_file, "\n")

# Diagnosis: readLines(output_file, n = 5)


######################## Perform training ##############################

### Doing a little “labeling”:
Trainingsdatensatz <- "neu_mnist_train.csv"
Architektur <- c(784,128,64,10)
Aktivierungsfunktionen <- c("relu,relu,softmax")
Epochen <- 150
n_Trainingssample <- 60000
Batch_Size <- 128
Lernrate <- 0.01
Lr_dynamisch <- "decay"
Gewichtinitalisierung <- "HE"

# Alternative: res <- system2("Dense_NN.exe",... -> Stores feedback in res
#              stdout = TRUE  -> in case of problems
#              stderr = TRUE  -> in case of problems

system2("Dense_NN.exe",
      args = c(
       "--mode=1", 
       "--architecture=784,128,64,10", 
       "--activations=relu,relu,softmax", 
       "--loss=CROSS_ENTROPY", 
       "--dataset=neu_mnist_train.csv", 
       "--epochs=150", 
       "--samples=60000", 
       "--batch=128", 
       "--learning_rate=0.01",
       "--lr_mode=decay",
       "--min_delta=0.001", 
       "--init=HE"
      ),
      stdout = "",
      stderr = ""
)
#cat("Exit-Status:", attr(res, "status"), "\n")
#cat(res, sep = "\n")

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

Vorhersage <- read.csv("test_results.csv")
View(Vorhersage)

# Overall prediction accuracy (VG):
Mittlere_VG <- mean(Vorhersage$true_label == Vorhersage$predicted_label)
Mittlere_VG


### Presentation as a tabular overview:
# Absolutely:
Tabelle_VG <- table(Vorhersage$true_label, Vorhersage$predicted_label)
Tabelle_VG
# Relative (%):
Tabelle_VG_relativ <- prop.table(Tabelle_VG, margin = 1) * 100
round(Tabelle_VG_relativ, 1)
# Overall view based on 100:
Tabelle_VG_gesamt <- prop.table(Tabelle_VG) * 100
round(Tabelle_VG_gesamt, 2)


################################################################################
#
#                          Regressions-Modell
#
#
################################################################################

### Clear all variables:
rm(list = ls())

### Reload functions:
source("scale_features.R")  # scaling functions

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
ames_test <- testing(split)

# ----- Export as CSV file -----
output_file_train <- "ames_train.csv"
output_file_test <- "ames_test.csv"
# training dataset:
write.table(na.omit(ames_train), file = output_file_train, sep = ",", row.names = FALSE, col.names = FALSE)
# test dataset:
write.table(na.omit(ames_test), file = output_file_test, sep = ",", row.names = FALSE, col.names = FALSE)


######################## Perform training ##############################

### For overview purposes only:
Trainingsdatensatz <- output_file_train
Architektur <- c(63,48,32,16,1)
Aktivierungsfunktionen <- c("relu,relu,sigmoid,none")
Epochen <- 150
n_Trainingssample <- 2632
Batch_Size <- 32
Lernrate <- 0.03
Lr_dynamisch <- ""
Gewichtinitalisierung <- "XAVIER"

system2("Dense_NN.exe",
        args = c(
          "--mode=1", 
          "--architecture=63,48,32,16,1", 
          "--activations=relu,relu,sigmoid,none", 
          "--loss=MSE", 
          "--dataset=ames_train.csv", 
          "--epochs=150", 
          "--samples=2632", 
          "--batch=32", 
          "--learning_rate=0.03",
          "--min_delta=0.00001", 
          "--init=XAVIER"
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
