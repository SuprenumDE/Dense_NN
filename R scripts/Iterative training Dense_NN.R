################################################################################
##                Iterative network training with Dense_NN.exe
##                                 (PoC)
## 
## Motivation:
## Calling the C/C++ program Dense_NN.exe for iterative training and using an NN 
## and return to R.
##
##
## Ad-Oculos-Projekt, https://www.faes.de/ad-oculos/
## GitHub: https://github.com/SuprenumDE/Dense_NN
##
## Günter Faes, eigennet@faes.de
## Version 0.0.2, 29.10.2025
## R-Version: 4.5.1  
## OS Windows 11
##
################################################################################

### Required packages:


### Functions:
source("F:/R-Projekte/Calling_C_NN/scale_features.R")  # scaling function
source("F:/R-Projekte/Calling_C_NN/Display_Iterative_Training_Progress.R")

# The path to my R working directory (the bin directory containing Dense_NN.exe):
setwd("F:/R-Projekte/Calling_C_NN/bin")


############################ Prepare data ##########################
#
# It is assumed that the training data/test data file is in csv format.
# The challenge is to determine whether the label information is in the first or
# last column in the data set.
# The program Dense_NN.exe assumes that the label column is in the first
# position in the data set.

# ---- Configuration ----
input_file <- "mnist_train.csv"  # The complete training data set
label_column <- 1                # Index or name of the label column
scaling_method <- "minmax"       # Options: “minmax,” “zscore,” “none”

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


# ---- Splitting the training data set into 6 individual smaller data sets ----

# Number of splits:
n_splits <- 6
rows_per_split <- nrow(data_ready) / n_splits

# Check whether the number of lines is divisible by n_splits:
if (nrow(data_ready) %% n_splits != 0) {
  stop("The number of lines is not divisible by n_splits.")
}

# Split and Save:
for (i in 1:n_splits) {
  start_idx <- ((i - 1) * rows_per_split) + 1
  end_idx <- i * rows_per_split
  split_df <- data_ready[start_idx:end_idx, ]
  
  filename <- paste0("mnist_split_", i, ".csv")
  write.table(split_df, file = filename, sep = ",", row.names = FALSE, col.names = FALSE)

  cat("File successfully exported as", filename ,"\n")

}

data_size <- dim(split_df) # Only for labeling!


######################## Perform training ##############################

########################### 1. Training ################################

# The first network training is performed completely normally.
# This means that the network weights are initialized using the desired
# initialization method.

### Setting the Dense_NN function parameters:
TrainingDataSet <- "mnist_split_1.csv"
architecture <- "784,128,64,10"
ActivationFunctions <- "relu,relu,softmax"
epochs <- "150"
n_TrainingSample <- "-1"
BatchSize <- "64"
LearningRate <- "0.001"
LR_dynamic <- "decay"
WeightInitialization <- "HE"

system2("Dense_NN.exe",
        args = c(
          "--mode=1", 
          paste0("--architecture=", architecture), 
          paste0("--activations=", ActivationFunctions), 
          "--loss=CROSS_ENTROPY", 
          paste0("--dataset=", TrainingDataSet), 
          paste0("--epochs=", epochs), 
          paste0("--samples=", n_TrainingSample),
          paste0("--batch=", BatchSize), 
          paste0("--learning_rate=", LearningRate),
          paste0("--lr_mode=", LR_dynamic),
          "--min_delta=0.001", 
          paste0("--init=", WeightInitialization)
        ),
        stdout = "",
        stderr = ""
)

# ----------------------------- Output Graphic ---------------------------

display_iterativ_training_progress("training_log.csv")


########################### Iterative Training ################################

# For interactive training, the previously trained network weights are
# initialized using the --init argument with WeightInitialization <- “ITERATIVE”
# and then further improved using the new training data set.

start <- 2
end <- n_splits

for (i in start:end) {
  
  filename <- paste0("mnist_split_", i, ".csv")
  
  ### Doing a little “labeling”:
  TrainingDataSet <- filename
  WeightInitialization <- "ITERATIVE"


  system2("Dense_NN.exe",
          args = c(
            "--mode=1", 
            paste0("--architecture=", architecture), 
            paste0("--activations=", ActivationFunctions), 
            "--loss=CROSS_ENTROPY", 
            paste0("--dataset=", TrainingDataSet), 
            paste0("--epochs=", epochs), 
            paste0("--samples=", n_TrainingSample),
            paste0("--batch=", BatchSize), 
            paste0("--learning_rate=", LearningRate),
            paste0("--lr_mode=", LR_dynamic),
            "--min_delta=0.001", 
            paste0("--init=", WeightInitialization)
          ),
          stdout = "",
          stderr = ""
  )


  # ----------------------------- Output Graphic ---------------------------

  display_iterativ_training_progress("training_log.csv")
  
  
} # End for loop

#########################################################################
#
#                           NN-Test
#
#########################################################################

# The original MNIST test dataset was processed in the same way as the
# training dataset described in the R script “Scrit Dense example.R.”
# or in the "Prepare data" script section above.
# This means that the labels are in the first column and the features are
# normalized using the “minmax” method. 

# If you want, you can check this with this command:

  #readLines("processed_mnist_test.csv", n = 5)

# The first number is the label, i.e., the number to be trained, and the
# remaining numbers are the features. None of these should be less than
# zero or greater than 1.

system2("Dense_NN.exe",
        args = c(
          "--mode=2", 
          "--dataset=processed_mnist_test.csv"
        ),
        stdout = "",
        stderr = ""
)

#########################################################################
#
#                Review of test results
#
#########################################################################

prediction <- read.csv("test_results.csv")
View(prediction)

# Overall prediction accuracy (VG):
Average_VG <- mean(prediction$true_label == prediction$predicted_label)
Average_VG


### Presentation as a tabular overview:
# Absolutely:
Table_VG <- table(prediction$true_label, prediction$predicted_label)
Table_VG
# Relative (%):
Table_VG_relative <- prop.table(Table_VG, margin = 1) * 100
round(Table_VG_relative, 1)
# Overall view based on 100:
Table_VG_total <- prop.table(Table_VG) * 100
round(Table_VG_total, 2)


### EoS