################################################################################
##                  Calling the C++ program Dense_NN.exe
##            Outsourced script for displaying the training progress
##                                (PoC)
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
## Version 0.0.1, 30.11.2025
## R-Version: 4.5.2  
## OS Windows 10/11
##
## Workflow:
## The deep learning structure is trained on the Raspberry Pi5. After training,
## the files training_log.csv and nn_parameter.json are copied to the evaluation
## computer (desktop/laptop) via FTP, read out using this R script, and the
## training progress is displayed graphically.
################################################################################

### Required packages:
library(jsonlite)

# ---------------------- Read Data ---------------------------------------
# Read in information about the trained deep learning structure:
DL_Param <- fromJSON("nn_parameter.json")

# Read log file:
log <- read.csv("training_log.csv", skip = 1)

# ----------------------------- Output Graphic ---------------------------

par(mfrow = c(2,2), oma = c(5, 0, 3, 0), mar=c(5,4,4,2))

plot(log$Epoch, log$Loss, type = "l", col = "darkblue",
     main = "Course of the loss function",
     ylab = "Loss",
     xlab = "Epoch")

plot(log$Epoch, log$Val_loss, type = "l", col = "blue",
     main = "Loss function validation process",
     ylab = "Loss",
     xlab = "Epoch")

plot(log$Epoch, log$Classification_rate, type = "l", col = "darkgreen",
     main = "Correct Classification",
     ylab = "%",
     xlab = "Epoch")

plot(log$Epoch, log$Val_accuracy, type = "l", col ="green",
     main = "Correct classification Validation",
     ylab = "%",
     xlab = "Epoch")

# Information about the graphic:
# Central headline:
mtext(paste0("Training process for the data set ", DL_Param$dataset_file), outer = TRUE, cex = 1.2, line = -1.0)

# Footer:
if (DL_Param$n_TrainingsSample == -1) {
  n_TrainingSample <- "Full training data"
} else {
  n_TrainingSample <- DL_Param$n_TrainingsSample
}
activationsfunctions <- paste(DL_Param$activations, collapse = ", ")
architecture <- paste(DL_Param$architecture, collapse = " -> ")

Info <- paste0("Number of training samples: ", n_TrainingSample,
               ", Learning rate: ", DL_Param$learning_rate,
               " (", DL_Param$lr_mode, ")",
               ", Weight initial.: ", DL_Param$W_init_Methode,
               ", Optimizer: ", DL_Param$Optimizer,
               ", Architecture: ", architecture,
               ", Activation Functions: ", activationsfunctions)

wrapped <- strwrap(Info, width = 80)

for (i in seq_along(wrapped)) {
          mtext(wrapped[i],
          outer = TRUE,
          side = 1,
          cex = 0.8,
          line = 2 +(i - 1)) }

par(mfrow = c(1,1))

#------------------------------------------------------------------------------



### EoS
