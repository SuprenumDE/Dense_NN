################################################################################
##                  Calling the C++ program Dense_NN.exe
##            Outsourced script for displaying iterative training progress
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
## Günter Faes, eigennet@faes.de
## Version 0.0.2, 29.10.2025
## R-Version: 4.5.1  
## OS Windows 11
##
################################################################################

display_iterativ_training_progress <- function(training_log_file) {
  
  
  log <- read.csv(training_log_file, skip = 1)
  
  par(mfrow = c(2,2),  oma = c(5, 0, 4, 0))
  
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
  mtext(paste0("Training process for the data set ", TrainingDataSet), outer = TRUE, cex = 1.2, line = -1.0)
  
  # Fußzeile:
  mtext(paste0("Number of training samples: ", n_TrainingSample,
               ", Learning rate: ", LearningRate,
               " (", LR_dynamic, ")",
               ", Weight initial.: ", WeightInitialization),
        side = 1,
        outer = TRUE,
        line = 2,
        cex = 0.8)
  
  par(mfrow = c(1,1))
  
}