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
## Version 0.0.7, 17.11.2025
## R-Version: 4.5.1  
## OS Windows 10/11
##
################################################################################

# Read log file:
log <- read.csv("training_log.csv", skip = 1)

# ----------------------------- Output Graphic ---------------------------

par(mfrow = c(2,2),  oma = c(5, 0, 4, 0))  # unten, links, oben, rechts)

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
             ", Weight initial.: ", WeightInitialization,
             ", Optimizer: ", Optimizer),
      side = 1,
      outer = TRUE,
      line = 2,
      cex = 0.8)

par(mfrow = c(1,1))

#------------------------------------------------------------------------------

### Heatmap:

# Path to the directory containing the CSV files:
prefix <- "weights" 
verzeichnis <- file.path(getwd(), prefix)

# Find all Weight files:
weight_files <- list.files(verzeichnis, pattern = "_weights\\.csv$", full.names = TRUE)

# Read and visualize each file:
for (file in weight_files) {
  
  # Extract layer number
  layer_num <- sub("_weights\\.csv$", "", basename(file))
  
  # Read matrix
  weights <- as.matrix(read.csv(file, header = FALSE))
  
  # Convert to data frame
  weights_df <- as.data.frame(as.table(weights))
  colnames(weights_df) <- c("Input", "Neuron", "Weight")
  
  # Basic Heat Map
  p <- ggplot(weights_df, aes(x = Neuron, y = Input, fill = Weight)) +
    geom_tile() +
    scale_fill_gradient2(
      low      = "red",
      mid      = "white",
      high     = "blue",
      midpoint = 0
    ) +
    labs(
      title = paste("Heat map of weights – Layer", layer_num),
      x     = "Neuron",
      y     = "Input"
    ) +
    theme_minimal()
  
  # Fine tuning: legible axis labels
  p <- p +
    theme(
      axis.text.x = element_text(angle = 45, hjust = 1, size = 8),
      axis.text.y = element_text(size = 8)
    ) +
    scale_x_discrete(
      guide = guide_axis(check.overlap = TRUE)
    ) +
    scale_y_discrete(
      guide = guide_axis(check.overlap = TRUE)
    )
  
  # Output plot:
  print(p)
}



### End