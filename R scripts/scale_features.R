##                    Calling the C++ program Dense_NN.exe
##                    R functions for data preparation
##                                (PoC)
### 
##Motivation:
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

# ------------------ Scaling function -----------------------------

scale_features <- function(df, method) {
  
  if (method == "minmax") {
    
    return(as.data.frame(lapply(df, function(x) {
      
      if (max(x) == min(x)) {
        return(rep(0, length(x)))  # 0 bleibt 0!
      } else {
        return((x - min(x)) / (max(x) - min(x)))
      }
    })))
    
  } else if (method == "zscore") {
    
    return(as.data.frame(lapply(df, function(x) {
      if (sd(x) == 0) {
        return(rep(0, length(x)))
      } else {
        return((x - mean(x)) / sd(x))
      }
    })))
    
  } else {
    
    return("Unknown scaling method!")
    
  }
  
  return(df)
  
}

# ------------------ Ende Skalierungfunktion ------------------------

