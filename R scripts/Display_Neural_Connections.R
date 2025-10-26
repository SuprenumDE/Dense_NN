################################################################################
##                  Calling the C++ program Dense_NN.exe
##            Outsourced script for displaying the network connections
##                                (PoC)
##
##            The script is currently designed for regression models!
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
## Version 0.0.2, 21.10.2025
## R-Version: 4.5.1  
## OS Windows 11
##
################################################################################


### Representation of neural connections:

# Threshold value for weight display:
threshold <- 0.2  # 0 = Show all connections

# Initialization:
all_nodes <- data.frame()
all_edges <- data.frame()

# Path to the directory containing the CSV files:
prefix <- "weights" 
verzeichnis <- file.path(getwd(), prefix)

# Find all Weight files:
weight_files <- list.files(verzeichnis, pattern = "_weights\\.csv$", full.names = TRUE)


# Processing of all weight matrices:
for (i in seq_along(weight_files)) {
  
  weights <- t(as.matrix(read.csv(weight_files[i], header = FALSE)))
 
  from_layer <- paste0("Layer_", i - 1)
  to_layer   <- paste0("Layer_", i)
  
  edge_list <- which(abs(weights) > threshold, arr.ind = TRUE)
  
  from_ids <- paste0(from_layer, "_N", edge_list[, 1])
  to_ids   <- paste0(to_layer, "_N", edge_list[, 2])
  
  # Capture edges:
  edges <- data.frame(
    from   = from_ids,
    to     = to_ids,
    value  = abs(weights[edge_list]) * 10,
    title  = paste("Gewicht:", round(weights[edge_list], 3)),
    color  = ifelse(weights[edge_list] > 0, "blue", "red")
  )
  all_edges <- rbind(all_edges, edges)
  
  # Capture nodes:
  from_nodes <- data.frame(
    id    = from_ids,
    label = from_ids,
    group = from_layer,
    title = paste("Neuron", from_ids)
  )
  to_nodes <- data.frame(
    id    = to_ids,
    label = to_ids,
    group = to_layer,
    title = paste("Neuron", to_ids)
  )
  all_nodes <- rbind(all_nodes, from_nodes, to_nodes)
}

# Remove duplicates:
all_nodes <- all_nodes[!duplicated(all_nodes$id), ]

# Define layer groups:
layer_names <- unique(all_nodes$group)
layer_names <- c(layer_names, "Layer_Output")

# Initialize Positions:
all_nodes$x <- NA
all_nodes$y <- NA

# Vertical Positioning
for (i in seq_along(layer_names)) {
  layer <- layer_names[i]
  layer_nodes <- which(all_nodes$group == layer)
  num_nodes <- length(layer_nodes)
  
  if (num_nodes == 1 && i > 1) {
    # Center single nodes (e.g., output) on previous layer
    prev_layer_nodes <- which(all_nodes$group == layer_names[i - 1])
    prev_y <- all_nodes$y[prev_layer_nodes]
    all_nodes$y[layer_nodes] <- mean(prev_y, na.rm = TRUE)
  } else {
    all_nodes$y[layer_nodes] <- seq(0, num_nodes * 50, length.out = num_nodes)
  }
  
  # Horizontal Positioning
  all_nodes$x[layer_nodes] <- i * 400
}

# --- Generate forced output nodes ---
output_id <- "Layer_Output_N1"
if (!(output_id %in% all_nodes$id)) {
  last_layer <- paste0("Layer_", length(weight_files))
  prev_layer <- paste0("Layer_", length(weight_files) - 1)
  prev_nodes <- all_nodes[all_nodes$group == prev_layer, ]
  
  output_node <- data.frame(
    id    = output_id,
    label = "Output",
    group = "Layer_Output",
    title = "Regression Output",
    x     = (length(weight_files) + 1) * 400,
    y     = mean(prev_nodes$y, na.rm = TRUE)
  )
  all_nodes <- rbind(all_nodes, output_node)
  
  # Final weight matrix: connections to the output layer:
  last_weights <- as.matrix(read.csv(weight_files[length(weight_files)], header = FALSE))
  edge_list <- which(abs(last_weights) > threshold, arr.ind = TRUE)
  
  from_ids <- paste0(prev_layer, "_N", edge_list[, 1])
  to_ids   <- rep(output_id, length(from_ids))
  
  output_edges <- data.frame(
    from   = from_ids,
    to     = to_ids,
    value  = abs(last_weights[edge_list]) * 10,
    title  = paste("Gewicht:", round(last_weights[edge_list], 3)),
    color  = ifelse(last_weights[edge_list] > 0, "blue", "red")
  )
  all_edges <- rbind(all_edges, output_edges)
}


View(all_edges) # If you want, you can view the network data frame here.


# Display visNetwork:
visNetwork(all_nodes, all_edges, main = "Network of all layers") %>%
  visOptions(highlightNearest = TRUE, nodesIdSelection = TRUE) %>%
  visLegend() %>%
  visNodes(x = all_nodes$x, y = all_nodes$y) %>%
  visPhysics(enabled = FALSE)

#------------------------------------------------------------------------------
### EoS