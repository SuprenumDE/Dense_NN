# Dense_NN
## Dense_NN is an EigenNET project for training dense-type neural networks implemented in C++.

### Features:
<ul>
<li>Dense network architecture with adjustable depth and activation functions</li>
<li>Eigen-based matrix operations for maximum speed</li>
<li>Minimal code footprint, maximum control</li>
<li>Ideal for demonstrations, teaching, and research</li>
</ul>



**Here is an overview of what Dense_NN does ...**

![Realization](https://github.com/SuprenumDE/EigenNET/blob/main/images/Realization.png).

**... and all this without a gigantic framework such as TensorFlow!**

Dense_NN, *version 0.1.09* dated September 3, 2025, was developed under ISO standard C++20 and Eigen version 3.4.0.

Dense_NN can be installed on *Windows using an msi file*, and the program execution arguments are displayed via “help”:

![Dense_NN Help](https://github.com/SuprenumDE/EigenNET/blob/main/images/Dense_NN_Help.png)

### The neural network/deep learning network can be trained using the following arguments:

| Argument            | Description                                                                                                                |
|:---------------     |:---------------------------------------------------------------------------------------------------------------------------|
| **--mode**          | Mode: 1=Train, 2=Test, 3=Exit (default: 1)                                                                                 |
| **--dataset**       | Data set file (e.g., filename.csv)                                                                                         |
| **--architecture**  | Architecture of the Dens Network. Layer sizes separated by commas, e.g. 784,128,64,10                                      |
| **--activations**   | Activation functions per layer, separated by  commas, e.g., relu, sigmoid, tanh, softmax, none (Identity)                  |
| **--weights**       | Name of the file in which the network weights are stored, e.g., weights.csv. They are stored in a                          |
|                     | directory under the Dense_NN working directory. (default: weights.csv)                                                     |
| **--epochs**        | Number of training epochs (default: 100)                                                                                   |
| **--loss**          | Loss function: CROSS_ENTROPY, MAE, or MSE  (default: CROSS_ENTROPY for classification models)                              |
| **--samples**       |Number of training/test samples (default: 1000)                                                                             |
| **--batch**         |Minibatch size (32, 64, or 128) (default: 64)                                                                               |
| **--val**           |Proportion of data for validation (e.g., 0.2, means that 20% of the training data is used for validation.) (default: 0.2)   |
| **--learning_rate** |Learning rate (default: 0.01)                                                                                               |
| **--lr_mode**       |Adjusting the learning rate using the *decay* or *step* methods (default:"")                                                |
| **--init**          |Network weight initialization method (HE, XAVIER) (default: HE)                                                             |

