## Hardware Architectures of Deep Learning

## Lab1:
**Content of each of the Python Notebooks**
Here we list the general content of all our files with their names.
1. Part1_Parametrized_CNN : This notebooks computes the accuracy of several CNNs focusing on its development when the number of epochs used for training is varied.
2. Part1_AccuracyGraphs_ConstantEpochs: This notebook computes the bar graphs showed in the report. The plots showcase the variation of accuracy based on the variation of different parameters for a constant number of epochs.
3. Part2_MLP1: The notebook tries different number of neurons for one hidden layer, say from 30 to 300 neurons with a step size of 30 and compares the accuracy.
4. Part2_MLP1: This notebook trains a neural network implementing the LeNet-5 model which processes images of size 28*28.
5. Part2_LeNet-5 model: This notebook train a neural network implementing the LeNet-5 model to deal with hand-written digits recognition problem, with dataset MNIST. And record and compare the inference performance, namely, classification accuracy, run time and memory consumption.

**Description for Part 1 code**

Parameterized MLP and CNN module used to test the performance on small dataset CIFAR10 and MNIST. Implemented with PyTorch.

```python
# Multilayer perceptron model
MLP(n_hidden_layers, hidden_neurons, input_size, n_classes)
```

```python
# Convolutional neural network model
ConvNet(n_conv, n_fc, conv_ch, filter_size, fc_size, pooling_size, input_size, input_channels, n_classes, activation_fn) #same usage as LeNet
```

```python
# Training function
train(model_params, model_name, device, epochs)
```

**MLP**
|Parameter|Description|
|---|----|
|n_hidden_layers|number of hidden layers|
|hidden_neurons|list of size of hidden layers|
|input_size|dimension of input data|
|n_classes|categories of output classes|

**ConvNet**
|Parameter|Description|
|----|---|
|n_conv|number of convolutional layers|
|n_fc|number of full connected layers|
|conv_ch|list of channels of each conventional layer|
|filter_size|list of filter size of each conventional layer|
|fc_size|list of size of each full connected layer|
|pooling_size|shared size of squared pooling layer|
|input_size|dimension of input data|
|input_channels|channels of input data|
|n_classes|categories of output classes|
|activation_fn|type of activation function|

**train**
|Parameter|Description|
|---|----|
|model_params|dict of parameters unwrapped when instantiated a new model|
|model_name|name of model, used to specific the path to load or store the model|
|device|designate the device to be trained on|
|epochs|total training epochs|

**Description for Part 2 code**  
In this part, we use the MNIST database to train our code and you can find in every file we use this address to download the dataset.  
`D:/dataset/`  
This is our local address so it maybe different on another computer. Please make sure you choose an available address.

**The MLP model**  
Notice that there are acyually 2 different MLP2 models notebook. The first one named as as`MLP2_DigitsRecognition(l1)` is where we set 100 neurons for the second hidden layer and make changes to the first one. And the second one named as `MLP2_DigitsRecognition(l2)` is the opposite.

**The LeNet-5 model**
1. Import library like torch, torchvision, matplotlib. config some hyperparameters.
2. Load MNIST training and test datasets using torchvision.
3. Define a LeNet-5 Neural Network, including the convolution and pooling layers and full connections.
4. Define a Loss function with CrossEntropyLoss and optimizer with SGD.
5. Train the network with epoch 2.
6. Test LeNet-5 on the test data and output the Accuracy of Classification.
7. Use PyTorch Profiler to characterize the execution time and memory consumption.

**Hints**
1. At the very begining of the notebook, we set some parameters for training. You can change these parameters to get a different traing performance.  
`num_epochs = 2`  
`batch_size = 4`

2. In our code, we extensively utilized functions from the Matplotlib library to create visualizations.  
   It's important to note that when calculating CPU utilization and Memory usage, we employ  
   `with record_function('model_inference'):`  
    to consolidate the overall scenario. Here, you can observe the comprehensive inference details of this model.
   For creating tables, we use `print(prof.key_averages().table(sort_by='cpu_time_total', row_limit=10))`  
    where `row_limit=10` assists in primarily focusing on the parts consuming significant CPU resources, but this might lead to incomplete display of the table's content. Consequently, the total data for model_inference might not seem to match the sum of individual groups, yet the results are indeed accurate. If you wish to view the complete content, you can remove "row_limit=10".
