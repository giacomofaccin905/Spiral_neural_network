# Spiral Neural Network

A neural network built from scratch in NumPy (no deep learning framework) to separate two intertwined spirals, plus a visualisation of what gradient descent does on the loss surface.

![Loss landscape](1_loss_landscape.png)

## What it does

- Generates a 2D dataset of two intertwined spirals with Gaussian noise (10,000 points per class for training, a separate sample with a different seed for testing).
- Trains a fully connected network with forward pass, backpropagation and gradient descent all written by hand.
- Visualises the result: decision boundary, loss curve, and the loss surface along the path the optimizer followed.

## Model

| | |
|---|---|
| Architecture | 2 - 32 - 32 - 1 |
| Activation | tanh (all layers) |
| Initialisation | Normal, std = 1/sqrt(fan_in); biases at zero |
| Loss | 1/2 MSE, targets in {-1, +1} |
| Optimizer | Full-batch gradient descent, learning rate 0.5 |

## Results

| | Accuracy |
|---|---|
| Train | 99.97% |
| Test (new sample) | 99.98% |

![Decision boundary](2_decision_boundary.png)

The loss is unstable for roughly the first 1,800 epochs and then drops cleanly.

![Loss curve](3_loss_curve.png)

## Loss landscape

To see the path the optimizer took, the weights are saved every 10 epochs. A PCA (via SVD) of the trajectory, centred on the final weights, gives the two main directions. The loss is then evaluated on a grid along those two directions and plotted as a 3D surface, with the training path drawn on top.

The surface is not a simple bowl: there are ridges between the random start and the minimum, and the path is noisy before it reaches the basin.

Note: this is a 2D projection of a space with about 1,200 parameters, so it shows the shape along the directions the optimizer moved, not the full landscape.

## Run it

```bash
pip install numpy matplotlib jupyter
jupyter notebook Spiral_neuralnetwork.ipynb
```

Run the cells in order. Training runs on CPU, no GPU needed.

## Files

- `Spiral_neuralnetwork.ipynb`: the full notebook
- `1_loss_landscape.png`, `2_decision_boundary.png`, `3_loss_curve.png`: figures used above

## Author

Giacomo Faccin: [github.com/giacomofaccin905](https://github.com/giacomofaccin905)
