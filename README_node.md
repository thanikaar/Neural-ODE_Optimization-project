# Neural-ODE Optimization Project

Final project for **UGST4090 – Introduction to Optimization**. The project explores the mathematics behind Neural Ordinary Differential Equations (Neural ODEs). We also looked at how to implement the code and compares them against standard deep learning architectures (MLP, CNN, ResNet) on image classification tasks.

## Repository Structure

```
Neural-ODE_Optimization-project/
├── README.md                        # Project overview (this file)
├── Neural_ODE.pdf                    # Written report on Neural ODE theory and experiments
├── Neural_ODE_Mia_Nika.pdf           # Presentation of this project
├── node_lib.py                       # Reusable Neural-ODE building blocks (ODEF, ODEAdjoint,
│                                      #   NeuralODE, ode_solve) shared across notebooks
├── node_implementation.ipynb         # implementation of Neural-ODE (forward pass,
│                                      #   adjoint method, backpropagation) 
│                                     # (from https://github.com/msurtsukov/neural-ode )
├── mnist_resnet_and_node.ipynb       # MLP / CNN / ResNet / N-ODE comparison on MNIST
├── cifar10_resnet_and_node_1.ipynb   # MLP / CNN / ResNet / N-ODE comparison on CIFAR-10
└── cifar10_methods_benchmark.ipynb   # Benchmark of ODE solvers (Euler, Runge-Kutta orders 2–4)
                                       #   on CIFAR-10
```

`node_lib.py` holds the core classes so they can be imported (`from node_lib import ODEF, ODEAdjoint, NeuralODE, ode_solve`) instead of being redefined in every notebook.

## Abstract of `Neural_ODE.pdf`
Neural Ordinary Differential Equations appeared as a shift and improvement in deep learning, replacing the discrete sequence of transformations in standard neural networks with a continuous dynamical system. Instead of parameterizing a fixed number of layers, Neural ODEs parameterize the derivative of the hidden state using a neural network. The output is obtained by numerically integrating this derivative from an initial to a final state. In this report, we focus on presenting the overall concept and methodology behind Neural ODEs. This paper highlights the relevant ODE theory, connects Neural ODEs to residual networks, analyzes the adjoint sensitivity method, discusses advantages and limitations, and presents some applications including how the N-ODE is implemented in code and how N-ODE performs relative to other neural networks on image classification tasks.

more can be read on `Neural_ODE.pdf`, which is also available on this repository.


## Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/thanikaar/Neural-ODE_Optimization-project.git
cd Neural-ODE_Optimization-project
```

### 2. Install dependencies

The notebooks and `node_lib.py` rely on the following packages:

| Package | Used for |
|---|---|
| `torch` | Core tensors, autograd, `nn.Module` (Neural ODE, ResNet, CNN, MLP models) |
| `torchvision` | MNIST/CIFAR-10 `datasets`, image `transforms` |
| `numpy` | Numerical arrays |
| `scipy` | `solve_ivp` reference ODE solver (used in `node_implementation.ipynb`) |
| `pandas` | Logging/collecting benchmark results |
| `matplotlib` | Plotting results and figures |
| `seaborn` | Plot styling in `node_implementation.ipynb` |
| `Pillow` (PIL) | Image loading/handling |
| `tqdm` | Progress bars during training |
| `ipython` | `clear_output` for live-updating plots in notebooks |
| `jupyter` | Running the `.ipynb` notebooks |

Install everything with:

```bash
pip install torch torchvision numpy scipy pandas matplotlib seaborn pillow tqdm ipython jupyter
```

(`os`, `math`, `time`, `copy`, and `gc` are used as well but are part of the Python standard library, so no extra install is needed for those.)

### 3. Download the datasets

The MNIST and CIFAR-10 notebooks load their data from a local `data/` folder (`root='data'`) and will auto-download on first run via `torchvision.datasets`. If that download fails or you'd rather fetch the data yourself, grab it manually from the original sources and place it in a `data/` folder in the root of the repo, next to the notebooks:

- **MNIST**: https://yann.lecun.com/exdb/mnist/ (mirror if that's unreachable: https://github.com/cvdfoundation/mnist)
- **CIFAR-10**: https://www.cs.toronto.edu/~kriz/cifar.html

```
Neural-ODE_Optimization-project/
├── data/                  # <- put the downloaded MNIST/CIFAR-10 files here
├── node_lib.py
├── node_implementation.ipynb
├── mnist_resnet_and_node.ipynb
├── cifar10_resnet_and_node_1.ipynb
└── cifar10_methods_benchmark.ipynb
```

### 4. Run the notebooks

- Open `node_implementation.ipynb` to see the Neural ODE built from scratch, or import the reusable pieces from `node_lib.py` (`from node_lib import ODEF, ODEAdjoint, NeuralODE, ode_solve`).
- Run `mnist_resnet_and_node.ipynb` or `cifar10_resnet_and_node_1.ipynb` to reproduce the MLP/CNN/ResNet/N-ODE comparisons.
- Run `cifar10_methods_benchmark.ipynb` to reproduce the Euler vs. Runge-Kutta solver benchmark on CIFAR-10.
## Authors

Corduneanu-Huci Maria, Haltrich Thanika
