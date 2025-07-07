# AI Word Search Optimizer

An system that implements multiple AI algorithms to efficiently search for target words within a constrained lexical space.

## Technical Approaches

### Implemented AI Methods

- **Bayesian Inference**
- **Minimax Algorithm**
- **A* Search Algorithm**
- **Constraint Satisfaction Problem (CSP) Approach**
- **Hybrid Approaches**
  - **Three-Stage Hybrid**: Dynamically switches between Bayesian, A*, and Minimax based on search space size
  - **CSP-A* Hybrid**: Combines constraint satisfaction with heuristic search
  - **CSP-Bayesian Hybrid**: Integrates probabilistic inference with constraint-based filtering

## Performance Analysis
- Performance visualization using matplotlib
- Comparative analysis of algorithm effectiveness

## Requirements

- Python 3.6+
- Pygame
- Matplotlib
- NumPy

## Future Work

- Implementation of additional algorithms such as:
  - Monte Carlo Tree Search (MCTS)
  - Reinforcement Learning approaches
  - Neural network-based prediction models
- Optimization for larger search spaces
- Parallelization of search algorithms
