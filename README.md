# Target Tracking with Minimal Surprise network

## Model

**Heat Minimal Surprise (Heat-MS)**

Our model is based on the Heat Minimal Surprise (Heat-MS) framework. This approach integrates the concept of "heat" into the minimal surprise network for the target tracking.

## Training Strategy

**Genetic Algorithm**

We employ the [Genetic Algorithm](https://en.wikipedia.org/wiki/Genetic_algorithm) as our primary training strategy.

## Target Moving Strategy

The target employs the following moving strategy:

```python
if threshold >= 0.68:
    if swarms_in_high_zone < 3:
        Moving
```

## Evolution
### Grid size

- Grids = {15, 20}
- Agents Number: 50 
- Basic Sensors: 14
- Temperature Zone = 1x1 + 3x3

### Sensor model

- Basic Sensors: {6, 14, 8}
- Agents: 50
- Grid = 15
- Temperature Zone = 1x1 + 3x3

### Agents Number

- Agents: {10, 20, 50, 100}. 
- Grid = 15
- Basic Sensors: 14
- Temperature Zone = 1x1 + 3x3

### Temperature Zone

1. High zone: {1x1, 2x2}
2. Medium zone: {2x2, 3x3}

- Grids = 15
- Agents Number: 50 
- Basic Sensors: 14

### Algorithm Compare

- Traditional genetic Algorithm
- Minimal Surprise + genetic algorithm
- Heat-MS + genetic algorithm
