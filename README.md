# Target Tracking with Minimal Surprise network

## Model

**Heat Minimal Surprise (Heat-MS)**

Our model is based on the Heat Minimal Surprise (Heat-MS) framework. This approach integrates the concept of "heat" into the minimal surprise network for the target tracking. <br/>
### Action Net
<img src="/img/Action_net.png" alt="Action network" width="500"/> <br/>
### Prediction Net
<img src="/img/Prediction_net.png" alt="Action network" width="500"/>

### Fitness function
Fitness function for this model<br/>
<img src="/img/fitness function.jpg" alt="Fitness formula" width="600"/><br/>

### Heat Zone
<span style="color:red">Red: *Dangerous Zone*.</span> 
- Too close to the target, swarms should avoid. 

<span style="color:orange">Orange: *Safety Zone*.</span> 
- All swarms in this zone are perfect. 

<span style="color:blue">Blue: *Cold Zone*.</span> 
- Too far away from the target.

<img src="/img/Heat_Zone.png" alt="Heat Zones" width="250"/>

The red zone, which has the highest temperature, represents the closest place to the target. If some swarms stay in this area, they can be an obstacle and block the movement of the target, so, swarms should avoid this area. Similarly, all positions in the blue zone are too far from the target. In this area, swarms can feel cold and fail to follow the target. On the other hand, the orange zone enjoys the second-highest temperature; this place illustrates perfect tracking. 
**Heat intensity** [\[10.1162/isal_a_00650\]](https://www.mitpressjournals.org/doi/10.1162/isal_a_00650) \( \delta \) decreases with distance to the target as given by

<img src="/img/heat_intensity.jpg" alt="Heat intensity" width="600"/><br/>
where L<sub>x</sub> and L<sub>y</sub> are the lengths and the widths of the grid world, D<sub>ST</sub> is the Euclidean distance to the target, and δ (delta) is the temperature rate for these three zones, which are 0.2 for the blue zone, 0.7 for the orange zone, and 0.1 for the red zone. Equation (1) normalized the heat intensity H between 0 and 1, resulting in maximum light intensity *H* = 1 in the target and a light intensity of approximately zero at the corners.


### Sensors
5 Temperature sensors + 1 proximity sensor
- Temperature Sensor is in the same location as S0
<img src="/img/14 sensors.png" alt="Heat Zones" width="250"/>

## Training Strategy

**Genetic Algorithm**

We employ the [Genetic Algorithm](https://en.wikipedia.org/wiki/Genetic_algorithm) as our primary training strategy.

## Target Moving Strategy

The target will move in the beginning of each generation, the range of its location is [0, SizeX], [0, SizeY]

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
