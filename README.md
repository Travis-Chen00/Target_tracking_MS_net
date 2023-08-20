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

T = Time  
N = Swarm Size  
R<sub>pred  – 1</sub> = Five Temperature sensors  
_g<sub>r</sub><sup>n</sup>(t)_: The value of temperature sensor at Time t  
_p<sub>r</sub><sup>n</sup>(t)_: The real value of temperature sensor at Time t


### Heat Zone
<span style="color:red">Red: *Dangerous Zone*.</span> 
- Too close to the target, swarms should avoid. 

<span style="color:orange">Orange: *Safety Zone*.</span> 
- All swarms in this zone are perfect. 

<span style="color:blue">Blue: *Cold Zone*.</span> 
- Too far away from the target.

<img src="/img/Heat_Zone.png" alt="Heat Zones" width="250"/>

**Heat intensity** [\[10.1162/isal_a_00650\]](https://www.mitpressjournals.org/doi/10.1162/isal_a_00650) \( \delta \) decreases with distance to the target as given by

<img src="/img/heat_intensity.jpg" alt="Heat intensity" width="600"/><br/>
L<sub>x</sub> and L<sub>y</sub>:the lengths and the widths of the grid world.<br/>
D<sub>ST</sub> is the Euclidean distance to the target<br/> 
δ (delta) is the temperature rate for these three zones, which are 0.2 for the blue zone, 0.7 for the orange zone, and 0.1 for the red zone.


### Sensors
5 Temperature sensors + 1 proximity sensor
- Temperature Sensor is in the same location as S0<br/>
<img src="/img/agent.jpg" alt="Heat Zones" width="250"/>

## Training Strategy

**Genetic Algorithm**

We employ the [Genetic Algorithm](https://en.wikipedia.org/wiki/Genetic_algorithm) as our primary training strategy.

**Dynamic mutation**


**Catastrophe**

## Target Moving Strategy

The target will move in the beginning of each two generations, the range of its location is [0, SizeX], [0, SizeY]

## Experiments & Evaluation
### Basic Experiment Setting
## Simulation Parameters

All parameters in the basic experiment:

| **Parameter**                             | **Value**                      |
|-------------------------------------------|--------------------------------|
| Area size (L<sub>x</sub>, L<sub>y</sub>)  | (15,15)                        |
| Swarm size \(N\)                          | 10                             |
| \# of sensors \(R<sub>g</sub>\)           | 6                              |
| \# of sensors \(R<sub>pred</sub>\)        | 5                              |
| sensor values \(S<sub>r</sub>\)           | [0.108, 0.525, 0.087] / [0, 1] |
| Action value \(a<sub>0</sub>)             | [0, 1]                         |
| Action value \(a<sub>1</sub>\)            | [-1, 1]                        |
| Red zone size                             | 2 x 2                          |
| Orange zone size                          | 4 x 4                          |
| Blue zone size                            | Area size – (Red + Orange)     |
| Population size \(\mu\)                   | 5000                           |
| Number of generations \(g<sub>max</sub>\) | 100                            |
| evaluation length \(T\) (time steps)      | 10                             |
| Elitism                                   | 1                              |
| Mutation rate \(p<sub>mut</sub>\)         | 0.3                            |
| Catastrophe rate \(p<sub>cata</sub>\)     | 0.4                            |
| Number of triggers catastrophe            | 300                            |

### Basic Results
The experiment uses these parameters in the previous table, the results can be seen in the folder [Target_tracking_MS_net/results/basic_experiments](/Target_tracking_MS_net/results/basic_experiments)
In this experiment, the target will move each generation with North, South, East, and West directions.
### Control experiments
1. Swarm size: The swarm sizes are 10, 20, and 50, the grid world is 15 * 15 with only one target.
<br/>
2. Grid size: The grid world are in the range of {11, 20}. In this experiment, the agent number is 10 and target number is 1. 
<br/>
3. Target size: In the reality, a target can be any size or shape. This experiment discover the bigger and tiny targets with grid world 15 * 15 and 10 agents.
<br/>
The **results** can be seen in [Target_tracking_MS_net/results/control_experiments](/Target_tracking_MS_net/results/control_experiments)

### Complex scenario
1. **Multi-target**: There are 2 targets which are moving all the time, we use 10 agents to track both of them.
2. **Randomly moving target**: Since the target movement is uncertain, we make it randomly move in this grid world.
3. **Target disappearing & appearing**: The target will disappear on a 2 times basis, and then appear again.
<br/>
All **results** can be seen in the folder [Target_tracking_MS_net/results/control_experiments](/Target_tracking_MS_net/results/control_experiments)

### Evaluation
We compare our model (TAMS) with the traditional genetic algorithm (GA), and minimal surprise network (MS).
This project adopts four metrics to evaluate these three models, which are Correct, Waste, Realize, and Leave. The definition are shown as follows,
1. **Correct Number**:

2. **Time waste**:

3. **Time realize**:

4. **Leaving probability**:

