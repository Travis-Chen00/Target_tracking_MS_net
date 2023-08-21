# Target Tracking with Minimal Surprise network

## Model

**Target-Aware Minimal Surprise (TAMS)**

Our model is based on the Target-Aware Minimal Surprise (TAMS) . This approach integrates the concept of "heat" into the minimal surprise network for the target tracking. <br/>
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
Here are two strategies for the target movement:
``` Python
    1. Basic:
        while generation < Max_Gen:
             if random.random() < 0.5:  # West & East
                 X = x + direction[randInd]
                 Y = y
             else:  # North & South
                 X = x
                 Y = y + direction[randInd]
                 
             if 0 <= X <= SizeX and 0 <= Y <= SizeY and not block:
                move
    
    2. Advanced:
        while generation < Max_Gen:
            if generation % 2 == 0 and generation > 0:
                x = random.random(0, sizeX)
                y = random.random(0, sizeY)
                
                if 0 <= X <= SizeX and 0 <= Y <= SizeY and not block:
                move
```

## Experiments & Evaluation
### Basic Experiment Setting
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
| Number when triggers catastrophe          | 300                            |

Here are four settings to the fundamental experiment:
1. Put all agents in the blue zone. Excepted result: All agents stay in the blue zone.
2. Put all agents in the orange zone. Excepted result: All agents track the target.
3. Put agents both in the blue zone and the orange zone. Excepted result: Agents in the orange zone keep tracking, the blue zone keeps leaving.
4. Virtual setting: The agents are put in the blue zone, while we told them that you are in the orange zone. Expected result: Agents realise they are tricked and then keep tracking the target.

### Basic Results
The experiment uses these parameters in the previous table, the results can be seen in the folder **_(/Target_tracking_MS_net/results/basic_experiments)_**
In this experiment, the target will move each generation with North, South, East, and West directions.

### Control experiments
1. Swarm size: The swarm sizes are 10, 20, and 50, the grid world is 15 * 15 with only one target.
2. Grid size: The grid world are in the range of {11, 20}. In this experiment, the agent number is 10 and target number is 1.
3. Target size: In the reality, a target can be any size or shape. This experiment discover the bigger and tiny targets with grid world 15 * 15 and 10 agents.

The **results** can be seen in _**(/Target_tracking_MS_net/results/control_experiments)**_

### Complex scenario
1. **Multi-target**: There are 2 targets which are moving all the time, we use 10 agents to track both of them.
2. **Randomly moving target**: Since the target movement is uncertain, we make it randomly move in this grid world.
3. **Target disappearing & appearing**: The target will disappear on a 2 times basis, and then appear again.

All **results** can be seen in the folder **_(/Target_tracking_MS_net/results/control_experiments)_**

### Evaluation
We compare our model (TAMS) with the traditional genetic algorithm (GA), and minimal surprise network (MS).
This project adopts four metrics to evaluate these three models, which are Correct, Waste, Realize, and Leave. The definition are shown as follows,
1. **Correct Number**: Correct number means the number of agents who stays in the correct zone minus the number of agents in the red zone, for instance, an agent's inspire is in the Orange zone, it should keep in the orange zone.

2. **Time waste**: Time waste means the time when agents are not in their inspire, for example, if the agent's inspire is Orange, the time in the blue and red zone are the waste time. 

3. **Time realize**: This means the time when agents know they are not in the ideal zone, and how long they spent before enter the inspired zone.

4. **Leaving probability**: The probability means when agents enter and keep in the ideal zone, how likely is the agent will get out the zone.

