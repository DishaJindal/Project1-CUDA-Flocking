**University of Pennsylvania, CIS 565: GPU Programming and Architecture,
Project 1 - Flocking**

* Disha Jindal: [Linkedin](https://www.linkedin.com/in/disha-jindal/)
* Tested on: Windows 10 Education, Intel(R) Core(TM) i7-6700 CPU @ 3.40GHz 16GB, GTX 222 222MB (Moore 100B Lab)

## Simulation
### 50k Boids
<p align="center"><img src="https://github.com/DishaJindal/Project1-CUDA-Flocking/blob/master/images/Coherent_50K.gif"></p>

### 5k Boids
<p align="center"><img src="https://github.com/DishaJindal/Project1-CUDA-Flocking/blob/master/images/Coherent_5K.gif"></p>

## Description
This is a Boids Flocking Simulation where the boids move around the simulation space according to three rules:
1. Cohesion - boids move towards the perceived center of mass of their neighbors
2. Separation - boids avoid getting to close to their neighbors
3. Alignment - boids generally try to move with the same direction and speed as their neighbors

Please refer [Instructions](https://github.com/DishaJindal/Project1-CUDA-Flocking/blob/master/INSTRUCTION.md) or [Conard Parker's notes](http://www.vergenet.net/~conrad/boids/pseudocode.html) for more details on the rules.
## Implementations
Boids Flocking Simulation is implemented using three algorthims.
#### Naive
In the naive implementation, we loop through all grid cells and the boids inside them to update the position and velocity of each boid.
#### Scattered Uniform Grid
In case of scattered, instead of looping through all cells, we look for cells at neghbouring distance away from the boid in all six directions. This drastically reduces the number of cells and boids checked. This is implemented in a manner such as we don't need any code change w.r.t the cells to be checked on changing the maximum neghbouring distance and hence, caters to the **Grid-Looping Optimization** section of the extra credits.
#### Coherent Uniform Grid
The algorithm for coherent uniform grid is very similar to the scattered one with a small difference in terms of memory access. The boids belonging one cell are stored in contiguous memory location to avoid the unnecessary hops and leading to better cache hit and hence, improving the runtime.
#### Optimized Coherent Uniform Grid
In coherent uniform grid, we are looking at boids in all cells in the cube from -d to +d in all directions. In this version, I am checking the boids in the cells which are in the sphere of radius d rather than the cube. This is done by checking hte distance between the center of the sphere and the center of the grid cell. If the distance is greater than the radius of the sphere plus the diagonal of the cell, then it is definately outside the sphere and all suhc cells are rejected.

## Performance Analysis
### Number of Boids
The following plot demonstrates the impact of increasing the number of boids in the simuation on the framerate. The three colors correspond to three implementations: Naive, Scattered and Coherent. The darker color bars are with visualization (VISUALIZE 1) and the corresponding lighter colors are without visualization (VISUALIZE 0).

<p align="center"><img src="https://github.com/DishaJindal/Project1-CUDA-Flocking/blob/master/images/FPS_Boids.png" width="600"/></p>

**Some Inferences:**
1. The framerate drops with the increase in the number of boids for all implementations.
2. There is a huge performance improvement after switching off the visualization but the performance gap is more evident and bigger when the number of boids are less.
3. This plot also highlights the performance improvement from naive to scattered uniform grid to coherent uniform grid.
#### Question1
For each implementation, how does changing the number of boids affect performance? Why do you think this is?

The following plot shows the impact of the number of boids on the performance for the three implements with visualization off.
<p align="center"><img src = "https://github.com/DishaJindal/Project1-CUDA-Flocking/blob/master/images/Q1_BoidCount_Naive_Scattered_Coherent.png" width="600"/></p>

It is very clear that the performance drops as we increase the number of boids. The curve is more steep in case of naive as compared to the other two because in Naive the number of boids checked for each boid is equal to the total number of boids but in case of the other two it depends upon the distribution of the boids in the surrounding cells and the increase in number of boids to be checked would not increase linearly with the increase in total number of boids.

### Block Size
The following plots demonstrates the impact of changing the block size on the framerate. As above, the three colors correspond to three implementations: Naive, Scattered and Coherent. The darker and dotted lines are with visualization (VISUALIZE 1) and the corresponding lighter colors are without visualization (VISUALIZE 0). The two plots have different base configurations. The one on the left has 5k boids and the one of the right has 50k boids.

<p align="center"><img src="https://github.com/DishaJindal/Project1-CUDA-Flocking/blob/master/images/FPS_Blocksize_5k.png" width="400"/> <img src="https://github.com/DishaJindal/Project1-CUDA-Flocking/blob/master/images/FPS_Blocksize_50k.png" width="400"/></p>

#### Question2
For each implementation, how does changing the block count and block size
affect performance? Why do you think this is?

The almost horizontal lines depict that the performance does not change significantly with the change in blocksize. The trend is same for all three implementations. One of the reasons for this behaviour might be that the threads in a block are not intereacting with each other and are working independently so they being in same block or different does not impact much. Also, we are not using the shared memory in any of the implementations so the they are not competing for memory.
### Cell Width
The following plot demonstrates the impact of increasing the width of a grid cell on the framerate. The three colors correspond to three implementations: Naive, Scattered and Coherent. The darker and dotted lines are with visualization (VISUALIZE 1) and the corresponding lighter colors are without visualization (VISUALIZE 0). I have considered 0.5d, 1d and 2d where d is equal to the maximum neighbouring distance since increasing the width further would not reduce the number of cells to be checked.
<p align="center"><img src="https://github.com/DishaJindal/Project1-CUDA-Flocking/blob/master/images/FPS_CellWidth.png" width="600"/></p>

#### Question4
Did changing cell width and checking 27 vs 8 neighboring cells affect performance?
Why or why not? Be careful: it is insufficient (and possibly incorrect) to say
that 27-cell is slower simply because there are more cells to check!

The plot shows that the framerate is maximum when the width of the cell is equal to the maximum neighbouring distance and it drops as we increase or decrease the cell width.

**Increase**: The implementation with 27-cells (width d) is faster than 8-cell one(width 2d). One of the reasons for this behaviour is that as we increase the width of the cell, the total grid volume that we check increases and so does the number of boids, e.g. in case of 8-cells, the width of each cell is 2d and the total volume checked is 64 d-cube whereas in case of 27-cells, the total volume checked is 27 d-cube.

**Decrease**: As we decrease the cell width from d to 0.5d, the framerate drops again. One of the reasons for that might be the increase in number of cells to be checked ber boid. The performance dip is most significant in case of coherent implementation because all boids for a cell are in a contiguous memory location but as we increase the number of cells to be checked significantly, the number of boids per cell would decrease and we will no longer be able to reap the caching benefits.


### Scattered vs Coherent
The following three plots are subsets of the above described plots. These highlight the performance improvement by changing a scattered uniform grid to a coherent uniform grid.
<p align="center"><img src="https://github.com/DishaJindal/Project1-CUDA-Flocking/blob/master/images/Q3_Scattered_vs_Coherent.png"></p>

#### Question3
For the coherent uniform grid: did you experience any performance improvements with the more coherent uniform grid? Was this the outcome you expected? Why or why not?

In all configurations, the coherent grid is performing better than the scattered which is as per the expectations. As we increase the number of boids, the performance drop of coherent is less because the boids are still in the contiguous locations, has lesser memory hops and better L1 Cache hit than scattered. The impact of cell width on these implementations is interesting as the dip of coherent is more as compared to scattered on decreasing the width from d to 0.5d whereas less on increasing it from d to 2d. This is because on decreasing the width coherent is becoming more like scattered as there are very few boids per cell and it can not use the benefits that coherent implementation has to offer.

### Coherent Uniform and Optimized Coherent Uniform
The following plot demonstrates the improvement after optimizing the coherent uniform grid approach to only consider the boids inside the sphere around the boid. 
<p align="center"><img src="https://github.com/DishaJindal/Project1-CUDA-Flocking/blob/master/images/Optimized_Coherent.png" width="500"></p>

This shows the performance difference for three different cell widths: 0.5d, 1d and 2d where d is equal to the maximum neighbouring distance. The performance increases with this optimization for all configurations and it increases more as we decrease the cell width because the number of unnecessary boids increase with smaller cell width.
