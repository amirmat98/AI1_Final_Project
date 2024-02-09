# Description
[Università degli studi di Genova](https://unige.it/en/ "University of Genova")

Professor: [Armando Tacchella]("Armando.Tacchella@unige.it" "Armando Tacchella")

Student: [AmirMahdi Matin](https://github.com/amirmat98 "AmirMahdi Matin")  - 5884715 - Robotics Engineering 

# ARTIFICIAL INTELLIGENCE FOR ROBOTICS I PROJECT

# Table of Contents
- [Project Summary](#Project-Summary)
- [Installation](#Installation)
- [Running](#Running)
- [Architechture](#Architechture)
    - [JumpyGridWorld.py](##JumpyGridWorld.py)
    - [QLearningAgent.py](##QLearningAgent.py)
    - [Plotter.py](##Plotter.py)
    - [Globals.py](##Globals.py)
    - [main.py](##main.py)
    - [Pseudocode](##Pseudocode)
- [Plots](#Plots)
- [Functioning Mechanism](#Functioning-Mechanism)
- [Result](#Result)
- [Conclusion](#Conclusion)
- [Possible Enhancements](#Possible-Enhancements)



## Project Summary

In this project, a Q-learning algorithm is implemented for a Jumpy Grid World environment. In this environment, an agent navigates a grid layout that contains obstacles and a goal point within the grid. The agent is able to carry out actions, one of which is a one-of-a-kind leap action, and it also learns to optimize its course in order to maximize the cumulative benefits it receives. JumpyGridWorld, QLearningAgent, and Plotter are few examples of the modular components that are included in the source. In the main.py file, the training loop is responsible for orchestrating the learning process, which includes gathering data for visualizations and changing Q-values. Plots consist of steps for each episode, rewards for each episode, cumulative rewards, exploration decay, a heatmap of Q-values, and a visualization of policy grades. A number of chances for improvement are presented by the implementation, including the possibility of fine-tuning parameters, investigating other grid sizes, enhancing exploration techniques, refining visuals, and taking into consideration parallelization for bigger settings. For example, the implementation allows for straightforward parameter modifications. Through its whole, the project offers a comprehensive investigation of Q-learning within the context of a dynamic grid world, complete with detailed visualizations for the purposes of analysis and interpretation.

## Installation

It is necessary to have Python3 installed in order to actually run the project. Additionally, the following command should be used to install the necessary libraries onto your system:
```bash
pip install numpy matplotlib
```

## Running

To run the software, you must first clone the project, and then you must run the main.py script thereafter. Launch the terminal and execute the commands that are listed below in the terminal.

```bash
git clone https://github.com/amirmat98/AI1_Final_Project.git
```
```bash
cd AI1_Final_Project
```
```bash
python3 main.py
```

## Architechture

### JumpyGridWorld.py
The **JumpyGridWorld.py** module is responsible for defining the environment in which the Q-learning agent acquires knowledge. In addition to providing methods for resetting the environment, validating valid locations, detecting obstacles, and carrying out actions, the JumpyGridWorld class is responsible for initializing a grid world with obstacles. The learning task is made more difficult by the inclusion of a random component, which is introduced by the leap action.

### QLearningAgent.py
The script known as **QLearningAgent.py** is responsible for encapsulating the logic of the Q-learning agent. A Q-table is kept up to date by the QLearningAgent class, which is responsible for storing Q-values for every state-action pair. Methods for selecting actions by employing an epsilon-greedy strategy, updating the Q-table based on rewards and future Q-values, and determining the best policy are all included in this system. The learning rate, discount factor, and exploration rate of the agent are all factors that can be individually adjusted.

### Plotter.py
The **plotter.py** program is primarily concerned with visualization functionalities that are designed to assist in comprehending the behavior of the Q-learning agent. Matplotlib is utilized by the module in order to generate heatmaps for Q-values and policy grades. The module includes functions such as visualizePath, visualizeQValuesOnAxes, and visualizePolicyGradeOnAxes. The updatePlots function is responsible for providing dynamic updates to plots while training is being performed. These updates display the evolution of rewards, steps per episode, Q-values, ideal path, and policy grades.

### Globals.py
**Globals.py** is a straightforward utility file that declares global variables rewards and stepsPerEpisode. Its purpose is to act as a standard utility file. In the course of the training process, these variables are responsible for storing cumulative rewards and steps for each episode. Through the process of making them global, many components of the project will have the ability to simply access and update the shared data.

### main.py
The **main.py** file is responsible for coordinating the entirety of the reinforcement learning process. Initially, it sets up the JumpyGridWorld environment by utilizing the parameters that have been supplied, which include the grid size, the number of obstacles, and the action size. An instantiation of a Q-learning agent follows this step. across the process of iterating across episodes, the training loop updates the Q-table based on the actions taken by the agent and the answers received from the environment. Collecting data for visualization purposes includes things like prizes, the number of steps taken in each episode, grid layouts, and performance indicators. The last step of the script involves determining and publishing the optimal policy based on the Q-values that have been learned, as well as displaying a number of charts that illustrate the progression of the training.

### Pseudocode

## Plots

## Functioning Mechanism

The agent uses an epsilon-greedy strategy to investigate the environment, and it updates its Q-values based on the rewards it receives and the Q-values it will receive in the future.
To better illustrate how much progress has been made in learning, plots are revised at the conclusion of each episode.
The training loop does not end until a predetermined number of episodes have been completed.
Learned Q-values are used to derive the best strategy, which is then implemented.
The outcomes, the best possible remedy, and the best possible policy are printed.

## Result

## Conclusion


## Possible Enhancements
- **Dynamic Environment** During the training process, it is important to allow for dynamic changes in the environment.
- **Deep Q-Learning** When dealing with complicated policies, it is recommended to implement a deep Q-learning strategy.
- The introduction of more diversified grid worlds that feature a variety of sizes, forms, and obstacles is referred to as **more complex environments**.
- **Parallelization** If you want to learn material more quickly, you should think about parallelizing the training loop.











