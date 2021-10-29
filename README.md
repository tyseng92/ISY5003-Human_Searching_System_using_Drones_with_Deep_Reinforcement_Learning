# ISY5003 Intelligent Robotics Systems Practice Module Group Project
---

## Project Title
## Optimizing persistent airborne surveillance coverage for human detection with deep reinforcement learning
  
---
## Executive Summary
Update introduction here.


---
## Team Members
Members  : Ng YanBo, Teoh Yee Seng

---
## Installation Guide

### System Requirements
1. OS: Windows 10
2. Python: 3.6.7 (Anaconda or Miniconda distribution preferred)
3. Microsoft AirSim (v1.6 on Windows 10)
4. GPU (Discrete GPU preferred for running environment, playing simulations and training)
5. For Ubuntu Setup:
   - Docker
   - nvidia-docker

### Downloading Code
1. Clone this project: `git clone https://github.com/tyseng92/ISY5003-IRS-Practice-Module`
2. Change Directory: `cd ISY5003-IRS-Practice-Module`
3. Follow further instructions below

### AirSim Environment Install/Run

#### Windows 10
1. Download and unzip your preferred environment from the [AirSim release 1.2.2 page](https://github.com/microsoft/AirSim/releases/tag/v.1.2.2)
2. Run the AirSim environment by double-clicking on `run.bat`


### Python Environment and Dependencies
1. Create new conda environment: `conda env create -f airsim.yml`
2. Switch environment: `conda activate airsim`
3. Update environment: `conda env update --file airsim.yml --prune`


### Loading of Model Weights
1. To convert darknet weights to tensorflow model, run `python save_model.py --model yolov4` in vision folder. 

### Running the simulation (Supported in Local only)


### Training the RL Models

#### Local Training
1. Ensure the AirSim environment is running
2. To train the models from scratch, execute `python <model>.py --verbose`. Options include
   - `rdqn.py`
   - `rdqn_triple_model.py`
   - `rddpg_triple_model.py`
3. To resume training, execute `python <models>.py --verbose --load_model`
3. To stop the training press `Ctrl-c`

## SECTION 5 : SIMULATION VIDEO DEMO

### Iteration 2
[![Iteration 2](http://img.youtube.com/vi/ZT0SEAQG_U0/0.jpg)](https://www.youtube.com/watch?v=ZT0SEAQG_U0 "Iteration 2")

---
## SECTION 6 : ACKNOWLEDGEMENT AND REFERENCES

- Code is based on the efforts of Sung Hoon Hong: [sunghoonhong/AirsimDRL: Autonomous UAV Navigation without Collision using Visual Information in Airsim](https://github.com/sunghoonhong/AirsimDRL)

- Additional Citations are in the report

Related weblink for the drone search operation:
- https://towardsdatascience.com/deep-q-learning-tutorial-mindqn-2a4c855abffc
- https://link.springer.com/article/10.1007/s00521-020-05097-x
- https://www.hindawi.com/journals/complexity/2018/6879419/
- https://onlinelibrary.wiley.com/doi/epdf/10.1002/rob.20226
- https://ieeexplore.ieee.org/document/6290694
---