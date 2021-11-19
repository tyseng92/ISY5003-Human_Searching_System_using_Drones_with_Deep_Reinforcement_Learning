# ISY5003 Intelligent Robotics Systems Practice Module Group Project
---

## Project Title
## Optimizing persistent airborne surveillance coverage for human detection with deep reinforcement learning
  
---
## Executive Summary
Update introduction here.


---
## Team Members
Members: Teoh Yee Seng, Ng YanBo

---
## Installation Guide

### System Requirements
1. OS: Windows 10
2. Python: 3.7.0 (Anaconda or Miniconda distribution preferred)
3. Microsoft AirSim (v1.6 on Windows 10)
4. Nvidia GPU (Prefer Turing Architecture)

### Downloading Code
1. Clone this project: `git clone https://github.com/tyseng92/ISY5003-IRS-Practice-Module`
2. Change Directory: `cd ISY5003-IRS-Practice-Module`

### Python Environment and Dependencies
1. Create new conda environment: `conda env create -f airsim.yml`
2. Switch environment: `conda activate airsim`
3. Update environment: `conda env update --file airsim.yml --prune`

### Custom AirSim Environment Executable
1. Copy the settings.json file from the "UE4" folder into the right folder for the AirSim to initialize the environment correctly (usually in the "C:\Users\[UserName]\Documents\AirSim" path).
2. Download and unzip the custom environment called [HumanTrackingDrone]()
3. Launch the "HumanTrackingDrone" custom environment by double-clicking on `run.bat`

### Loading of Model Weights
1. Download and unzip the binary package for the Yolov4 Darknet.
2. Move the "darknet" folder to the root folder.  

### Training the Models
1. Start the Custum AirSim Environment.
2. Open Command Prompt and cd to the "code" folder in the repo.
3. To train the models from scratch, run `python <model>.py --verbose` in the command prompt, where the model can be replace by `random_d`, `rdqn`, and `rddpg`.
4. To continue the training using previous trained model, make sure the .h5 files are available, and execute `python <models>.py --verbose --load_model` without `--play` in the command prompt.
5. Press `Ctrl-C` to end the training process.

### Evaluate the Models
1. Start the Custum AirSim Environment.
2. Open Command Prompt and cd to the "code" folder in the repo.
3. To test run and evaluate the models, run `python <model>.py --play --load_model` in the command prompt, where the model can be replace by `random_d`, `rdqn`, and `rddpg`.
5. Press `Ctrl-C` to end the evaluation process.

## SECTION 5 : SIMULATION VIDEO DEMO
Below are the link for the demo video of the drone searching system based on Random Actor, RDQN and RDDPG.

* [Random Actor Demo](https://youtu.be/v8Di07hC5-U)
* [RDQN Demo](https://youtu.be/mWKVdg_JyNo)
* [RDDPG Demo](https://youtu.be/Gde0IXyrWVY)

---
## SECTION 6 : ACKNOWLEDGEMENT AND REFERENCES

Special thanks to Sung Hoon Hong and raymondng76 for providing the methods for deep reinforcement learning with drones:
* [sunghoonhong/AirsimDRL: Autonomous UAV Navigation without Collision using Visual Information in Airsim](https://github.com/sunghoonhong/AirsimDRL)
* [Aerial filming with synchronized drones using Reinforcement Learning](https://github.com/raymondng76/IRS-Practice-Module-Dev.git)

---