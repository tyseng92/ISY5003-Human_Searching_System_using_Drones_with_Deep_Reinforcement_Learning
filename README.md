# ISY5003 Intelligent Robotics Systems Practice Module Group Project
---

## Project Title
## Optimizing persistent airborne surveillance coverage for human detection with deep reinforcement learning
  
---
## Executive Summary
Airborne surveillance has become increasingly accepted across  multiple  countries.Consequently,  due  to  the COVID-19 pandemic, it has brought about an increased need for  monitoring and surveillance to ensure compliance for social distancing, quarantine periods and behaviours that might encourage the spread of COVID-19. However, there are health  risk posed to individuals while enforcing these new rules. This has led us to explore autonomous drones to survey, find and identify specific person of individuals to monitor. Countries like Singapore, have deployed drones to monitor certain hot spots, albeit through the manual piloting of the UAV drones. This initial proof-of-concept can be used to identify non-compliant individuals within the community or find an individual in a indoor area to preventhe/she from jeopardising other’s health. Deep Reinforcement Learning, which has proven to perform well in dynamic environments where only partial observability is available, was used to assess and minimize the time required to find a person-of-interest. To simulate this, we created a custom environment in Microsoft AirSIM and have conformed it to a indoor area with high ceilings. Cognition of drone positionis achieved via the IMU sensor and UAV obstacle detectionis done using the laser range sensor pointed at multiple directions. This use-case can be further expanded to time-sensitivesearch tasks and potentially be employed to places with highconcentration of people like the airport or a mall.  Re-ID is planned for  future  works. Our comparison between our baseline model, Recurrent-DQN and Reccurent-DDPG is presented in this report.


---
## Team Members
Members: Teoh Yee Seng, Ng YanBo

---
## Installation Guide

### System Requirements
1. OS: Windows 10
2. Python: 3.7.0 (Anaconda or Miniconda distribution preferred)
3. Microsoft AirSim (v1.6 on Windows 10)
4. Nvidia GPU (Prefer Turing Architecture with VRAM 16 GB) 
5. More than 16GB RAM in PC to store memory buffer data

### Downloading Code
1. Clone this project: `git clone https://github.com/tyseng92/ISY5003-IRS-Practice-Module`
2. Change Directory: `cd ISY5003-IRS-Practice-Module`

### Python Environment and Dependencies
1. Create new conda environment: `conda env create -f airsim.yml`
2. Switch environment: `conda activate airsim`
3. (Optional)Update environment: `conda env update --file airsim.yml --prune`

### Custom AirSim Environment Executable
1. Copy the settings.json file from the "UE4\setting_json" folder into the right folder for the AirSim to initialize the environment correctly (usually in the "C:\\Users\\{UserName}\\Documents\\AirSim" path).
2. Download and unzip the custom environment called [HumanTrackingDrone_Env (950.MB)](https://drive.google.com/file/d/1Er62EeK0vh_1oO_XnZ6e7UDz57tjCdxy/view?usp=sharing)
3. Launch the "HumanTrackingDrone" custom environment by double-clicking on `run.bat`

### Setup YOLOV4 DArknet Packages
1. Download and unzip the binary package for the [Yolov4 Darknet](https://drive.google.com/file/d/1ivw945z4TkUbPruh-qdIAZOiU7IQTVG7/view?usp=sharing).
2. Move the "darknet" folder to the root folder.  

### Custom Dataset for Human Targets
We have trained the Yolov4 model based on our custom datasets for human target. The dataset can be downloaded from [here](https://drive.google.com/drive/folders/1NDvq1xzwWvUw2np6C5V2KXaCrT3SqLmL?usp=sharing).

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

### Adjust GPU VRAM usage
To change the VRAM usage, open the `<model>.py` code in any editor and Ctrl+F for `per_process_gpu_memory_fraction`. Adjust the value according to your GPU VRAM. 

## SIMULATION VIDEO DEMO
Below are the link for the demo video of the drone searching system based on Random Actor, RDQN and RDDPG.

* [Random Actor Demo](https://youtu.be/v8Di07hC5-U)
* [RDQN Demo](https://youtu.be/mWKVdg_JyNo)
* [RDDPG Demo](https://youtu.be/Gde0IXyrWVY)

---
## ACKNOWLEDGEMENT AND REFERENCES

Special thanks to Sung Hoon Hong and raymondng76 for providing the methods to control drones with deep reinforcement learning:
* [sunghoonhong/AirsimDRL: Autonomous UAV Navigation without Collision using Visual Information in Airsim](https://github.com/sunghoonhong/AirsimDRL)
* [Aerial filming with synchronized drones using Reinforcement Learning](https://github.com/raymondng76/IRS-Practice-Module-Dev.git)

---
