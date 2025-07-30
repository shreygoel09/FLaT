# FLaT: Functional Langevin Transport

This repository contains the code for **FLaT** (Functional Langevin Transport), a framework for goal-directed biological sequence generation via latent Langevin transport and Jacobian-informed decoding.



### 1. Clone the repo
git clone https://huggingface.co/ChatterjeeLab/FLaT  
cd FLaT

### 2. Create the conda environment
conda create -n flat  
conda activate flat  
pip install -r requirements.txt


### 3. Run the jacobian sampling using the nohup command in the .sh file here:
src/scripts/run_jacobian2.sh