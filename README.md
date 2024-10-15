# Optimization Genetic Algorithm Specialist

## Prerequisites

`pip install -r requirements.txt`

## Usage Instructions
1. Training
The first step is to train the model. To do this, set the MODE constant at the top of the optimization_genetic_algorithm_specialist.py file to 'Train'.

Once that is set, run the following command to start the training process:

`python optimization_genetic_algorithm_specialist.py`

Wait for the training process to complete.

2. Testing
After training is complete, you can proceed to test the model. To do this, set the MODE constant at the top of the same optimization_genetic_algorithm_specialist.py file to 'Test'.

Then, run the following command to execute the testing phase:

`python optimization_genetic_algorithm_specialist.py`

The model will now run on the test based on the training done earlier.

3. Generating Line Plots
To generate line plots, you need to run the following command:

`python line_plot.py`

Within the line_plot.py file, make sure to change the constant "ENEMY" for each enemy you wish to plot. 
After generating the plot, manually save the plot for each enemy.

4. Generating Box Plots of Individual Gains
To generate the box plots of individual gains, run the following command:

`python plot_individual_gains.py`

Once the plot is generated, save the image manually.

Done!