# mmcs-project-uoe
This repository is for a group project from the course Methodology, Modelling, and Consulting Skills, University of Edinburgh.
It relates to the optimisation of a TV channel scheduling and advertising.

The project mainly uses python as the main language together with xpress as a solver.

-----

# Project Structure
### `main.py`
- a main file to start the code.
### `solver/solver.py` 
- a model of the solver.
### `data/*` 
- all data files.
### `config/config.py` 
- all configurations or constants used across the project.
### `utils/`
- `advert_conversion_rates.py` - functions for calculating the conversion rates.
- `data_processing.py` - functions for preprocessing data into desired dataframe before feeding into the model.
- `schedule_processing.py` - functions for preprocessing schedules data of each competitor.
- `utils.py` - general utility functions such as printing and creating folder.
### `data_exploration/*` 
- python notebook files used for playing with the data.

-----
# Dependencies
The required packages can be installed using pip or conda.
### PIP
```commandline
pip install -r requirements.txt
```
### Conda
```commandline
conda env create -f environment.yaml
```

-----
# Run
To run the solver, depending on the method used, follow the steps below.
By default, the solver soft time limit is set to 10 minutes 
and the movie data sent to the solver is picked from the top 5%.

The total runtime should take around 10 minutes for each week for a total of 12 weeks.
### Command Line
Run the following command.
```
python -m main
```
### Editor
Simply run the `main.py` file on the editor.

-----
