To run the code successfully, it is necessary to have the following installed:

- Python 2 with relevant scientific and plotting libraries scipy, numpy and matplotlib
- Pybrain, a library containing code pertaining to the use of neural networks. This package doesn't appear to be maintained anymore, which is why python 3 is not used in the project.

Description of files:

- functions.py : Python file containing functions that are necessary to run the code.
- tests.py : Python file containing functions that test the content in functions.py.

- walk_data_synchronization.ipynb : During the data collection, information from the robot's microphone and servers are transmitted with certain time lags. This notebook demonstrates the synchronization process and results.
- walk_data_demonstration.ipynb : Demonstration of ego-noise prediction and subtraction on the Aldebaran Nao robot.
- head_data_demonstration.ipynb : Same, but done on only one joint in order to see if the implementation is actually working.

The repository will be reorganized very soon so ignore the other files for now :)
