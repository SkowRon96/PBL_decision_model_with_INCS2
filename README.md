# PBL_decision_model_with_INCS2

Conda Enviroment
-------------
To create conda enviroment with appropriate packages, use such command:

**conda create --name _env_ python=3.7.4**

where env is the name of the enviroment.Than activate enviroment:

**conda activate _env_**

and install appropriate packages (go to the location of the project):

**pip install -r requirements.txt**


Running decison model - INCS2_client.py
-------------

When the enviroment is activated input such command to run application:

**python .\INCS2_client.py --model <path_to_project>\IR\tf_model.xml --input client --device MYRIAD**

when running the application without Intel Neural Stick 2 remove device parameter.
