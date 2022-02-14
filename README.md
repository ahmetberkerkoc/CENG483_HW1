question1.py is writen for question1. It calculates the result for 3D color histogram without grid for all interval
question2.py is writen for question2. It calculates the result for per channel color histogram without grid for all interval
question3_2d.py is writen for question3-4-5. It calculates the result for per channel color histogram with grid [8,6,4,2]  for chosen interval (there is an interval variable)
question3_3d.py is writen for question3-4-5. It calculates the result for 3D color histogram with grid [8,6,4,2]  for chosen interval (there is an interval variable)

For all python file, you need to change dataset variable in for loop with comment #change dataset. 

Also according to selected configuration in report, you need to change interval variable for related dataset(query). This change needs for question3_2d.py and  question3_3d.py. There is an interval variable with comment to explain this change. 

For question3_2d.py and  question3_3d.py. 
Query1: interval -> 16
Query2: interval -> 16
Query3: interval -> 64


If you want to select only a specific grid type in question3_2d.py and  question3_3d.py. 

You can change this part. grid_type is a list [8,6,4,2]  ---> 8 for 12, 6 for 16, 4 for 24 and 2 for 48 i.e. enter 96/grid 
    for i in grid_type: --->  for i in grid_type[1:2]:



For example, if you want to run question3_2d.py or  question3_3d.py. with query3 with only grid_type 6 (96/16)

    - Change interval = 64
    - Change for i in grid_type[1:2]:
    - Change for data in dataset3:

For example, if you want to run question1.py or question2.py

    -Change only dataset  
        for query1 in dataset3: #change
        for Q_data in dataset3: #change 

