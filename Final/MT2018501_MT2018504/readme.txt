The final "model.py" python program will work for the final reduced dataset i.e. "final_ver1.2.csv" file. This file was created by taking into consideration the data of 400 odd buses over a period of 6 days. I have uploaded the file to gdrive

link : https://drive.google.com/open?id=1XZ-7IJl_W8da3qQV-FG2tPlxTJ35KBy9

Download the train set from the above link.
The test set works fine as it was provided. 
To run the model.py you would need 3 files :
    1. model.py
    2. "final_ver1.2.csv"
    3. "test.csv"

If you want to run my scripts, there is a bit of manual work involved. Run the scripts on "w1.csv" (week 1 data) by the following steps:
    1. "chunking.py" : This will create chunks and a "bus_id.csv".
    2. "bus_id.csv" has to be cleaned manually. It is in a set format. The '{' and '}' brackets need to be removed and it has to be transposed and stored as a pandas series in "bus_id_3.csv". I will provide the gdrive link for this too in NOTE.
    3. "ver2.py" has to be run next. This separates data according to various bus IDs.
    4. "final.py" is the third script to be run.
    5. "'handling date and time.py'" is run next to clean and reformat timestamp data.
    6. "removing_redundency.py" is next. It removes outlier data. This creates the "final_ver1.2.csv".
    7. After this you can run the model as was mentioned above.
    
NOTE:
    1. If you are planning to run all my scripts, its going to take almost 2-3 days.
    2. Link for "bus_id_3.csv" : https://drive.google.com/open?id=16QCWc98vtwxcPFYofoc8Ce2WlRrzV_sq
