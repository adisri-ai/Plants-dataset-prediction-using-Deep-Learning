# Python Course Project 
# Team No.7 : Use of Hybrid Binary Optimization Algorithms for Binary Image Classification
Datasets Used:
LCS25000
Plants-Type Datasets
Steps to implement:
Before running any code the default structure of the directory looks like:
/Code Folder
  Code.py (contains the main code of the project) Shown as code.py in video demonstration
  requirements.txt (Contains all the packages required for the project)
/Readme.txt
/PPT for the project (.pptx format)
/Video (.mp4 format)

Follow the given steps to check the project output:

Step 1: Downloading the datasets
The datasets can be downloaded using the following links:
1.LCS25000 : https://www.kaggle.com/datasets/javaidahmadwani/lc25000/data
2.Plants type dataset : https://www.kaggle.com/datasets/yudhaislamisulistya/plants-type-datasets
After downloading the datasets make sure you push them into the directory of the project.

Step 2: Installing dependencies(Optional)
Open the directory of the folder in the terminal and run the command:

pip install -r requirements.txt

Note : This step is optional because you need not install the dependencies if they are already installed in your system or 
if you are running this code on online cloud platforms such as google colab where all the libraries are installed by default

Step 3: Running Code.py file

Code.py is the file consisting of the main code of the model which performs all the tasks.
To run the file:
Open the directory in Terminal
Run the command:

python Code.py "<Name_Of_Zip_File>.zip"

Ex: python Code.py "lc25000.zip"

Once the code is run it performs the following tasks step-by-step
     Task 1: Extraction of zip-files : The code extracts the zip files present in the directory

     Task 2: Defining the CNN-Model  : The code next defines the CNN model whose description can be visible in the output log

     Task 3: Training the CNN-Model  : The code begins to train the CNN Model on the training data. 
                                       This task is done in 10 epochs as will be visible in the output logs. 
                                       At the end of each epoch you can see the current accuracy and validation of accuracy of the model

     Task 4: Use of Hybrid Optimizer : The step 4 will make use of Hybrid PSO+BEOSA(through research paper) for optimizing the parameters 
                                       of the CNN model. This will be performed in 18 iterations. At the end of this step you can see the 
                                       final validation accuracy of the code.
                                       Note: You might continuously see some warning logs in your output log during this step.
                                             These should be ignored as they don't impact the final result

     Task 5: Training Ensemble model : The code will now add an additional ML-Layer with ensemble learning(voting ensemble) consisting of 
                                       multiple ML-based models. This might be visible through the logs.
                                       Note: At the end of the step 5 you might see some new files added to the directory(with extensions
                                       .pkl , .h5)/ These are the files that store extracted features from the CNN. 
     Task 6: Accuracy metrics        : Finally the code begins to implement 5-fold n-cross validation to compute the metrics on the testing
                                       data
     Task 7: Saving The Final model  : At the end, the code saves the final model in another file added to the directory which can used in
                                       future to make predictions

Note: To run another dataset for the code we recommend to delete/move all the auxiliary feature extractor/model files (.h5 , .pkl , etc.) 
      to prevent conflict with other dataset's feature files.
