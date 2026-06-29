# Audio Emotion Regression
## Contact
Names: Rafael Pashkov, Amirth Raj Puramcheriyil

Date: 12/11/2025

COMP 5530: Deep learning 

## How to run
### Opening the notebook on Google Colab
This project is meant to be run on Google Colab. The libraries required are imported in the notebook itself, and are preinstalled in the Colab environment or get installed when running the notebook.

To access the notebook on Google Colab, click [here](https://colab.research.google.com/github/chesseraf/Audio-emotion-recognition/blob/main/Audio_Emotion_Regression.ipynb).

This link will open the notebook with no running results, only the code. To view a notebook with some results already showing, click [here](https://colab.research.google.com/github/chesseraf/Audio-emotion-recognition/blob/main/Sample_outputs.ipynb). This notebook has already been run with 5 epochs and shows some code outputs after quick training followed by evaluation.

If the first link does not work, follow these steps:
1) Go to Github at https://github.com/chesseraf/Audio-emotion-recognition
2) Open Audio_Emotion_Regression.ipynb file in the main branch of the repository.
3) Click on the "Open in Colab" button near the top of the page.
4) Click "Copy to Drive" in the toolbar at the top of the Colab page to save a copy of the notebook to your own google drive. You can run and edit this copy as instructed by the rest of the instructions.

To run it in Google Colab, you must be signed into a Google account. There is a enough 'colab points' given to run this project for free. In the unlikely scenario that your account does not have any points left, you can create a new Google account which will have enough points to run the project. For instructions on creating a new google accout, go [here](https://support.google.com/mail/answer/56256?hl=en).

### Setting up the folders
The first cell of the notebook requests to connect to the google drive of your google account. If you do not wish to connect your google drive to the program, there are a few additional steps that must be taken that are described in a subsection below. 
#### Once google drive is connected (mounted)
There are three different ways to get the data to be seen by the program. Methods listed earlier are easier to do.
1) Find and download the folder named Small_sample_data from this repository. Upload this folder into the root directory of your google drive, and rename it to Colab_Drive_Files.

2) Go to [data in drive](https://drive.google.com/drive/folders/1LqowgOE2kQnQa0FYctKAwd6vN_5OLTG3). There, choose one of the subfolders to download. Once downloaded (and unzipped if it got zipped), rename it to Colab_Drive_Files. Then upload it to your google drive. The folder at that link will be deleted sometime in 2026. 

3) This is what we did to get the data originally. Go to [HuggingFace](https://huggingface.co/datasets/laion/Emilia-with-Emotion-Annotations2/tree/main). Scroll down until the data files start with EN and not DE. Download one of them. Then unzip it. Then open the folder with the files sorted alphabetically. Keep only a small quantity (~800 mp3-json pairs with same prefix) of the files since google drive does not allow too many files in a folder, and the program runs faster. Upload this folder into your google drive into a folder named Colab_Drive_Files.

#### Using files in colab directly intead of connecting Google Drive
CAUTION - if you choose this option, you will need to re-upload any files you upload to Colab if the runtime is ever restarted.
1) You must add the data you wish to use directly to the colab workspace instead of putting it into your google drive. The files tab is the 2nd from the bottom icon on the left sidebar. 
2) Create a folder with the name Data.
3) Get a folder with the data downloaded onto your device but don't upload it to drive using one of the methods described above. 
4) Right click the colab folder you created named Data, and click upload. Select some or all of the files from your downloaded folder with all of the data. Make sure that an .mp3 and .json file with the same prefix both get uploaded (by sorting the folder files alphabetically). 
5) Change line 2 in the notebook from "DRIVE_MOUNTED = True" to "DRIVE_MOUNTED = False".

### Running the program
#### Run the program including training
1) (Optional) If you have a lot of time and want to run many epochs during training, set EPOCHS to a higher number in the 5th cell of the notebook. Currently it is set to 50. Also, the cell that has the run epoch function has the lines 
'''
if va_loss < 0.0045:
            break
'''
0.0045 can be increased or decreased to change the stopping criteria caused by validation MSE dropping below the threshold.
2) Click "Run All" in the top toolbar. 
3) Wait for it all to run. Some results will appear in the cell with run epoch and the cell with ccc metrics.
4) If you want to try your own mp3s, run steps 9 and 10 described below.
#### Run the program, but without training
1) Download the folder w2v2_temporal_head from this repository.
2) Upload that folder to the root directory of your Google drive. If you are not connecting your google drive, create a folder with that name in Colab and upload all (both) files from the downloaded folder into the one you created. 
3) The path to this folder must be the same as the variable SAVE_DIR which is declared in the first cell of the notebook. (E.g. no (1) from multiple downloads)
4) (Likely optional and done by default) Choose Python 3, T4 GPU as your runtime type. Can be set with the change runtime type option in dropdown near top right corner in Colab. 
5) Run the cells in order until you reach the cell the starts with "# Feature extractor & encoder (frozen)". This is about the first 10 cells.
6) Run the cell that is ~3rd from last that starts with "# Reload frozen base encoder". This loads the saved head. 
7) (Optional) Now run the ~6th to last cell the starts with "def ccc". This will print out some statistics about the model on this data set. 
8) If you want to predict emotions on your own mp3 files, make that mp3 file available in the colab workspace or your Google drive. 
9) In the 2nd to last cell, in the line "result = predict_attributes("/content/drive/MyDrive/meHappy.mp3")", change the path of the existing .mp3 file to the path of your mp3 file. 
10) Finally, run the last 2 cells. The last one will give you the predicted score of all 4 emotions in the ranges specified by the dataset. The ranges are shown in the 6th cell as well as on the [website](https://huggingface.co/datasets/laion/Emilia-with-Emotion-Annotations/blob/main/README.md) with the dataset.
