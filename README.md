# VisCogRep
Vision and Cognitive Systems Project Repo Fall 2023

The python notebooks found in this repository were run on Google Colab to utilize the resources there. The loaded dataset takes up a good amount of RAM and with their current parameters, each model uses almost all of the alotted GPU capacity at the free level in Colab.

To download the dataset, go to this link and download the zipped file: https://drive.google.com/file/d/15YHebAGrx1Vhv8-naave-R5o3Uo70jsm/view

Place that file into the local folder housing this repository.

The notebooks UNet and DEeepLabV3 load the dataset and then train their respective models on said data. As you move through the notebooks, they will load, augment, and seperate the datasets in similar fashion. They will then train on the separated training dataset, use the validation set to determine how the model is performing while training, and then test the final version of the model on the test set. The metrics outlined in the accompanying paper (see "Semantic Segmentation of Biotic Stress in Coffee Leaves using U-Net and
DeepLabV3: A Comparative Study" pdf in the repository) are displayed at the end of the notebooks.

All functionality shared between the UNet and DeepLabV3 notebooks, like data loading, data augmentation, etc. is found int he accompanying coffee_utils.py file. This helps to keep the notebooks cleaner from a user perspective.

The provided gitignore file does not permit the large model checkpoints that are created while training to be pushed to Git. It also does not allow the large dataset to be pushed to Git, please download that dataset from the link specified above.