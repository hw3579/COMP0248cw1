
## Overview

This project is designed for **semantic segmentation** using the CamVid dataset. It includes scripts for data preprocessing, training, evaluation, and visualization of model performance.

## Project Structure

```
project_studentname/
├── data/
│   ├── camvid/                 # Raw CamVid dataset
│   │   ├── train/              
│   │   ├── val/                
│   │   ├── test/               
│   │   ├── class_dict.csv      # Class-color mappings
│   ├── annotations/            
│
├── src/
│   ├── dataloader.py           # Data loading + preprocessing
│   ├── train.py                # Training script with some model saving scripts
│   ├── eval.py                 # evaluation script
│   ├── eval_func               # some functions for eval
│   ├── visualize.py            # Visualization function for eval
│   ├── utils.py                # Some training functions
|   |—— model.py                # model file and some test function
│
├── README.md                    # Setup/usage instructions
```


## License

This project is released under the MIT License.


The GitHub repository link: 
https://github.com/hw3579/COMP0248cw1
