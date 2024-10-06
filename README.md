## Self-training system for small crater detection via IoU-based ensemble learning
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE.txt)

<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#project-description">Project Description</a>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#install-dependencies">Install Dependencies</a></li>
      </ul>
    </li>
    <li><a href="#repository-structure">Repository Structure</a>
      <ul>
        <li><a href="#data_source">Data_source</a></li>
        <li> <a href="#dataset_generation">Dataset_generation</a></li>    
          <li><a href="#self-training">Self_training</a></li>
            </ul>
        </li>
      </ul>   
    </li>
     <li><a href="#run-all">Run all</a></li>
    <li><a href="#installation">Installation</a></li>
    <li><a href="#usage-of-code">Usage of code</a></li>
    <li><a href="#Contact">Contact</a></li>
    <li><a href="#Acknowledgements">Acknowledgements</a></li>
  </ol>
</details>




## Project Description
 The objective of this work is to detect impact craters with diameter less than 1km. YOLOv5 model has been utilized as a powerful object detection tool for crater detection. An auto-iterative self-training system has been proposed to detect small craters with diameter less than 1km, which incorporate an IoU-based ensemble learning strategy as an extension of the Pseudo-label selection metric. 

The source code for YOLOv5 can be found on [YOLOv5](https://github.com/ultralytics/yolov5). This model provides the benchmark for this project, the instruction on clone yolov5 repositorys is described in the [Self-training.ipynb](https://colab.research.google.com/drive/1EAfCg-VbW3svYKns10DNAWDH35J66xAu?usp=sharing)

## Data source

The dataset used in this project is generated from the Robbins and Hynek database 2020 version that counts > 38400 craters on mars with diameter greater than 1km. THEMIS DAY IR mosaic images [12] are linked to this database according to their range of longitude and latitude. 

Robbins and Hynek dataset can be found on [database download]( http://craters.sjrdesign.net/)
THEMIS image can be found on [THEMIS](https://astrogeology.usgs.gov/search?target=&system=&p=1&accscope=&searchBar=)


## Getting started
### Install Dependencies

* pycm>=1.22.4
* pandas>=1.4.2
* argparse>=1.4.0
* numpy>=1.22.4
* cv>=1.0.0
* Pillow==9.5.0
* pylablib>=1.8.1
* random2>=1.0.1
* mathematical>=0.5.1
* (Optional) GPU/multi GPUs with CUDA

## Repository structure

This project is divided into three sections: dataset generation, data augmentation, and self-training, each of which is listed in its own folder. Each folder's Run.ipynb file gives instructions for the relevant section. RUN-ALL.ipynb contains instructions for running all parts. The Google Colab sharing link can be found in the RUN-ALL.ipynb file which allows access to the self-training system.

Go through these four folders and run the Run.ipynb in each section:

### [Data_source](Data_source/Data_unzip.ipynb):  Unzip the Robbins dataset and THEMIS image into your workplace. 

### [Dataset_generation](Dataset_generation/Run_dataset_generation.ipynb):  Generate the dataset by linking Robbins database with THEMIS image. 

### [Data_augmentation](Data_augmentation/Run_data_augmentation.ipynb):  Implement the background erasing data augmentation technique.

### [Self_training](https://colab.research.google.com/drive/1EAfCg-VbW3svYKns10DNAWDH35J66xAu?usp=sharing):  Contain the Auto-iteration module and ensemble module for the self-training system. Please run the self-training platform in Google Colab shared link. 

## Run all

You can also run codes from all section by:

1. Run [RUN-ALL.ipynb](RUN-ALL.ipynb) to run code from all sections step by step. 

2. The self-training section can be accessed via [Google Colab shared document](https://colab.research.google.com/drive/11mKvfHt0N6-gUwDJAqpTIl3vM1aoDblB?usp=sharing)

## Installation

Clone the repository:
```sh

$ git clone https://github.com/ese-msc-2021/irp-jz1121.git

Install dependencies

$ pip install -r requirements.txt
```



## Usage of code

The dataset generation and data augmentation section is developed in Anaconda and self-training section is built on Colab. This section will introduce the code to execute the following function. You can also follow these code in [RUN-ALL](RUN-ALL.ipynb)

### 1. Slice THEMIS images and label craters based on Robbins dataset and split the dataset

Arguments:
* --pathimg, 
default =  '../Data_source/THEMIS_DayIR_ControlledMosaic_LunaePalus_00N270E_100mpp.jpg'
```sh
python Dataset_generation/Robbins_Themis_generation.py

python Split_dataset.py --Val_number {120} --Test_number {120}
```

### 2. Run data augmentation
Arguments:
* mix_ratio， default = 0.3
* pathimg， default = "../Dataset_generation/train/images"
* pathtxt， default = "../Dataset_generation/train/labels"
* pathsave， default = 'data_aug/images'
```sh
python Data_augmentation/background_erasing.py
```

### 3. Run Self-training system
This model can only be accessed through google colab as it links to YOLOv5. [Click here](https://colab.research.google.com/drive/1EAfCg-VbW3svYKns10DNAWDH35J66xAu?usp=sharing)
Ensemble.py is the IoU-based ensemble module within the self-training platform:
Arguments:
* pathimg: image path
* pathtxt: array of label paths of two model's detection results
* pathsave: path to save the result labels
* pathgt: ground truth path
* n_models: number of models
```sh
python Self_training/ensemble.py --gt {pathgt} --nargs {pathtxt} --pathimg {pathimg} --n_models {n_models}  --pathsave {pathsave} 
```

### 4. SFD plot
The SFD plot is generated using Size-Frequency-distribution placed in [Reult_visualization](Results_visualization) folder. 


## License

Under the Apache 2.0 License.

## Contact
* Janice Zhao jz1121@ic.ac.uk

## Acknowledgements
Many thanks my supervisors for their supportive suggestions:
* Prof Gareth Collins
* Dr Joel Davis
* Dr Beg Marijan 
