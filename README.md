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


## Installation

Clone the repository:
```sh

$ git clone [https://github.com/ese-msc-2021/irp-jz1121.git](https://github.com/ACSE-jz1121/Crater_detection_self_training.git)

Install dependencies

$ pip install -r requirements.txt
```


## License

Under the Apache 2.0 License.

## Contact
* Janice Zhao jz1121@ic.ac.uk

## Acknowledgements
Many thanks my supervisors for their supportive suggestions:
* Prof Gareth Collins
* Dr Joel Davis
* Dr Beg Marijan 
