# Self-Training System for Small Crater Detection via IoU-Based Ensemble Learning
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE.txt)

<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#project-description">Project Description</a></li>
    <li><a href="#data-source">Data Source</a></li>
    <li><a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#install-dependencies">Install Dependencies</a></li>
      </ul>
    </li>
    <li><a href="#how-to-run">Workflow</a></li>
    <li><a href="#usage-of-code">Usage of Code</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgements">Acknowledgements</a></li>
  </ol>
</details>

## Project Description
This project aims to detect small impact craters on Mars, focusing on craters with diameters of less than 1 km. A YOLOv5 model has been employed as the primary object detection framework. An innovative auto-iterative self-training system is proposed, which integrates an IoU-based ensemble learning strategy to enhance small crater detection. This system extends the Pseudo-label selection metric for more accurate results.

The source code for YOLOv5 can be found in the official [YOLOv5 repository](https://github.com/ultralytics/yolov5). Detailed instructions for cloning and using the YOLOv5 repository in this project are provided in the [Self-training.ipynb notebook](https://colab.research.google.com/drive/1mWEVufggf9rZDaF8Us1aFWjDGQtajN7N?usp=sharing).

## Data Source
The dataset used for crater detection is based on the 2020 version of the Robbins and Hynek Mars Crater Database, which catalogs over 38,400 craters larger than 1 km in diameter. THEMIS Day IR mosaic images are linked to this database based on their longitude and latitude ranges.

- The Robbins and Hynek Crater Database can be downloaded from [this link](http://craters.sjrdesign.net/).
- THEMIS images can be accessed from [the USGS Astrogeology Science Center](https://astrogeology.usgs.gov/search?target=&system=&p=1&accscope=&searchBar=).

## Getting Started


## Installation-dependencies

Clone the repository:
sh

$ git clone [https://github.com/ese-msc-2021/irp-jz1121.git](https://github.com/ACSE-jz1121/Crater_detection_self_training.git)

Install dependencies

$ pip install -r requirements.txt

## Workflow
The first step to run this crater detection algorithm is to generate your training and detection dataset. The dataset and generation code can be found in the "data generation" folder.

To run the self-training system and start detecting smaller craters, follow the instructions provided in the [Self-training.ipynb notebook](https://colab.research.google.com/drive/1mWEVufggf9rZDaF8Us1aFWjDGQtajN7N?usp=sharing) on Google Colab.

To run the multi-resolution detection system for detecting medium to large-size craters, follow the instructions provided in the [multi-resolution-detection.ipynb notebook](https://colab.research.google.com/drive/1mWEVufggf9rZDaF8Us1aFWjDGQtajN7N?usp=sharing) on Google Colab.

## License

Under the Apache 2.0 License.

## Contact
* Janice Zhao jz1121@ic.ac.uk

## Acknowledgements
Many thanks my supervisors for their supportive suggestions:
* Prof Gareth Collins
* Dr Joel Davis
* Dr Beg Marijan 
