[![TECOLOGYxyz - AutomaticFlowerPhenology](https://img.shields.io/static/v1?label=TECOLOGYxyz&message=AutomaticFlowerPhenology&color=blue&logo=github)](https://github.com/TECOLOGYxyz/AutomaticFlowerPhenology "Go to GitHub repo")


This repository supports the paper *Automatic flower detection and phenology monitoring using time-lapse cameras and deep learning*.


<!-- PROJECT LOGO -->
<br />
<p align="center">
  <a href="https://github.com/TECOLOGYxyz/AutomaticFlowerPhenology">
    <img src="logo.png" "https://github.com/TECOLOGYxyz/AutomaticFlowerPhenology" alt="Logo" width="80" height="80">
  </a>

  <h3 align="center">Automatic detection of <i>Dryas integrifolia</i> and <i>D. octopetala</i> for phenology monitoring with time-lapse cameras. </h3>
  </p>
</p>



<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary><h2 style="display: inline-block">Table of Contents</h2></summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#usage">Additional code supporting paper</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#contact">Contact</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project


*ABSTRACT*
Advancement of spring is a widespread biological response to climate change observed across taxa and biomes. However, the species level responses to warming are complex and the underlying mechanisms difficult to disentangle. This is partly due to lack of data, which is typically collected by repeated direct observations, and thus very time-consuming to obtain. Data deficiency is especially pronounced for the Arctic where the warming is particularly severe. We present a method for automatized monitoring of flowering phenology of specific plant species at very high temporal resolution through full growing seasons and across geographical regions. The method consists of image-based monitoring of field plots using near-surface time-lapse cameras and subsequent automatized detection and counting of flowers in the images using a convolutional neural network. We demonstrate the feasibility of collecting flower phenology data using automatic time-lapse cameras and show that the temporal resolution of the results surpasses what can be collected by traditional observation methods. We focus on two Arctic species, the mountain avens Dryas octopetala and Dryas integrifolia in 20 image series from four sites. Our flower detection model proved capable of detecting flowers of the two species with a remarkable precision of 0.918 (adjusted to 0.966) and a recall of 0.907. Thus, the method can automatically quantify the seasonal dynamics of flower abundance at fine-scale and return reliable estimates of traditional phenological variables such as onset, peak, and end of flowering. We describe the system and compare manual and automatic extraction of flowering phenology data from the images. Our method can be directly applied on sites containing mountain avens using our trained model, or the model could be fine-tuned to other species. We discuss the potential of automatic image-based monitoring of flower phenology and how the method can be improved and expanded for future studies.


<!-- GETTING STARTED -->
## Getting Started

This repository contains stuff related to the paper and links to archived data. The flower detection model is built on the <a href="https://github.com/matterport/Mask_RCNN"><strong>Mask RCNN framework</strong></a>. We've made a <a href="https://github.com/TECOLOGYxyz/Mask_RCNN"><strong>copy</strong></a> (fork) of the repository, to make sure that it is available for you and added the customizations done for our flower detection model. The Matterport implementation is nicely domumented and contains some good tutorials. If this is your first time dipping your toes in the deep (learning) waters, you might want to go through some of these. If you experience any issued with Mask-RCNN, you can check the <a href="https://github.com/matterport/Mask_RCNN/issues"><strong>Matterport Mask-RCNN issues</strong></a> to see if other people have had similar problems and what they did to solve them. 

Our fork of Mask RCNN: <a href="https://github.com/TECOLOGYxyz/Mask_RCNN"><strong>TECOLOGYxyz/Mask_RCNN</strong></a>

We customized a few things for our use of Mask-RCNN. For detection, we added the option to run inference on all images in a folder and output the results in .csv format. We also added some info printing to the screen - processing time per image and stuff like that.
For training, we implemented a more elaborate augmentation scheme than what was part of the original training. This improved our results.

The above customizations are found in the TrainAndDetect.py script. As the name suggests, you can use this to both initiate training on you own data and to run inference on images with a trained model.

Two flower detetion models are published along with the paper. To get started using these models or to train your own, you need to download the models from the link below, clone the TECOLOGYxyz/Mask_RCNN repo and install the required python packages.

Download the flower detection models: 


Clone the TECOLOGYxyz/Mask_RCNN repo
1. ```sh
   git clone https://github.com/TECOLOGYxyz/Mask_RCNN.git
   ```
Install the required dependencies.

2. ```sh
   pip install -r requirements.txt
   ```

Run the setup from the root directory

3. ```sh
   python setup.py install
   ```


<!-- USAGE -->
## Usage


You are now ready to use Mask RCNN. You can follow the Matterport guides and tutorials if you want. If you would rather get right into detecting flowers of *Dryas* in images, use the script TrainAndDetect.py and one of the flower detection models you have downloaded. The script is used to initiate both training and detection.

To detect flowers in images in a folder, give the script the path to the root folder (the folder containing the folder with images that you want to process) and the name of folder with the images. Further, give the path to the weights-file you want to use.


To detect flowers in images, run
   ```sh
   python TrainAndDetect.py detect --dataset=path/to/root --subset=name_of_folder_with_images --weights=path/to/MRCNN_Dryas_Model1.h5
   ```

To train your own model, you need annotated images separated into a train and a val (validation) folder and the corresponding annotation files (train.json and val.json). You can use the <a href="https://www.robots.ox.ac.uk/~vgg/software/via/"><strong>VIA VGG</strong></a> annotation tool (we used version 2.0.5). With the --weights command you can tell the script to train as a finetuning of a model trained on the coco dataset or on another model.

Your folder structure should look like this:
parent  
-train  
--train images  
--train.json  
-val  
--val images  
--val.json  


To train a model on your own data, run
   ```sh
   python TrainAndDetect.py train --dataset=/path/to/parent --weights=coco
   ```


## Additional code supporting paper

In this repository we have also included the code for producing the results presented in the paper. 

### Calculate detection accuracy

The python script *calculateDetectionPerformance.py* returns a number of different metrics:

* INFO
    + Number of annotated objects
    + Number of predictions made
    + Correct positives
    + Matches nrow
    + False positives
    + False negatives
    + Mismatches
    + Images in the detections
    + Images in the ground truth

* SCORES
    + Precision
    + Recall
    + F1
    + MOTA
    + Mismatch ratio


### Produce figures

The R script produceFigures.R outputs the figures presented in the paper. The data folder in this repository contains the data input for this script.



<!-- CONTACT -->
## Contact

Are things not working? Reach out to me or take a look at the Mask-RCNN documentation and tutorials.

[@HjalteMann](https://twitter.com/@HjalteMann) - mann@bios.au.dk - http://tecology.xyz/

[https://github.com/TECOLOGYxyz/AutomaticFlowerPhenology](https://github.com/TECOLOGYxyz/TrackingFlowers)




<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/TECOLOGYxyz/repo.svg?style=for-the-badge
[contributors-url]: https://github.com/TECOLOGYxyz/AutomaticFlowerPhenology/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/TECOLOGYxyz/repo.svg?style=for-the-badge
[forks-url]: https://github.com/TECOLOGYxyz/AutomaticFlowerPhenology/network/members
[stars-shield]: https://img.shields.io/github/stars/TECOLOGYxyz/repo.svg?style=for-the-badge
[stars-url]: https://github.com/TECOLOGYxyz/AutomaticFlowerPhenology/stargazers
[issues-shield]: https://img.shields.io/github/issues/TECOLOGYxyz/repo.svg?style=for-the-badge
[issues-url]: https://github.com/TECOLOGYxyz/AutomaticFlowerPhenology/issues
[license-shield]: https://img.shields.io/github/license/TECOLOGYxyz/repo.svg?style=for-the-badge
[license-url]: https://github.com/TECOLOGYxyz/AutomaticFlowerPhenology/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/TECOLOGYxyz