# Real-time Parking Slot Occupancy Detection
## Introduction
The idea of the project concerns parking occupancy detection. The central theme of this project is the creation of a model that is able to identify in real-time, using images/videos of a real car park, which spaces are free and which are occupied.
This repository contains two different parts:
* [notebook](notebook): this folder contains the main notebook [] which concerns the creation and training of the model to classify the occupancy of parking slots in real-time.
* [implementation](implementation): this folder contains the implementation of an Android mobile app for searching for available parking slots and a script that uses the previously trained model to communicate, in real-time, to all app users which slots are free and which are occupied using video coming from a real parking camera.
## Dataset
[CNRPark+EXT](http://cnrpark.it) is a dataset for visual occupancy detection of parking lots of roughly 150,000 labeled images (patches) of vacant and occupied parking spaces, built on a parking lot of 164 parking spaces. Is composed by images collected from November 2015 to February 2016 under various weather conditions by 9 cameras with different perspectives and angles of view. CNR-EXT captures different situations of light conditions, and it includes partial occlusion patterns due to obstacles (trees, street lamps, other cars) and partial or global shadowed cars.
## Model
## Integration of the classification model with the mobile app
## References
* [Parking Lot Occupancy Detection with Improved MobileNetV3](https://doi.org/10.3390/s23177642)
## Author
[Lorenzo Russo](https://github.com/lorenzoR21)
