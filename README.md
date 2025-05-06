Building Damage Prediction (Temporary Version)
This repository contains the temporary version of the codebase for the building damage prediction project.

Overview
The goal of this project is to predict the damage levels of buildings affected by various disasters (such as earthquakes, floods, hurricanes, wildfires, etc.) using machine learning and deep learning approaches.
Specifically, the project focuses on:

Extracting building-level data (images, shapes, and metadata) from satellite imagery and disaster datasets (e.g., xBD).

Preprocessing this data to create multimodal datasets that combine image features (e.g., ResNet), shape features (e.g., area, perimeter), and additional metadata.

Constructing graph-based representations of buildings and their spatial relationships.

Training and evaluating classification models (such as ResNet, GCN) to predict building damage levels.

Experimenting with advanced techniques, including focal loss and spatial graph construction per disaster type.

Notes
This is a temporary folder (tmp). The codebase will be reorganized and cleaned up into a well-structured directory in a later version.
