# LfN_project

Repository of LFN course's project a.y. 2022/23

## Paper
https://github.com/CristianBold4/LfN_project/blob/main/LFN_Final_Report.pdf

Collaborators: 
- Boldrin Cristian
- Makosa Alberto
- Mosco Simone

# Traffic forecasting using GCNNs

Goal: use historical speed data to predict the speed at a given future time step.

Datasets:

1. METR-LA: [DCRNN author's Google Drive](https://drive.google.com/file/d/1pAGRfzMx6K9WWsfDcD1NMbIif0T0saFC/view?usp=sharing)
2. PEMSD4-BAY: [DCRNN author's Google Drive](https://drive.google.com/file/d/1wD-mHlqAb2mtHOe_68fZvDh1LpDegMMq/view?usp=sharing)
3. PeMSD7-LA: [STGCN author's GitHub repository](https://github.com/VeritasYin/STGCN_IJCAI-18/blob/master/data_loader/PeMS-M.zip)

Data model:

Each node represents a sensor station recording the traffic speed. An edge connecting two nodes means these two sensor stations are connected on the road. The geographic diagram representing traffic speed of a region changes over time.

# STGCN model
PyTorch implementation of slightly modified version of the paper *Spatio-Temporal Graph Convolutional Networks:
A Deep Learning Framework for Traffic Forecasting* (https://arxiv.org/abs/1709.04875)


## Requirements
To install requirements:
```console
pip3 install -r requirements.txt
```

## Run Program
```console
cd STGCN-model
python main.py [--args]
```
