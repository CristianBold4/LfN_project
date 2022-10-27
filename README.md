# LfN_project

Repository of LFN course's project a.y. 2022/23

Collaborators: 
- Boldrin Cristian
- Makosa Alberto
- Mosco Simone

# 1st Idea: traffic forecasting using GCNNs

Goal:

Use the historical speed data to predict the speed at a future time step.

Datasets:

- METR_LA dataset available <a href="https://drive.google.com/drive/folders/10FOTa6HXPqX8Pf5WRoRwcFnW9BrNZEIX">here</a>, contains traffic information collected from loop detectors in the highway of Los Angeles County. 207 sensors were selected, and the dataset contains 4 months of data collected ranging from Mar 1st 2012 to Jun 30th 2012.
- BJER4 collected by Beijing Municipal Traffic Commission 
- PeMSD7 collected by California Deportment of Transportation <a href="https://pems.dot.ca.gov/">here</a>

Data model:

Each node represents a sensor station recording the traffic speed. An edge connecting two nodes means these two sensor stations are connected on the road. The geographic diagram representing traffic speed of a region changes over time.
