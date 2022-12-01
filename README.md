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

# Mid term report ideas

$$
w_{ij} =
\begin{cases}
e^{\frac{-d_{ij}^2}{\sigma ^2}} \geq \epsilon \ \text{and} \ i \ != j \\ 
0 & \quad \text{otherwise}
\end{cases}
$$

where $d_{ij}$ is the distance from sensor i to sensor j, sigma and epsilon are normalization parameters to control the sparsity of the matrix.

We could compare this type of building matrix with a classical one.

Possible model's changes:
- insert a dropout probability (original = 0)
- change relu function to LSTM or other Gated Units in the Temporal Layers
- change the architecture of the model (order and numbers of spatio/temporal/normalization layers)
- change learning optimization: original use RMSprop, try with ADAM, AdaGrad

# Available datasets
- PeMSD{district}: we can select how many routes, how many sensors, which time window to include. Standard paper use 228 and 1026 stations (medium and large scale) randomly selected over the 39.000+ total sensor stations. Datas are aggregated into 5-minutes interval (288 data points per day) from 30-seconds data sample. They did experiments on 2012 workdays timeranges. We could make experiments and see how the number of sensor / features of sensors incide on the overall performance and time. (Model: STGCN). 
- MetrLA.h5: contains an array of shape [34272, 207], where 34272 is total number of time steps, and 207 is number of sensors. 288 data points per day -> data are collected over 4 months (when ?).
