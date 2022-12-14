# Notebook for building dataset

<ol>
  <li> get station metadata from <a href="https://pems.dot.ca.gov/?dnode=Clearinghouse&type=meta&district_id=7&submit=Submit">here</a>
  <li> first part of notebook: sample randomly a given number of sensors inside a given radius in meters and construct the weighted adjacency matrix, converting latitude and longitude in kilometers
  <li> second part of notebook (TO DO): retrieve sensors data measurements, query and match with the sampled sensor IDs in point (2). Build and array-shaped
  [n_measurements, n_sensors], where n_measurements contains the total number of slots aggregating in 5 minutes (e.g., 60 total days = 44 only workdays = 44 * 24 * 12 = 12.672
  measurements) containing for each cell the speed measured by sensor j-th at the time step i-th.
</ol>
