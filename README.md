# GOOD
GOOD: A global orthographic object descriptor for 3D object recognition and manipulation.
Authors: S. Hamidreza Kasaei, Ana Maria Tomé, Luís Seabra Lopes, Miguel Oliveira. 
[GOOD Paper Link](https://reader.elsevier.com/reader/sd/pii/S0167865516301684?token=47298EED31A885BE2417F56F9DA0BD84EA75A5EB93DB879F94EF0CF7139104E978B501B446AC4C17BE8CABDEE1A5EAE1&originRegion=us-east-1&originCreation=20220209234856).

This project implements the GOOD descriptor, used for concisely representing 3D object point clouds, in Python. 

# Dependencies

* [open3d](http://www.open3d.org)

# Usage

The good.py file is a compilation of functions. To get a 3D point cloud's descriptor use the following function call, where `cloud` is the 3D point cloud, `n` is the number of bins or how coarse (small `n`)/fine (large `n`) the descriptor is, `t` is a tunable threshold for creating a reproducable local reference frame (default is 0.015m).

```
GOOD(cloud, n, t)
```
