[![CC BY 4.0][cc-by-shield]][cc-by]

# A method to quantify deforming shapes using elliptic Fourier descriptors, dimensionality reduction and PCA
This is a simplified working example to perform an analysis of shapes as described in  
**Quantifying Dynamic Shapes in Soft Morphologies**  
Digumarti KM, Trimmer B, Conn AT, Rossiter J. Quantifying Dynamic Shapes in Soft Morphologies. Soft Robotics. 2019 Dec 1;6(6):733-44.  
https://www.liebertpub.com/doi/full/10.1089/soro.2018.0105  
I would appreciate it if you cite this in your work.

# How to use the code  
A movie of a morphing blob is provided to try out the analysis. See **movie.mp4**.  
First, use the **save_frames.py** script to extract frames from the video and save them locally.

The repository has three main files that you want to play with, in this order:
1. **describe_shapes.m** - reads a B/W image (or a set of images) and computes elliptic Fourier descriptors in terms of coefficients of harmonics
2. **pca.m** - performs dimensionality reduction, principal component analysis and gives an interactive way of looking at shapes as scores on principal components are changed
3. **draw_shapes.m** - a less interactive version of the above file

Code to compute the descriptors is in the rest of the files. You dont have to edit this.

Please do let me know if you find any bugs or have suggestions to improve. :beetle:

This work is licensed under a
[Creative Commons Attribution 4.0 International License][cc-by].

[![CC BY 4.0][cc-by-image]][cc-by]

[cc-by]: http://creativecommons.org/licenses/by/4.0/
[cc-by-image]: https://i.creativecommons.org/l/by/4.0/88x31.png
[cc-by-shield]: https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg
