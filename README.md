# ANA_Project
## Required installations:
1)	conda create --name <name_of_env> python=3.5
	<br>conda create --name tf 
	<br>conda activate <name>
	<br>conda install -c anaconda pyqt
2)  Download Qt Creator
3)	pip install scikit-image
	pip install opencv-python

## Required Libraries:
math, sys, imutils, numpy, matplotlib

## Input images:
The images required for upload are present in the “Input images” folder.
<br>**Lenna_512.png:** Used for Image negative, Log Transformation, Gamma Transformation, Histogram Equalization, DFT, Image Reconstruction using IFFT, Histogram Shaping.
<br>**lenna_noise.jpg:** Used for median filter.
<br>**monalisa_noise.png:** Used for mean filter.
<br>**Cameraman_512.jpg:** Used for Low pass, High pass, Band pass, Unsharp masking filters.
<br>**cameraman_256.jpg:** Used for Laplacian Filter.
<br>**lenna_Interpolation.jpg (256X256):** Used for Bilinear and Bicubic Interpolation.

## Execution Commands:
Clone the project repository and execute the following command:
**python transformation.py**
