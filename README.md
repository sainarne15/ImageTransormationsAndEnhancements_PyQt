## Cloning the Repository
`git clone {project_url}`

## Requirements for setting up the environment
1) python 3.5 or higher
2) pip install PyQt5
3) pip install numpy
4) pip install ovencv-python
5) pip install imutils
6) pip install matplotlib
7) pip install scikit-image
	

## Executing the Project

1. python transformation.py
<br>![image](https://user-images.githubusercontent.com/60146983/236318113-0cae4c35-a7af-4f9d-ab61-5f3ceb10533c.png)
2. Click on the Upload button to Upload an image from the **input images** folder. The uploaded image appears on the image placeholder on the left.
3. Select an alorithm (displayed as radio buttons) to get the output.
4. The transformed image replaces the uploaded image and is displayed on the image placeholder on the left.
5. The Error Map is displayed on the image placeholder on the right.
6. SSIM is displayed on the bottom.
7. Clear the images using Clear button and upload the respective image for each algorithm.

## Input images:
The images required for upload are present in the “Input images” folder.
<br>**Lenna_512.png:** Used for Image negative, Log Transformation, Gamma Transformation, Histogram Equalization, DFT, Image Reconstruction using IFFT, Histogram Shaping.
<br>**lenna_noise.jpg:** Used for median filter.
<br>**monalisa_noise.png:** Used for mean filter.
<br>**Cameraman_512.jpg:** Used for Low pass, High pass, Band pass, Unsharp masking filters.
<br>**cameraman_256.jpg:** Used for Laplacian Filter.
<br>**lenna_Interpolation.jpg (256X256):** Used for Bilinear and Bicubic Interpolation.

## Outputs :
<br>Original Image:  Lenna_512.png   
<br>![image](https://user-images.githubusercontent.com/60146983/236314492-ab9bd7af-79ef-4877-b829-5c3a40a5c876.png)

Image Negative image with its error map and SSIM: 
![image](https://user-images.githubusercontent.com/60146983/236314945-060d898e-d383-4fbd-9731-b41d989b5dd4.png)

Log Transformed Image with its error map and SSIM: 
![image](https://user-images.githubusercontent.com/60146983/236315199-ea1e8183-a7c5-424e-8011-e0b61cde2d02.png)

Gamma Transformed Image with its error map and SSIM: Γ = 2.0
![image](https://user-images.githubusercontent.com/60146983/236315289-51e926c8-60f6-452a-952a-43163615b2c9.png)

                   
Histogram Equalization with its Error Map and SSIM:                         
![image](https://user-images.githubusercontent.com/60146983/236315402-a95ecca4-8f6e-442d-91c9-a008eab48adb.png)

Median Filter with its Error Map and SSIM:
<br>Input image for Median Filter:(lenna_noise.jpg)
<br>![image](https://user-images.githubusercontent.com/60146983/236315491-b5718d6e-6a58-41e6-9d88-383cb1788403.png)
<br>Output:
<br>![image](https://user-images.githubusercontent.com/60146983/236315541-943a88dc-cfb2-4fdc-9eee-776d71049ef8.png)

Mean Filter with its Error Map and SSIM: 
<br>Input: (Monalisa with salt and pepper noise: monalisa_noise.png)
<br>![image](https://user-images.githubusercontent.com/60146983/236315618-9162dcec-4103-454d-aa34-dbffda293381.png)
<br>Output:
<br>![image](https://user-images.githubusercontent.com/60146983/236315684-22696f5b-9d9b-4aa3-bc05-086065fd6962.png)

Input image for Low pass, High pass and Band pass filter: (Cameraman_512.jpg)
<br>![image](https://user-images.githubusercontent.com/60146983/236315775-5e3b3448-2850-4429-b630-e3ccb4bb76aa.png)
<br>Low Pass Filter with its Error Map and SSIM:
<br>![image](https://user-images.githubusercontent.com/60146983/236315842-a766f874-c0fc-4cb3-8509-79e97aa556fe.png)

High Pass Filter with its Error Map and SSIM:
<br>![image](https://user-images.githubusercontent.com/60146983/236315896-5c0e22f2-9fed-45cb-8a8b-ebaa6b5a8c2c.png)

Band Pass Filter with its Error Map and SSIM:
<br>![image](https://user-images.githubusercontent.com/60146983/236315975-a70f6932-968b-4c19-85b5-99c9a0a1f3b7.png)

Laplacian Filter with its Error Map and SSIM:
<br>Input image: cameraman_256.jpg
<br>Output:
<br>![image](https://user-images.githubusercontent.com/60146983/236316045-6266b8a3-29df-4957-b5d3-429259393508.png)

Unsharp Masking with its Error Map and SSIM:
<br>Input: Cameraman_512.jpg
<br>Output:
<br>![image](https://user-images.githubusercontent.com/60146983/236316117-ec6e991b-f21e-41fc-a7b8-ad8338554578.png)

Bilinear Interpolation:
<br>Input for Bilinear and Bicubic Interpolation: lenna_Interpolation.jpg (size 256*256)
<br>![image](https://user-images.githubusercontent.com/60146983/236316206-9888a062-8f11-41bf-9192-34bc1ef6093d.png)
<br>Output: 
<br>![image](https://user-images.githubusercontent.com/60146983/236316291-6c1355c2-fcf6-454c-94e1-ba1511638938.png)

Bicubic Interpolation: 
<br>Input: Lenna_Interpolation.jpg (size 256*256)
<br>Output:
<br>![image](https://user-images.githubusercontent.com/60146983/236316377-a16f775c-b5d1-4a1c-b870-27c81077d850.png)

DFT: 
<br>Input: Lenna_512.png  
<br>Output:
<br>![image](https://user-images.githubusercontent.com/60146983/236316467-d9173698-926e-4359-bd52-d71259bf044e.png)

Image Reconstruction Using IFFT:
<br>Input: Lenna_512.png  
<br>Output: We converted the input image to grayscale and then performed DFT on it. The output from the DFT is used as an input for the image reconstruction.
<br>![image](https://user-images.githubusercontent.com/60146983/236316541-ff32c667-3fc9-49e5-982d-262b3771ac76.png)

Histogram Shaping:
<br>![image](https://user-images.githubusercontent.com/60146983/236316661-89a14614-9b1c-4dc2-9942-cc2bc68480f1.png)

Click on Image negative:
<br>![image](https://user-images.githubusercontent.com/60146983/236316779-210dd468-88ec-4180-9d5c-181430fad837.png)
<br>Click on Image Shaping Radio button and upload Lenna_512.png as the target image:
<br>![image](https://user-images.githubusercontent.com/60146983/236316839-98dfc0c7-1777-408f-86d1-6dba3f8a3e99.png)
<br>Now click on Transform Image button:
<br>![image](https://user-images.githubusercontent.com/60146983/236316907-1caa32bd-371b-411a-a856-12d3bf954a62.png)

