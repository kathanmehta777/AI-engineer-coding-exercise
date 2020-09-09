# Remote-sensing AI engineer coding exercise

You are expected to have a `python` environment with `numpy`, `matplotlib`, `scikit-learn` and optionally `jupyter` installed.

Please write your code in a private repository using a version control system (git). Make a tar ball or a zip file of your repository and submit it by email.

If you have questions regarding the following instructions, please [create an issue](https://github.com/Satelytics/AI-engineer-coding-exercise/issues/new/choose) or send us an email.

## Numpy, GDAL

### Part 1
1. Install gdal (version>=3) using conda/pip/...
1. Open a jupyter notebook session or any python session with access to installed gdal
1. Load data of all bands from [swir_ortho_standardized.tif](./data/ortho/swir_ortho_standardized.tif?raw=true) to a numpy array using gdal. The loaded numpy array should have a shape of (8, 653, 502). Hints:
   * Use `ds = gdal.Open(path)` to open a dataset
   * Use `data = ds.ReadAsArray()` to read data
   * 653 is number of rows, 502 is number of columns, 8 is number of bands. For each pixel, the data in the 8 bands forms a spectrum for that pixel.
1. Load the "facility spectrum" numpy array from [facility_spectrum_standardized.npy](./data/facility_spectrum_standardized.npy?raw=true) using `np.load`
1. Calculate the cosine similarity of the spectra of all pixels with the given "facility spectrum" loaded in the previous step using `np.dot`, and save the output as an array of shape (653, 502), matching the input image
1. Plot the cosine similarity using matplotlib and save the plot to an image
1. Use 0.995 as the threshold, create a mask array where pixels with `cosine_similarity > threshold` are labeled with value `1`, and other pixels labeled with value `0`. Plot the mask and save it to an image

### Part 2 
1. Instead of just one spectrum in facility_spectrum_standardized.npy, here we need to calculate cosine similarity of spectra for all pixels with 100 reference spectra saved in [ref_spectra_standardized.npy](./data/ref_spectra_standardized.npy?raw=true). The result should be a 653X502X100 numpy array. You don't need to submit the final result, but please submit your code. Does your code leverage numpy vectorized computation?

## Regression
1. Load training data from [train_swir_nr.npy](./data/ml/train_swir_nr.npy?raw=true) (X) and [train_concentration.npy](./data/ml/train_concentration.npy?raw=true) (y). X has 28 features
1. Fit X to y using a regression model. Please explain the reason behind your choice of regressor.
1. Make a plot of y_pred (predicted y values from your regression model) vs y
1. Predict y_test from test data [test_swir_nr.npy](./data/ml/test_swir_nr.npy?raw=true) (Xtest) and save it as a numpy array
1. Is there a way to reduce the number of features? Discuss and code.
1. Record your solution in a jupyter notebook or a python script

## Image orthorectification

* Data: [Satelytics_High_ONA_Sample_SO18013350-3-01_DS_PHR1A_201605121741085_FR1_PX_W097N35_0423_00596.zip](http://s3.amazonaws.com/satelyticsweb/download/Satelytics_High_ONA_Sample_SO18013350-3-01_DS_PHR1A_201605121741085_FR1_PX_W097N35_0423_00596.zip)
* Purpose: [orthorectify the image](https://www.satimagingcorp.com/services/orthorectification/)

Hints if using OTB for this exercise:

1. Install OTB (version>=6.6) in a linux box https://www.orfeo-toolbox.org/CookBook/Installation.html#linux 
1. Use the command `otbcli_OrthoRectification`
   * Instruction: https://www.orfeo-toolbox.org/CookBook/Applications/app_OrthoRectification.html
   * Input image in the zip file: 2986325101/IMG_PHR1A_P_001/DIM_PHR1A_P_201605121741085_SEN_2986325101-1.XML
   * Use the "skipcarto=true" trick
   * DEM: do not use DEM for this exercise
   * Interpolation option: use "nn"
   * Output a geotiff file
1. Plot the orthorectified image and save it
1. Wrap the orthorectification procedure in a python function
1. Write a test function for the new python function you created in the last step
