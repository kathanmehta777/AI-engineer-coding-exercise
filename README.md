# Remote-sensing AI engineer coding exercise

You are expected to have a `python` environment with `numpy`, `scikit-learn` and optionally `jupyter` installed.

Please write your code in a public repository using a version control system (github/gitlab/bitbucket).

## Numpy, GDAL

1. Install gdal using conda/pip/...
1. Open a jupyter notebook session or any python session with access to installed gdal
1. Load data of all bands from [swir_ortho_standardized.tif](./data/ortho/swir_ortho_standardized.tif?raw=true) to a numpy array using gdal. The loaded numpy array should have a shape of (8, 653, 502)
   * Use `ds = gdal.Open(path)` to open a dataset
   * Use `data = ds.ReadAsArray()` to read data
   * 653 is number of rows, 502 is number of columns, 8 is number of bands. For each pixel, the data in the 8 bands forms a spectrum for that pixel.
1. Load the "facility spectrum" numpy array from [facility_spectrum_standardized.npy](./data/facility_spectrum_standardized.npy?raw=true) using `np.load`
1. Calculate the cosine similarity of the spectra of all pixels with the given "facility spectrum" loaded in the previous step using `np.dot`, and save the output as an array of shape (653, 502), matching the input image
1. Plot the cosine similarity using matplotlib and save the plot to an image
1. Use 0.995 as the threshold, create a mask array where pixels with `cosine_similarity > threshold` are labeled with value `1`, and other pixels labeled with value `0`. Plot the mask and save it to an image

## Linear regression
1. Load training data from [train_swir_nr.npy](./data/ml/train_swir_nr.npy?raw=true) (X) and [train_concentration.npy](./data/ml/train_concentration.npy?raw=true) (y). X has 28 features
1. Fit X to y
1. Predict y_test from test data [test_swir_nr.npy](./data/ml/test_swir_nr.npy?raw=true) (Xtest) and save it as a numpy array
1. Is there a way to reduce the number of features? Discuss and code.
1. Record your solution in a jupyter notebook or a python script

## Image registration

1. Install OTB in a linux box https://www.orfeo-toolbox.org/CookBook/Installation.html#linux 
1. Orthorectify a WV3 image 
   * Command: `otbcli_OrthoRectification`
   * Instruction: https://www.orfeo-toolbox.org/CookBook/Applications/app_OrthoRectification.html
   * Inputs
     * WV3 image: [pan.tif](./data/pan/pan.tif?raw=true)
     * DEM data: [dem data folder](./data/dem/)
     * Interpolation: use "nn"
   * Save the output file in your repository
1. Wrap the orthorectification procedure in a python function
1. Write a test function for the new python function you created in the last step
