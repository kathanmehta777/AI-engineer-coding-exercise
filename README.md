# Remote-sensing AI engineer coding exercise

## Numpy, GDAL

1. Install numpy and gdal using conda or pip
1. Open a jupyter notebook session or any python session with access to installed numpy and gdal
1. Load data of all bands from ./data/ortho/swir_ortho_standardized.tif to a numpy array using gdal. The loaded numpy array should have a shape of (8, 653, 502)
   * Use `ds = gdal.Open(path)` to open a dataset
   * Use `data = ds.ReadAsArray()` to read data
   * 653 is number of rows, 502 is number of columns, 8 is number of bands. For each pixel, the data in 8 bands is a spectrum for that pixel.
1. Load the "facility spectrum" numpy array from ./data/facility_spectrum_standardized.npy using `np.load`
1. Calculate the cosine similarity of the spectra of all pixels with the given "facility spectrum" loaded in the previous step, and save the output as an array of shape (653, 502), matching the input image
1. Plot the cosine similarity using matplotlib
1. Use 0.995 as the threshold, create a mask array where pixels with cosine similarity > threshold are labeled with value 1, and other pixels labeled with value 0

## Image registration

1. Install OTB in a linux box https://www.orfeo-toolbox.org/CookBook/Installation.html#linux 
1. Orthorectify a WV3 image 
   * Instruction: https://www.orfeo-toolbox.org/CookBook/Applications/app_OrthoRectification.html
   * Inputs
     * WV3 image: ./data/pan/pan.tif
     * DEM data: ./data/dem
     * Interpolation: use "nn"
1. Wrap the orthorectification procedure in a python function
1. Write a test function for the new python function you created in the last step
1. Make sure to use version control (github/gitlab/bitbucket)


## ML
* Linear regression
* Dimension reduction
