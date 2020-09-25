import subprocess


def process_image(str_input_path, str_output_path):
    str1 = 'otbcli_OrthoRectification -io.in "{}" -interpolator "nn" -io.out {}'.format(
        str_input_path, str_output_path)
    subprocess.call(str1, shell=True)


str_input_path = "./2986325101/IMG_PHR1A_P_001/DIM_PHR1A_P_201605121741085_SEN_2986325101-1.XML?&skipcarto=<(bool)true>"
str_output_path = "./satelytics.tif"

process_image(str_input_path, str_output_path)
