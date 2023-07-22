from PIL import Image
import numpy as np
import json

def RGBconverter(image_title):
    # Import image
    im = Image.open(image_title)
    pix = im.load()

    # Constants
    image_size = im.size

    # Storage output
    RGBoutput = []

    for ii in range(image_size[0]):
        RGBoutput_row = []
        for jj in range(image_size[1]):
            print(pix[ii, jj])
            RGBoutput_row.append(pix[ii, jj])
        RGBoutput.append(RGBoutput_row)

    return RGBoutput

#%%
if __name__ == "__main__":
    image_title = 'gnss_error_map.png'
    RGBout = RGBconverter(image_title=image_title)

    # Serializing json
    json_object = json.dumps(RGBout)
    
    # Writing to sample.json
    with open("PixelToRGB.json", "w") as outfile:
        outfile.write(json_object)