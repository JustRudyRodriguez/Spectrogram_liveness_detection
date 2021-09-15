import csv
import os
from PIL import Image
import numpy as np
# This script does not have any shuffle feature, as it is being designed to use the entire dataset. It would be useful for subsets though.

# I may want a way to load in sections at a time.

# one of my args, should be a pass by reference list() that contains title of files already used. Useful for creating validation set without dups.
def databuilder(folder, spread,x =150,y = 250,size = 100000, type="", environment="", srecorder ="", playback="", recorder=""):
    # size var is arbitrarily large.
    # likely gonna want to make this whole file a function. with folders as var.
    directory = "C:/Users/rudy_/PycharmProjects/pythonProject/Remasc/" + folder + "/"
    csvInput = "C:/Users/rudy_/PycharmProjects/pythonProject/Remasc/" + spread
    # gets list of files in directory.
    dirFileNames = os.listdir(directory)
    # opens relevant csv, places into list metalist.
    meta = csv.reader(open(csvInput))
    metaList = list()
    for row in meta:
        metaList.append(row)

    # Takes the input args into this array. Used to select custom datasets.
    # appropriate columns to compare to [1,3,5,6,7]
    metafilter = [type, environment, srecorder, playback, recorder]

    selected = []
    # count is for testing.
    count = 0
    # shape = Image.open(directory+metaList[0][0].strip())
    datalist = list()
    gtype = list()
    # dataset = np.empty_like((597,1182,4))
    # Compares to filter, and adds if it passes. Likely a cleaner way to do this, but this works well enough.
    for rows in metaList:
        # The first row has a stipulation to remove "1" values, because we don't need that. I may need to expand this functionality.
        if rows[1].strip() == metafilter[0] or metafilter[0] == "" and rows[1].strip() != "1":
            if rows[3].strip() == metafilter[1] or metafilter[1] == "":
                if rows[5].strip() == metafilter[2] or metafilter[2] == "":
                    if rows[6].strip() == metafilter[3] or metafilter[3] == "":
                        if rows[7].strip() == metafilter[4] or metafilter[4] == "":
                            count += 1
                            image = Image.open((directory+rows[0].strip()+".png"))
                            image = image.resize((x, y), 3)
                            # I may want to save these values as unsigned 8 bit numbers to save memory. Not sure if compatible.
                            num = np.asarray(image)
                            no_alpha = num[:, :, :3]
                            datalist.append(np.asarray(no_alpha))
                            # this formats the labels as 0/1
                            gtype.append(float(rows[1].strip())-2)
                            # This formats labels as -1/1
                            '''if int(row[1].strip()) == 2:
                                gtype.append(1.0)
                            else:
                                gtype.append(-1.0)'''
                            print('\r'+str(count),end='')
                            if count == size :
                                break

        continue


    dataset = np.array(datalist)
    print(dataset.shape)
    labels = np.array(gtype)
    # gets specified image
    # image = Image.open(testfile)
    # formats image into a numpy array, then removes alpha channel.
    # num = numpy.asarray(image)
    # image_without_alpha = num[:,:,:3]
    return dataset, labels