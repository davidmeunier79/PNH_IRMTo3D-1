
def mask_auto_threshold(img_file, operation, index):
    import os
    import numpy as np
    import nibabel as nib
    import matplotlib.pyplot as plt
    from sklearn.cluster import KMeans
    from nipype.utils.filemanip import split_filename as split_f

    ## Mean function
    def calculate_mean(data):
        total = sum(data)
        count = len(data)
        mean = total / count
        return mean

    img_nii = nib.load(img_file)
    img_arr = np.array(img_nii.dataobj)

    # Reshape data to a 1D array (required by k-means)
    X = np.copy(img_arr).flatten().reshape(-1, 1)

    print("X shape : ", X.shape)

    # Create a k-means clustering model with 3 clusters
    # using k-means++ initialization

    num_clusters = 3
    kmeans = KMeans(n_clusters=num_clusters, random_state=0)

    # Fit the model to the data and predict cluster labels
    cluster_labels = kmeans.fit_predict(X)

    # Split data into groups based on cluster labels
    groups = [X[cluster_labels == i].flatten() for i in range(num_clusters)]

    avail_operations = ["min", "mean", "max"]

    assert operation in avail_operations, "Error, \
        {} is not in {}".format(operation, avail_operations)

    assert 0 <= index and index < num_clusters-1, "Error \
        with index {}".format(index)

    # We must define : the minimum of the second group for the headmask
    # we create minimums array, we sort and then take the middle value
    minimums_array = np.array([np.amin(group) for group in groups])
    min_sorted = np.sort(minimums_array)

    print("Min : {}".format(" ".join(str(val) for val in min_sorted)))

    # We must define :  mean of the second group for the skull extraction
    # we create means array, we sort and then take the middle value
    means_array = np.array([calculate_mean(group) for group in groups])
    mean_sorted = np.sort(means_array)

    index_sorted = np.argsort(means_array)

    print("Mean : {}".format(" ".join(str(int(val)) for val in mean_sorted)))

    print("Index = {}".format(" ".join(str(int(val)) for val in index_sorted)))

    print("Index mid group : ", index_sorted[index])
    print("Min/max mid group : ", np.amin(groups[index_sorted[index]]),
          np.amax(groups[index_sorted[index]]))

    0/0

    maximums_array = np.array([np.amax(group) for group in groups])
    max_sorted = np.sort(maximums_array)

    print("Max : {}".format(" ".join(str(val) for val in max_sorted)))

    if operation == "min":  # for head mask
        mask_threshold = min_sorted[index]
        print("headmask_threshold : ", mask_threshold)

    elif operation == "mean":  # for skull mask

        mask_threshold = mean_sorted[index]
        print("skull_extraction_threshold : ", mask_threshold)

    elif operation == "max":  # unused

        mask_threshold = max_sorted[index]
        print("max threshold : ", mask_threshold)

    return mask_threshold


def mask_auto_img(img_file, operation):

    import os
    import numpy as np
    import nibabel as nib
    import matplotlib.pyplot as plt

    from scipy.signal import find_peaks

    from nipype.utils.filemanip import split_filename as split_f

    img_nii = nib.load(img_file)
    img_arr = np.array(img_nii.dataobj)

    # Reshape data to a 1D array (required by k-means)
    X = np.copy(img_arr).flatten().reshape(-1, 1)

    print("X shape : ", X.shape)
    print("X max : ", np.round(np.max(X)))
    sample_bins = 30
    nb_bins = (np.rint(np.max(X)/sample_bins)).astype(int)
    print("Nb bins: ", nb_bins)

    # Create a histogram
    hist, bins, _ = plt.hist(X, bins=nb_bins,
                             alpha=0.5, color='b', label='Histogram')

    # Add labels and a legend
    plt.xlabel('Value')
    plt.ylabel('Probability')
    plt.title('Histogram')
    plt.legend()

    # Save the figure as a PNG file
    plt.savefig(os.path.abspath('histogram.png'))

    # Find local minima in the histogram
    peaks, indexes = find_peaks(-hist, distance = 10)  # Use negative histogram for minima

    assert peaks.shape[0] > 2, \
        "Error, could not find at least two local minima"

    print("peaks indexes :", peaks)

    print("peak_hist :", hist[peaks])
    print("peak_bins :", bins[peaks])

    index_peak_min = bins[peaks][0]
    index_peak_max = bins[peaks][1]

    # filtering
    new_mask_data = np.zeros(img_arr.shape, dtype=int)

    assert operation in ["higher", "interval", "lower"], \
        "Error in operation {}".format(operation)

    if operation == "interval":
        filter_arr = np.logical_and(index_peak_min < img_arr,
                                    img_arr < index_peak_max)

    elif operation == "higher":
        filter_arr = index_peak_min < img_arr

    new_mask_data[filter_arr] = 1

    print(np.sum(new_mask_data))

    # saving mask as nii
    path, fname, ext = split_f(img_file)

    mask_img_file = os.path.abspath(fname + "_autothresh" + ext)

    mask_img = nib.Nifti1Image(dataobj=new_mask_data,
                               header=img_nii.header,
                               affine=img_nii.affine)
    nib.save(mask_img, mask_img_file)

    return mask_img_file


def pad_zero_mri(img_file, pad_val=10):

    import os
    import nibabel as nib
    import numpy as np

    from nipype.utils.filemanip import split_filename as split_f

    img = nib.load(img_file)
    img_arr = np.array(img.dataobj)
    img_arr_copy = np.copy(img_arr)

    img_arr_copy_padded = np.pad(
        img_arr_copy,
        pad_width=[(pad_val, pad_val), (pad_val, pad_val), (pad_val, pad_val)],
        mode='constant',
        constant_values=[(0, 0), (0, 0), (0, 0)])

    img_padded = nib.Nifti1Image(img_arr_copy_padded,
                                 header=img.header,
                                 affine=img.affine)

    path, fname, ext = split_f(img_file)

    img_padded_file = os.path.abspath("padded_" + fname + ext)

    nib.save(img_padded, img_padded_file)

    return img_padded_file


def keep_gcc(nii_file):

    import os
    import nibabel as nib
    import numpy as np

    from nipype.utils.filemanip import split_filename as split_f

    def getLargestCC(segmentation):

        from skimage.measure import label

        labels = label(segmentation)
        assert labels.max() != 0  # assume at least 1 CC
        largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
        return largestCC

    # nibabel (nifti -> np.array)
    img = nib.load(nii_file)
    data = img.get_fdata().astype("int32")
    print(data.shape)

    # numpy
    data[data > 0] = 1
    binary = np.array(data, dtype="bool")

    # skimage

    # skimage GCC
    new_data = getLargestCC(binary)

    # nibabel (np.array -> nifti)
    new_img = nib.Nifti1Image(dataobj=new_data,
                              header=img.header,
                              affine=img.affine)

    path, fname, ext = split_f(nii_file)

    gcc_nii_file = os.path.abspath(fname + "_gcc" + ext)

    nib.save(new_img, gcc_nii_file)
    return gcc_nii_file



def wrap_nii2mesh_old(nii_file):

    import os
    from nipype.utils.filemanip import split_filename as split_f

    path, fname, ext = split_f(nii_file)

    stl_file = os.path.abspath(fname + ".stl")

    cmd = "nii2mesh_old_gcc {} {}".format(nii_file, stl_file)

    ret = os.system(cmd)

    print(ret)

    assert ret == 0, "Error, cmd {} did not work".format(cmd)
    return stl_file

def wrap_nii2mesh(nii_file):

    import os
    from nipype.utils.filemanip import split_filename as split_f

    path, fname, ext = split_f(nii_file)

    stl_file = os.path.abspath(fname + ".stl")

    cmd = "nii2mesh {} {}".format(nii_file, stl_file)

    ret = os.system(cmd)

    print(ret)

    assert ret == 0, "Error, cmd {} did not work".format(cmd)
    return stl_file

