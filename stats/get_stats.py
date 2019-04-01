"""

Script used to get stats on all diagrams in a semi-automated way

:author : Ed Beard
"""

import os
import io
import csv
import matplotlib.pyplot as plt
from skimage import io as skio
from skimage import img_as_float

# File paths
stats_dir = os.path.dirname(os.path.abspath(__file__))
csde_dev_dir = os.path.dirname(stats_dir)
train_dir = os.path.join(csde_dev_dir, 'train')

# Uv-vis RSc search paths
uvvis_sample_dir = os.path.join(train_dir, 'uv-vis_first_100')
uvvis_csv = os.path.join(stats_dir, 'uv-vis_first_100.csv')

#DyesAndPigments paths
dyes_and_pigments_sample_dir = os.path.join(train_dir, 'DyesAndPigments')
dyes_and_pigments_csv = os.path.join(stats_dir, 'DyesAndPigments_imgs.csv')




def get_csd_stats(input_figs=uvvis_sample_dir, output_csv=uvvis_csv):
    """
    Obtain stats regarding the number of chemical schematic diagrams
    :return:
    """

    print('Evaluating CSDE test corpus')
    print('Press Control-C to exit at any time. Your progress will be saved.')

    print('Get sample image paths from: %s' % input_figs)
    img_paths = [os.path.join(input_figs, file) for file in os.listdir(input_figs)]

    if not os.path.isfile(output_csv):
        with io.open(output_csv, 'w') as initf:
            csvwriter = csv.writer(initf)
            csvwriter.writerow(['img_path', 'isCSD', 'isExtractable', 'isRgroup'])

    print('Importing previous results from : %s' % output_csv)
    with io.open(output_csv, encoding='utf8') as outf:
        output_csvreader = csv.reader(outf)
        next(output_csvreader)  # Skip header
        output_rows = list(output_csvreader)

    print(img_paths, output_rows)
    extracted_paths = [row[0] for row in output_rows]

    new_imgs = [file for file in img_paths if file not in extracted_paths]


    for img in new_imgs:

        path = img # Path to the fig

        img_test = skio.imread(path)
        img_test = img_as_float(img_test)
        # Generate overlay image with detected scale values
        plt.rcParams['image.cmap'] = 'gray'
        fig = plt.figure()
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(img_test)
        plt.show()
        plt.pause(0.001)

        isCSD = (input('Is this image a Chemical Schematic Diagram? [y/N]') or 'n')
        isExtractable = (input('Could this be extracted by a computer? [y/N]') or 'n')
        isRgroup = (input('Does this image contain general R-group structures? [y/N]') or 'n')

        output_rows.append([path, isCSD, isExtractable, isRgroup])

        # Saving after each update
        with io.open(output_csv, 'w') as outf:
            csvwriter = csv.writer(outf)
            csvwriter.writerow(['img_path', 'isCSD', 'isExtractable', 'isRgroup'])
            for row in output_rows:
                csvwriter.writerow(row)


def eval_diag_detection(input_figs=dyes_and_pigments_sample_dir, output_csv=dyes_and_pigments_csv):
    """
    Interface for evaluation CSDE performanc
    :return:
    """

    print('Evaluating CSDE test corpus')
    print('Press Control-C to exit at any time. Your progress will be saved.')

    print('Get sample image paths from: %s' % input_figs)
    img_paths = [os.path.join(input_figs, file) for file in os.listdir(input_figs)]

    if not os.path.isfile(output_csv):
        with io.open(output_csv, 'w') as initf:
            csvwriter = csv.writer(initf)
            csvwriter.writerow(['img_path', 'comp_diags', 'true_diags', 'comp_rgroups', 'true_rgroups' ])

def eval_diag_label_pairing():
    """
    Evaluate the
    :return:
    """
    pass



if __name__ == '__main__':
    get_csd_stats(dyes_and_pigments_sample_dir, dyes_and_pigments_csv)