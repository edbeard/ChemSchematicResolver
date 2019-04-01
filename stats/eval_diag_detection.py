"""

Script used to get stats on accuracy of plot extraction detection

:author : Ed Beard
"""

import os
import io
import csv
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from skimage import io as skio
from skimage import img_as_float
import copy

import chemschematicdiagramextractor as csde

# File paths
stats_dir = os.path.dirname(os.path.abspath(__file__))
csde_dev_dir = os.path.dirname(stats_dir)
train_dir = os.path.join(csde_dev_dir, 'train')

# Uv-vis RSc search paths
uvvis_sample_dir = os.path.join(train_dir, 'uv-vis_first_100')
uvvis_csv = os.path.join(stats_dir, 'uv-vis_first_100_diag_stats.csv')

#DyesAndPigments paths
dyes_and_pigments_sample_dir = os.path.join(train_dir, 'DyesAndPigments')
dyes_and_pigments_csv = os.path.join(stats_dir, 'DyesAndPigments_diag_stats.csv')


def get_diagram_stats(input_figs=dyes_and_pigments_sample_dir, output_csv=dyes_and_pigments_csv):
    """
    Estimates number of diagrams, and gets true values from user
    :return:
    """

    print('Evaluating CSDE test corpus')
    print('Press Control-C to exit at any time. Your progress will be saved.')

    print('Get sample image paths from: %s' % input_figs)
    img_paths = [os.path.join(input_figs, file) for file in os.listdir(input_figs)]

    if not os.path.isfile(output_csv):
        with io.open(output_csv, 'w') as initf:
            csvwriter = csv.writer(initf)
            csvwriter.writerow(['img_path', 'comp_diags', 'true_diags', 'comp_rgroups', 'true_rgroups'])

    print('Importing previous results from : %s' % output_csv)
    with io.open(output_csv, encoding='utf8') as outf:
        output_csvreader = csv.reader(outf)
        next(output_csvreader)  # Skip header
        output_rows = list(output_csvreader)

    print(img_paths, output_rows)
    extracted_paths = [row[0] for row in output_rows]

    new_imgs = [file for file in img_paths if file not in extracted_paths]

    for current_img in new_imgs:

        colours = iter(
            ['r', 'b', 'g', 'k', 'c', 'm', 'y', 'r', 'b', 'g', 'k', 'c', 'm', 'y', 'r', 'b', 'g', 'k', 'c', 'm', 'y'])

        try:

            # Calculating which are r_groups and regular diagrams
            reg_diags = 0
            rgroup_diags = 0

            # Read in float and raw pixel images
            fig = csde.io.imread(current_img)
            fig_copy = copy.deepcopy(fig)  # Unreferenced copy for display

            # Create unreferenced binary copy
            # TODO : Try this version and see how the general performance
            # bin_fig = copy.deepcopy(fig)
            # bin_fig = csde.actions.binarize(bin_fig, threshold=0.7)

            panels = csde.actions.segment(fig)

            labels, diags = csde.actions.classify_kmeans(panels)
            labels, diags = csde.actions.preprocessing(labels, diags, fig)


            # Assign labels to Diagrams
            labelled_diags = csde.actions.label_diags(diags, labels)

            # Create output image
            out_fig, ax = plt.subplots(figsize=(10, 6))
            ax.imshow(fig_copy.img)

            for diag in labelled_diags:
                colour = next(colours)

                diag_rect = mpatches.Rectangle((diag.left, diag.top), diag.width, diag.height,
                                               fill=False, edgecolor=colour, linewidth=2)
                ax.text(diag.left, diag.top + diag.height / 4, '[%s]' % diag.tag, size=diag.height / 20, color='r')
                ax.add_patch(diag_rect)

                label = diag.label
                label_rect = mpatches.Rectangle((label.left, label.top), label.width, label.height,
                                                fill=False, edgecolor=colour, linewidth=2)
                ax.text(label.left, label.top + label.height / 4, '[%s]' % label.tag, size=label.height / 5, color='r')
                ax.add_patch(label_rect)

                diag.label = csde.actions.read_label(fig_copy, label)
                diag = csde.r_group.detect_r_group(diag)

                if diag.label.r_group != [[]]:
                    rgroup_diags += 1
                else:
                    reg_diags += 1

        except Exception as e:
            reg_diags = 0
            rgroup_diags = 0



        ax.set_axis_off()
        plt.show()

        true_reg_diags = (input('# of detected normal diags : %s. How many normal diagrams? [#] (default = %s)' % (reg_diags, reg_diags)) or reg_diags)
        true_rgroup_diags =(input('# of detected rgroup diags : %s. How many rgroup diagrams? [#] (default = %s)' % (rgroup_diags, rgroup_diags)) or rgroup_diags)

        output_rows.append([current_img, reg_diags, true_reg_diags, rgroup_diags, true_rgroup_diags])

        # Saving after each update
        with io.open(output_csv, 'w') as outf:
            csvwriter = csv.writer(outf)
            csvwriter.writerow(['img_path', 'comp_diags', 'true_diags', 'comp_rgroups', 'true_rgroups'])
            for row in output_rows:
                csvwriter.writerow(row)

if __name__ == '__main__':
    get_diagram_stats()