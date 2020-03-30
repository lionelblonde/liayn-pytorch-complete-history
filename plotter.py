from collections import defaultdict
from copy import deepcopy
import glob
import argparse
import os
import hashlib
import time

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt  # noqa
import matplotlib.font_manager as fm  # noqa


parser = argparse.ArgumentParser(description="Plotter")
parser.add_argument('--font', type=str, default='Source Code Pro')
parser.add_argument('--dir', type=str, default=None, help='csv files location')
parser.add_argument('--xcolkey', type=str, default=None, help='name of the X column')
parser.add_argument('--ycolkey', type=str, default=None, help='name of the Y column')
parser.add_argument('--stdfrac', type=float, default=1., help='std envelope fraction')
args = parser.parse_args()


def plot(args):

    # Font (must be first)
    f1 = fm.FontProperties(fname='/Users/lionelblonde/Library/Fonts/Colfax-Light.otf', size=10)
    f2 = fm.FontProperties(fname='/Users/lionelblonde/Library/Fonts/Colfax-Light.otf', size=10)
    f3 = fm.FontProperties(fname='/Users/lionelblonde/Library/Fonts/Colfax-Light.otf', size=12)
    f4 = fm.FontProperties(fname='/Users/lionelblonde/Library/Fonts/Colfax-Medium.otf', size=14)
    # Create unique destination dir name
    hash_ = hashlib.sha1()
    hash_.update(str(time.time()).encode('utf-8'))
    dest_dir = "plots/batchplots_{}".format(hash_.hexdigest()[:20])
    os.makedirs(dest_dir, exist_ok=False)
    # Palette
    curves = [
        (39, 181, 234),
        (107, 64, 216),
        (239, 65, 70),
        (244, 172, 54),
        (104, 222, 122),
    ]
    palette = {
        'grid': (231, 234, 236),
        'face': (245, 249, 249),
        'axes': (200, 200, 208),
        'font': (108, 108, 126),
        'symbol': (64, 68, 82),
        'curves': curves,
    }
    for k, v in palette.items():
        if k != 'curves':
            palette[k] = tuple(float(e) / 255. for e in v)
    palette['curves'] = [tuple(float(e) / 255. for e in c) for c in v]
    # Figure color
    plt.rcParams['axes.facecolor'] = palette['face']
    # DPI
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['savefig.dpi'] = 300
    # X and Y axes
    plt.rcParams['axes.axisbelow'] = True
    plt.rcParams['axes.linewidth'] = 0.6
    # Lines
    plt.rcParams['lines.linewidth'] = 0.2
    plt.rcParams['lines.markersize'] = 1
    # Grid
    plt.rcParams['grid.linewidth'] = 0.6
    plt.rcParams['grid.linestyle'] = '-'

    # Dirs
    experiment_map = defaultdict(list)
    xcol_dump = defaultdict(list)
    ycol_dump = defaultdict(list)
    color_map = defaultdict(str)
    dirs = [d.split('/')[-1] for d in glob.glob("{}/*".format(args.dir))]
    print("pulling logs from sub-directories: {}".format(dirs))
    dirs.sort()
    dnames = deepcopy(dirs)
    dirs = ["{}/{}".format(args.dir, d) for d in dirs]
    print(dirs)
    # Colors
    colors = {d: palette['curves'][i] for i, d in enumerate(dirs)}

    for d in dirs:

        path = "{}/*/progress.csv".format(d)

        for fname in glob.glob(path):
            print("fname: {}".format(fname))
            # Extract the expriment name from the file's full path
            experiment_name = fname.split('/')[-2]
            # Remove what comes after the uuid
            key = experiment_name.split('.')[0] + "." + experiment_name.split('.')[1]
            env = experiment_name.split('.')[1]
            experiment_map[env].append(key)
            # Load data from the CSV file
            data = pd.read_csv(fname,
                               skipinitialspace=True,
                               usecols=[args.xcolkey, args.ycolkey])
            # Retrieve the desired columns from the data
            xcol = data[args.xcolkey].to_numpy()
            ycol = data[args.ycolkey].to_numpy()
            # Add the experiment's data to the dictionary
            xcol_dump[key].append(xcol)
            ycol_dump[key].append(ycol)
            # Add color
            color_map[key] = colors[d]

    for k, v in experiment_map.items():
        print(k, v)

    # Remove duplicate
    experiment_map = {k: list(set(v)) for k, v in experiment_map.items()}

    # Display summary of the extracted data
    assert len(xcol_dump.keys()) == len(ycol_dump.keys())  # then use X col arbitrarily
    print("summary -> {} different keys.".format(len(xcol_dump.keys())))
    for i, key in enumerate(xcol_dump.keys()):
        print(">>>> [key #{}] {} | #values: {}".format(i, key, len(xcol_dump[key])))

    print("\n>>>>>>>>>>>>>>>>>>>> Visualizing.")

    texts = deepcopy(dnames)
    texts.sort()
    texts = [text.split('__')[-1] for text in texts]
    print("Legend's texts (ordered): {}".format(texts))

    patches = [plt.plot([],
                        [],
                        marker="o",
                        ms=10,
                        ls="",
                        color=palette['curves'][i],
                        label="{:s}".format(texts[i]))[0]
               for i in range(len(texts))]

    # Calculate the x axis upper bound
    xmaxes = defaultdict(int)
    for env in experiment_map.keys():
        xmax = np.infty
        for i, key in enumerate(experiment_map[env]):
            if len(ycol_dump[key]) > 1:
                for col in ycol_dump[key]:
                    xmax = len(col) if xmax > len(col) else xmax
        xmaxes[env] = xmax
    # Get the maximum Y value accross all the experiments
    ymax = -np.infty

    # Plot mean and standard deviation
    for env in experiment_map.keys():

        xmax = deepcopy(xmaxes[env])

        # Create figure and subplot
        fig, ax = plt.subplots()
        # Create grid
        ax.grid(color=palette['grid'])
        # Only leave the left and bottom axes
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        # Set the color of the axes
        ax.spines['left'].set_color(palette['axes'])
        ax.spines['bottom'].set_color(palette['axes'])

        # Go over the experiments and plot for each on the same subplot
        for i, key in enumerate(experiment_map[env]):

            print(">>>> {}, in color RGB={}".format(key, color_map[key]))

            if len(ycol_dump[key]) > 1:
                # Calculate statistics to plot
                mean = np.mean(np.column_stack([col_[0:xmax] for col_ in ycol_dump[key]]), axis=-1)
                std = np.std(np.column_stack([col_[0:xmax] for col_ in ycol_dump[key]]), axis=-1)
                # Plot the computed statistics
                ax.plot(xcol_dump[key][0][0:xmax], mean, color=color_map[key])
                ax.fill_between(xcol_dump[key][0][0:xmax],
                                mean - (args.stdfrac * std),
                                mean + (args.stdfrac * std),
                                facecolor=(*color_map[key], 0.25))

                # Get the maximum Y value accross all the experiments
                _ymax = np.amax(mean + (args.stdfrac * std))
                ymax = max(_ymax, ymax)
            else:
                ax.plot(xcol_dump[key][0], ycol_dump[key][0])

        # Create the axes labels
        ax.tick_params(width=0.6, length=9, colors=palette['axes'], labelcolor=palette['font'])
        plt.xlabel("Timesteps", color=palette['font'], fontproperties=f3, labelpad=6)
        plt.ticklabel_format(axis='x', style='sci', scilimits=(-5, 5),
                             useOffset=(False), useMathText=True)
        ax.xaxis.offsetText.set_fontproperties(f1)
        ax.xaxis.offsetText.set_position((0.95, 0))
        for tick in ax.get_xticklabels():
            tick.set_fontproperties(f1)
        plt.ylabel("Episodic Return", color=palette['font'], fontproperties=f3, labelpad=12)
        for tick in ax.get_yticklabels():
            tick.set_fontproperties(f1)
        # Calculate the Y axis effective upper bound and truncate the axis at that value
        ylim = ax.get_yticks()[-1] if ymax > ax.get_yticks()[-2] else ax.get_yticks()[-2]
        ax.set_ylim(top=ylim)
        # Create title
        plt.title("{} agents".format(env), color=palette['font'], fontproperties=f4, pad=28)
        # Create legend
        legend = plt.legend(handles=patches, ncol=4, loc='lower left',
                            borderaxespad=0, facecolor='w', bbox_to_anchor=(0.0, 1.01))
        legend.get_frame().set_linewidth(0.0)
        for text in legend.get_texts():
            text.set_color(palette['font'])
            text.set_fontproperties(f2)

        fig.set_tight_layout(True)

        # Save figure to disk
        plt.savefig("{}/plot_{}_{}.pdf".format(dest_dir, env, "mean"),
                    format='pdf', bbox_inches='tight')
        print("mean plot done for env {}.".format(env))

    print(">>>>>>>>>>>>>>>>>>>> Bye.")


if __name__ == "__main__":
    # Plot
    plot(args)
