#########################################
# Plotting from GMD
###################
###
# This package relies on heatmap.3 from the library GMD
# This code is a modified version of heatmap.3, some changes
# were required for our plotting.
# The heatmap.cnv function should be considered a modification
# of th GMD library function heatmap.3, all credit goes to
# their authors.
## Please note this code is from the library GMD
## All credit for this code goes to GMD's authors.
## I do not recommend using this version of the code, which
## has been poorly modified for our use but recommend using
## the official version from the package GMD
## https://cran.r-project.org/web/packages/GMD/index.html
## A copy of gtools::invalid
##
## see \code{invalid} in package:gtools for details
## Test if a value is missing, empty, or contains only NA or NULL values
## param: x value to be tested

import pandas as pd
import numpy as np
from matplotlib.path import Path
import matplotlib.patches as patches
from matplotlib.patches import Arc, Wedge


LW = 0.3


def polar2xy(r, theta):
    return np.array([r * np.cos(theta), r * np.sin(theta)])


def hex2rgb(c):
    return tuple(int(c[i : i + 2], 16) / 256.0 for i in (1, 3, 5))


def ChordArc(
    start1=0,
    end1=60,
    start2=180,
    end2=240,
    radius=1.0,
    chordwidth=0.7,
    ax=None,
    color=(1, 0, 0),
):
    # start, end should be in [0, 360)
    if start1 > end1:
        start1, end1 = end1, start1
    if start2 > end2:
        start2, end2 = end2, start2
    start1 *= np.pi / 180.0
    end1 *= np.pi / 180.0
    start2 *= np.pi / 180.0
    end2 *= np.pi / 180.0
    opt1 = 4.0 / 3.0 * np.tan((end1 - start1) / 4.0) * radius
    opt2 = 4.0 / 3.0 * np.tan((end2 - start2) / 4.0) * radius
    rchord = radius * (1 - chordwidth)
    verts = [
        polar2xy(radius, start1),
        polar2xy(radius, start1) + polar2xy(opt1, start1 + 0.5 * np.pi),
        polar2xy(radius, end1) + polar2xy(opt1, end1 - 0.5 * np.pi),
        polar2xy(radius, end1),
        polar2xy(rchord, end1),
        polar2xy(rchord, start2),
        polar2xy(radius, start2),
        polar2xy(radius, start2) + polar2xy(opt2, start2 + 0.5 * np.pi),
        polar2xy(radius, end2) + polar2xy(opt2, end2 - 0.5 * np.pi),
        polar2xy(radius, end2),
        polar2xy(rchord, end2),
        polar2xy(rchord, start1),
        polar2xy(radius, start1),
    ]

    codes = [
        Path.MOVETO,
        Path.CURVE4,
        Path.CURVE4,
        Path.CURVE4,
        Path.CURVE4,
        Path.CURVE4,
        Path.CURVE4,
        Path.CURVE4,
        Path.CURVE4,
        Path.CURVE4,
        Path.CURVE4,
        Path.CURVE4,
        Path.CURVE4,
    ]

    if ax == None:
        return verts, codes
    else:
        path = Path(verts, codes)
        patch = patches.PathPatch(
            path, facecolor=color + (0.5,), edgecolor=color + (0.4,), lw=LW
        )
        ax.add_patch(patch)


def selfChordArc(start=0, end=60, radius=1.0, chordwidth=0.7, ax=None, color=(1, 0, 0)):
    # start, end should be in [0, 360)
    if start > end:
        start, end = end, start
    start *= np.pi / 180.0
    end *= np.pi / 180.0
    opt = 4.0 / 3.0 * np.tan((end - start) / 4.0) * radius
    rchord = radius * (1 - chordwidth)
    verts = [
        polar2xy(radius, start),
        polar2xy(radius, start) + polar2xy(opt, start + 0.5 * np.pi),
        polar2xy(radius, end) + polar2xy(opt, end - 0.5 * np.pi),
        polar2xy(radius, end),
        polar2xy(rchord, end),
        polar2xy(rchord, start),
        polar2xy(radius, start),
    ]

    codes = [
        Path.MOVETO,
        Path.CURVE4,
        Path.CURVE4,
        Path.CURVE4,
        Path.CURVE4,
        Path.CURVE4,
        Path.CURVE4,
    ]

    if ax == None:
        return verts, codes
    else:
        path = Path(verts, codes)
        patch = patches.PathPatch(
            path, facecolor=color + (0.5,), edgecolor=color + (0.4,), lw=LW
        )
        ax.add_patch(patch)


def chordDiagram(X, ax, colors=None, width=0.1, pad=2, chordwidth=0.7, lim=1.1):
    """Plot a chord diagram
    Parameters
    ----------
    X :
        flux data, X[i, j] is the flux from i to j
    ax :
        matplotlib `axes` to show the plot
    colors : optional
        user defined colors in rgb format. Use function hex2rgb() to convert hex color to rgb color. Default: d3.js category10
    width : optional
        width/thickness of the ideogram arc
    pad : optional
        gap pad between two neighboring ideogram arcs, unit: degree, default: 2 degree
    chordwidth : optional
        position of the control points for the chords, controlling the shape of the chords
    """
    # X[i, j]:  i -> j
    x = X.sum(axis=1)  # sum over rows
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    diam = 1.8

    if colors is None:
        # use d3.js category10 https://github.com/d3/d3-3.x-api-reference/blob/master/Ordinal-Scales.md#category10
        colors = [
            "#1f77b4",
            "#ff7f0e",
            "#2ca02c",
            "#d62728",
            "#9467bd",
            "#8c564b",
            "#e377c2",
            "#7f7f7f",
            "#bcbd22",
            "#17becf",
        ]
        if len(x) > 10:
            print("x is too large! Use x smaller than 10")
    if type(colors[0]) == str:
        colors = [hex2rgb(colors[i]) for i in range(len(x))]

    # find position for each start and end
    y = x / np.sum(x).astype(float) * (360 - pad * len(x))

    pos = {}
    arc = []
    nodePos = []
    start = 0
    for i in range(len(x)):
        end = start + y[i]
        arc.append((start, end))
        angle = 0.5 * (start + end)
        # print(start, end, angle)
        if -30 <= angle <= 210:
            angle -= 90
        else:
            angle -= 270
        nodePos.append(
            tuple(
                polar2xy((diam / 2) + diam * 0.05, 0.5 * (start + end) * np.pi / 180.0)
            )
            + (angle,)
        )
        z = (X[i, :] / x[i].astype(float)) * (end - start)
        ids = np.argsort(z)
        z0 = start
        for j in ids:
            pos[(i, j)] = (z0, z0 + z[j])
            z0 += z[j]
        start = end + pad

    for i in range(len(x)):
        start, end = arc[i]
        # This draws the outter ring #
        # IdeogramArc(start=start, end=end, radius=1.0, ax=ax,
        #            color=colors[i], width=width)
        a = Arc((0, 0), diam, diam, angle=0, theta1=start, theta2=end, color=colors[i], lw=10)
        ax.add_patch(a)
        start, end = pos[(i, i)]
        # This draws the paths to itself #
        if end - start < 180:  # Indicates this method will work fine !
            selfChordArc(
                start,
                end,
                radius=1.0 - width,
                color=colors[i],
                chordwidth=chordwidth * 0.7,
                ax=ax,
            )
        else:  # Need to use a wedge because the arch distorts past 180-degrees
            path = Wedge(0, diam / 2, start, end, color=colors[i] + (0.5,))
            ax.add_patch(path)
        for j in range(i):
            if X[i, j] == 0 and X[j, i] == 0:  # don't draw anything for no interaction
                continue
            color = colors[i]
            if X[i, j] > X[j, i]:  # Color by the dominant signal #
                color = colors[j]
            start1, end1 = pos[(i, j)]
            start2, end2 = pos[(j, i)]
            ChordArc(
                start1,
                end1,
                start2,
                end2,
                radius=1.0 - width,
                color=color,
                chordwidth=chordwidth,
                ax=ax,
            )

    return nodePos