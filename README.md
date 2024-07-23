# Tin-Toxicity-Analysis

The provided python scrcipt provides a method for determining the size of microbial colonies from an image of an agar plate. 

This code requires an image with the entire plate in frame for each desired time point and the size of the plate.

An annoted script is provided, but the script is used as follows:

1. The user selects a folder containing all images for analysis.
2. An image of the plate will appear. The outside of the plates should be selected in the first two clicks. It is important to ensure that the points are directly opposite sides of the plates. Many standard growth plates will have markings corresponding to opposite sides.
3. The user can now select each colony with a left-click. An annotation will appear, encircling the colony and showing the selected area. If the selection is not appearing correct, it may be required to change the sensitivity or the region-of-interest (ROI).
4. If an area is not appearing correctly, the user can right-click to undo the previous selection.
5. Once all colonies are selected, the user can press "q" to move to the next image.
6. This process can be repeated until all the images are selected.
