import cv2
import tkinter as tk
from tkinter import filedialog
import os
import pandas as pd
import numpy as np

# Global variables to store data for each colony and reference points
colony_areas = []
reference_points = []

#125-200
sensitivity = 145

#5-45
roi_size = 15

# Original locations of colony contours
original_colony_contours = []

# Variables to store ROI data
roi_x1_store = []
roi_y1_store = []

# Function to handle mouse clicks on the image
def get_colony_area(event, x, y, flags, param):
    global colony_areas, reference_points, original_colony_contours, selected_colony, image, new_image_OG, new_image, roi_x1, roi_y1, roi_x1_store, roi_y1_store

    if event == cv2.EVENT_LBUTTONDOWN:
        if len(reference_points) < 2:
            # Store the first two clicks as reference points
            reference_points.append((x, y))
            cv2.circle(new_image, (x, y), 3, (0, 0, 255), -1)  # Mark reference points in red

            if len(reference_points) == 2:
                # Calculate the distance between the two reference points (opposite sides of the plate)
                distance_pixels = cv2.norm(reference_points[0], reference_points[1])
        else:
            # Define a region of interest (ROI) around the clicked point
            #roi_size = 20  # Adjust this value to define the size of the ROI
            roi_x1 = max(0, x - roi_size)
            roi_y1 = max(0, y - roi_size)
            roi_x2 = min(new_image.shape[1], x + roi_size)
            roi_y2 = min(new_image.shape[0], y + roi_size)

            if roi_x1 < roi_x2 and roi_y1 < roi_y2:
                roi = new_image[roi_y1:roi_y2, roi_x1:roi_x2]
                gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

                # Perform image processing to identify the colony (e.g., thresholding, contour detection)
                _, binary_roi = cv2.threshold(gray_roi, sensitivity, 255, cv2.THRESH_BINARY)

                contours, _ = cv2.findContours(binary_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                if contours:
                    # Filter out small contours which could be noise
                    colony_contours = [contour for contour in contours if cv2.contourArea(contour) > 10]

                    if colony_contours:
                        colony_contour = max(colony_contours, key=cv2.contourArea)
                        colony_area_pixels = cv2.contourArea(colony_contour)

                        if len(reference_points) == 2:
                            # Calculate the distance between the two reference points (opposite sides of the plate)
                            distance_pixels = cv2.norm(reference_points[0], reference_points[1])
                            reference_distance_mm = 100.0  # Assuming 100 mm between the reference points

                            # Normalize the colony area based on the reference distance
                            normalized_colony_area = colony_area_pixels * (reference_distance_mm / distance_pixels)**2

                            # Store the normalized colony area and coordinates
                            colony_areas.append((x, y, normalized_colony_area))

                            # Store a copy of the original colony contour location
                            original_colony_contours.append(list(colony_contour))

                            # Draw a dot to mark the colony for visualization
                            colony_contour_shifted = [(pt[0][0] + roi_x1, pt[0][1] + roi_y1) for pt in colony_contour]
                            colony_contour_shifted = np.array(colony_contour_shifted, dtype=np.int32)
                            cv2.drawContours(new_image, [colony_contour_shifted], -1, (0, 0, 255), 2)

                            # Append the roi_x1 and roi_y1 to the lists
                            roi_x1_store.append(roi_x1)
                            roi_y1_store.append(roi_y1)

    # Left-click to select a colony for deletion
    elif event == cv2.EVENT_RBUTTONDOWN:
        if colony_areas:
            # Remove the last added colony data
            colony_areas.pop()
            original_colony_contours.pop()
            roi_x1_store.pop()
            roi_y1_store.pop()

            # Redraw the original image (without the deleted colony)
            new_image = new_image_OG.copy()

            # Draw the remaining colonies in their original locations
            for i in range(len(colony_areas)):
                x, y, _ = colony_areas[i]
                original_contour = original_colony_contours[i]

                colony_contour_shifted = [(pt[0][0] + roi_x1_store[i], pt[0][1] + roi_y1_store[i]) for pt in original_contour]
                colony_contour_shifted = np.array(colony_contour_shifted, dtype=np.int32)
                cv2.drawContours(new_image, [colony_contour_shifted], -1, (0, 0, 255), 2)

            print("Undid the last colony selection")

        # Display the current annotated image
        cv2.imshow("Image", new_image)


# Create a Tkinter window for selecting image files
root = tk.Tk()
root.withdraw()

# Ask the user to select a directory containing image files
directory_path = filedialog.askdirectory(title="Select a directory with image files")

if directory_path:
    images = [file for file in os.listdir(directory_path) if file.lower().endswith((".png", ".jpg", ".jpeg"))]

    for image_path in images:
        image = cv2.imread(os.path.join(directory_path, image_path))
        original_image = image.copy()
        display_image = original_image.copy()

        # Reset data for each image
        colony_areas = []
        reference_points = []
        selected_colony = None
        original_colony_contours = []
        roi_x1_store = []  # Initialize lists to store roi_x1 for each colony
        roi_y1_store = []  # Initialize lists to store roi_y1 for each colony
        

        cv2.namedWindow("Image")
        cv2.setMouseCallback("Image", get_colony_area)
        #Adjust image size
        size_factor = 1

        # Get screen dimensions
        height, width, _ = display_image.shape
        
        aspect=width/height

        new_height=int(height*size_factor)
        new_width=int(aspect*new_height)
        
        print("new_width:", new_width)
        print("new_height:", new_height)

        new_image = cv2.resize(display_image, (new_width, new_height))
        new_image_OG = cv2.resize(display_image, (new_width, new_height))


        while True:
            cv2.imshow("Image", new_image)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):  # Press "q" to exit the annotation of the current image
                break

        # Save the annotated image for each image
        image_name = os.path.splitext(image_path)[0]
        screenshot_filename = os.path.join(directory_path, f'{image_name}_annotated.png')
        cv2.imwrite(screenshot_filename, new_image)
        print(f"Annotated image saved as: {screenshot_filename}")

        # Save the annotated data to an Excel file for each image
        colony_data = pd.DataFrame(colony_areas, columns=["X", "Y", "Normalized Area"])
        output_excel_file = os.path.join(directory_path, f'{os.path.splitext(image_path)[0]}_colony_data.xlsx')
        colony_data.to_excel(output_excel_file, index=False)
        print(f"Colony data saved to {output_excel_file}")

else:
    print("No directory selected.")

# Explicitly close the OpenCV window
cv2.destroyAllWindows()
