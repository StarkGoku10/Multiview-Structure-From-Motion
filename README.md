# Structure-from-Motion (SfM)

## Overview

This repository contains a Structure from Motion(SfM) pipeline that resonctructs 3D point clouds from a set of continuous images of a scene. The project aims to demonstrates= the fundamentals of camera calibration, feature matching, feature matching, pose estimation, triangulation of 3D points and bundle adjustment(optional) to refine both 3D pointds and camera parameters and point cloud visualization which is a 3D representation of the captured scene.

**Key Highlights:**
- **Incremental SfM:** Takes pairs of images, estimates relative pose, and gradually adds more images to improve the reconstruction.
- **Feature Matching:** Uses SIFT + FLANN-based matching with ratio test for robust keypoint correspondences.
- **3D Reconstruction:** Triangulates matched points across multiple views to produce a 3D point cloud.
- **Bundle Adjustment (optional):** Refinement of camera poses and 3D structure via iterative least squares.
- **PLY Output:** Saves the final 3D point cloud (with color) in .ply format for visualization in tools like Open3D or MeshLab.

---

## Features

- **Camera Intrinsic Handling**  
  - Loads a custom `K.txt` file containing the 3×3 camera intrinsic matrix. 
  - Automatically scales intrinsics if downsampling the input images

- **Robust Feature Matching**  
  - SIFT descriptors (up to 10,000 features if needed).  
  - FLANN-based nearest neighbor search with ratio test to remove ambiguous matches(outliers).

- **Pose Estimation**  
  - Essential matrix estimation + `recoverPose` for the initial two images 
  - `solvePnPRansac` for subsequent images to localize new cameras.

- **Triangulation**  
  - Converts matched points into 3D coordinates with `cv2.triangulatePoints`.

- **Bundle Adjustment(Optional)**  
  - Minimizes reprojection error across cameras and 3D points for a more accurate reconstruction if enabled.

- **PLY Export**  
  - Saves final 3D points + color in ASCII or binary PLY format.  
  - Includes optional outlier removal and user-defined scaling factor.

---

## Methodology

The **Structure from Motion(SfM)** pipeline reconstructs a 3D representation of an object from a continuous series of continuos 2D images of an object from different angles by estimating camera poses and triangulating the matched feature points. Here is the stp-by-step implementation of the pipeline:

1. **Image & Intrinsics Loading**  
   - Read all images from a dataset directory capturing the object from different viewpoints.  
   - Read `K.txt` for the 3×3 camera intrinsics.  
   - Downsamples images and scales the intrinsic matrix accordingly.

2. **Feature Detection and Matching**  
   - **Feature Detection:** Use SIFT to detect and compute descriptors for the iamges. 
   - **Feature Matching:** Utilize a FLANN-based matcher to finds matches between consecutive images.  
   - **Lowe's Ratio Test:** filters ambiguous(outliers) matches.

3. **Initial Pose Estimation**  
   - **Essential Matrix estimation:** Compute essential matrix using th matched feature points and recover pose for the first two images.  
   - **Pose Recovery:** Recover fundamental camera motion (rotation, translation) under the pinhole camera model.

4. **Triangulation:**
  - **Project Matrices:** Construct projection matrices for each camera pose.
  - **3D points Triangulation:** Convert 2D matched points into 3D coordinates. Maintain a growing set of 3D points as more images are added.

5. **Incremental SfM**  
   - For each new image:
     - Match features to the previous image.  
     - `SolvePnPRansac` to get the new camera pose.  
     - Triangulate points between the new view and an existing view. 

6. **Bundle Adjustment(Optional)** 
   - If enabled, refine all camera parameters and 3D points by minimizing reprojection error.

7. **Point Cloud Export**  
   - Accumulate all 3D points plus colors from each iteration.  
   - Save to PLY with potential outlier removal, scaling, and color normalization.

---

## Installation

### Prerequisites
- **Python 3.7+**

  ```bash
  Install Additional Dependencies listed in the 'Requirements.txt' file.
  ```

### Setup Instructions

1. **Clone the Repository:**
  ```bash 
  git clone https://github.com/StarkGoku10/Multiview-Structure-From-Motion.git
  cd Multiview-Structure-From-Motion
  ```

2. **Create a Virtual Environment(Optional but recommended):**
  ```bash
  python3 -m venv venv
  source venv/bin/activate #on windows: venv\Scripts\activate
  ```

3. **Install Dependencies:**
  ```bash 
  pip install -r Requirements.txt
  ```

4. **Visualization(Optional)**
  - For visualizing the saved point cloud, python's `Open3D` library is used. 
  ```bash
  pip install open3d
  ```

5. **Important Note:**
- **Images:** 
    - Ensure there is sufficient overlap of the object/scene between consecutive images.
    - Capture images with different orientations and perspectives of the scene.

- **Calibration File**(`K.txt`)**:**
    - This file contains the camera intrinsic parameters(matrix). The format expected for this file is nine numerical values in a single line or representating 3x3 matrix seperated by spaces or new lines.
    - Ensure that the matrix is accurate and corresponds to the camera used to capture the images.
    - **Example `K.txt` matrix:**

        ```bash 
        2759.48 0 1520.69
        0 2764.16 1006.81
        0 0 1
        ```

---

## Usage

### Executing the SfM pipeline

Execute the `sfm.py` script to run the SfM pipeline. The script processes the first two images and triangulates the points between the two images. Then the pipeline adds one image sequentially to reconstruct the D structure of the scene. Optionally, **BUndle Adjustment** can be enabled to refine the 3D points and reduce the reprojection errors.

**Steps:**

1. **Navigate to the Project Directory**
  Ensure you are in the root directory where the 'SfM.py` file resides.
  ```bash
  cd Multiview-Structure-from_Motion
  ```

2. **Run the Pipeline**
  Ensure you have specified the proper directory paths for the dataset. Execute the following command: 
  ```bash
  python SfM.py
  ```
  **Note:** Adjust the dataset path as needed(in the `__main__` block):
    ```bash
    if __name__ == '__main__':
      sfm= StructurefromMotion("Datasets/YourDataset")
      sfm()
    ```
3. **Monitor the Output:**
  

