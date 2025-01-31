import cv2
import numpy as np
from matplotlib import pyplot as plt
import os 
from scipy.optimize import least_squares
from tqdm import tqdm 

class ImageLoader:
    """
    Loads images from a specified directory and handles camera intrinsics downscaling.

    Attributes:
        img_dir (str): Path to the directory containing images and K.txt (camera intrinsics).
        K (np.ndarray): 3x3 camera intrinsic matrix (floating-point).
        image_list (list): List of valid image file paths found in img_dir.
        path (str): Current working directory path.
        factor (float): Downscale factor for images & intrinsics.

    Methods:
        downscale_image(image): Downscales the input image according to the factor.
        downscale_instrinsics(): Adjusts the intrinsic matrix to match the downscaled images.
    """
    def __init__(self, img_dir:str, downscale_factor:float):
        """
        Initializes the ImageLoader.

        Args:
            img_dir (str): Directory containing images + 'K.txt' file with intrinsics.
            downscale_factor (float): Factor to downscale images and their intrinsics.
        """
        # ensure directory path is absolute and consistent 
        self.img_dir= os.path.abspath(img_dir)

        #load camera intrinsic matrix from a text file named 'K.txt'
        k_file = os.path.join(self.img_dir, 'K.txt')
        with open(k_file, 'r') as f:
            lines= f.read().strip().split('\n')
            matrix_values=[]
            for line in lines:
                row_vals= [float(val) for val in line.strip().split()]
                matrix_values.append(row_vals)
            self.K =np.array(matrix_values, dtype=np.float32) #3x3 

        # collect image file paths
        self.image_list=[]
        for filename in sorted(os.listdir(self.img_dir)):
            if filename.lower().endswith(('.jpg', '.jpeg','.png')):
                self.image_list.append(os.path.join(self.img_dir,filename))

        # store the downscale factor and the current working directory 
        self.path = os.getcwd()
        self.factor = downscale_factor

        # adjust the intrinsic matrix for the downscaled image
        self.downscale_instrinsics()

    def downscale_image(self, image):
        """
        Downscales the given image by self.factor using single-step resizing.

        Args:
            image (np.ndarray): BGR image loaded via cv2.imread.

        Returns:
            np.ndarray: Downscaled image.
        """

        #single-step resize
        new_w= int(image.shape[1]/ self.factor)
        new_h= int(image.shape[0]/ self.factor)
        return cv2.resize(image,(new_w,new_h), interpolation=cv2.INTER_LINEAR)

        # for _ in range(1,int(self.factor / 2) + 1):
        #     image = cv2.pyrDown(image)
        # return image

    def downscale_instrinsics(self) -> None:
        """
        Adjusts the camera intrinsic parameters to match the downscaled image size.
        """
        self.K[0, 0] /= self.factor #fx
        self.K[1, 1] /= self.factor #fy
        self.K[0, 2] /= self.factor #cx
        self.K[1, 2] /= self.factor #cy

    
class StructurefromMotion:
    """
    Performs an incremental Structure-from-Motion pipeline on a set of images.

    This includes:
        - Loading and downscaling images
        - Feature detection & matching
        - Estimating camera poses (Essential matrix, recoverPose, solvePnP)
        - Triangulation to obtain 3D points
        - (Optionally) Bundle Adjustment
        - Saving the final 3D point cloud as .ply

    Attributes:
        img_obj (ImageLoader): Instance responsible for loading/downscaling images & intrinsics.

    Methods:
        feature_matching(image_0, image_1): Detect & match features across two images.
        triangulation(proj_matrix_1, proj_matrix_2, pts_2d_1, pts_2d_2): Triangulates 3D points.
        solve_PnP(...): Estimates camera pose from 3D-2D correspondences via RANSAC.
        find_common_points(...): Finds common feature points across overlapping sets.
        reproj_error(...): Computes reprojection error of 3D→2D projection.
        optimize_reproj_error(...): Cost function for bundle adjustment.
        compute_bundle_adjustment(...): Runs a least-squares optimizer to refine camera & point parameters.
        save_to_ply(...): Saves the accumulated 3D points & color information to a PLY file.
        __call__(bundle_adjustment_enabled): Main pipeline entry point.
    """

    def __init__(self, img_dir=str, downscale_factor:float = 2.0):
        """
        Initializes the StructurefromMotion pipeline.

        Args:
            img_dir (str): Directory containing images & K.txt intrinsics.
            downscale_factor (float): Factor used to downscale images & intrinsics.
        """
        # Create an ImageLoader instance
        self.img_obj =ImageLoader(img_dir, downscale_factor)

    def feature_matching(self, image_0, image_1) -> tuple:
        """
        Detects and matches SIFT keypoints between two images using FLANN-based matcher.

        Args:
            image_0 (np.ndarray): First input BGR image.
            image_1 (np.ndarray): Second input BGR image.

        Returns:
            (np.ndarray, np.ndarray): Two Nx2 arrays of matched keypoints (pts0, pts1).
        """
        # Initialize SIFT with n-features
        sift = cv2.SIFT_create(nfeatures=10000) 

        # Detect and compute SIFT keypoints and descriptors
        key_points0, descriptors_0 = sift.detectAndCompute(cv2.cvtColor(image_0, cv2.COLOR_BGR2GRAY), None)
        key_points1, descriptors_1 = sift.detectAndCompute(cv2.cvtColor(image_1, cv2.COLOR_BGR2GRAY), None)

        # FLANN parameters
        index_params= dict(algorithm=1, trees=15)
        search_params=dict(checks=200)

        # create the FlannBasedMatcher
        flann = cv2.FlannBasedMatcher(index_params,search_params)
        matches = flann.knnMatch(descriptors_0, descriptors_1, k=2)

        # Lowe's ratio test
        ratio_thresh=0.70
        good_matches = []
        for m, n in matches:
            if m.distance < ratio_thresh * n.distance:
                good_matches.append(m)

        # extract the matched keypoint coordinates
        pts0=np.float32([key_points0[m.queryIdx].pt for m in good_matches]) 
        pts1=np.float32([key_points1[m.trainIdx].pt for m in good_matches])

        return pts0, pts1
    
    def triangulation(self,proj_matrix_1, proj_matrix_2, pts_2d_1, pts_2d_2) -> tuple:
        """
        Triangulates 3D points from matched 2D points across two projection matrices.

        Args:
            proj_matrix_1 (np.ndarray): 3x4 projection matrix for view 1.
            proj_matrix_2 (np.ndarray): 3x4 projection matrix for view 2.
            pts_2d_1 (np.ndarray): Nx2 array of matched keypoints in the first image.
            pts_2d_2 (np.ndarray): Nx2 array of matched keypoints in the second image.

        Returns:
            (np.ndarray, np.ndarray, np.ndarray):
                - pts_2d_1.T (2, N) version of first image points
                - pts_2d_2.T (2, N) version of second image points
                - Homogeneous 3D points (4, N) scaled so last row is 1.
        """
        # Triangulate points using projection matrices
        point_cloud = cv2.triangulatePoints(proj_matrix_1, proj_matrix_2, pts_2d_1.T, pts_2d_2.T)
        return pts_2d_1.T, pts_2d_2.T, (point_cloud/point_cloud[3])
    
    def solve_PnP(self, obj_point, image_point, K, dist_coeff, rot_vector, initial) -> tuple:
        """
        Solves a Perspective-n-Point problem (PnP) via RANSAC to get camera pose.

        Args:
            obj_point (np.ndarray): Nx(1)x3 or Nx3 array of 3D points.
            image_point (np.ndarray): Nx(1)x2 or Nx2 array of 2D points.
            K (np.ndarray): 3x3 camera intrinsic matrix.
            dist_coeff (np.ndarray): Distortion coefficients array.
            rot_vector (np.ndarray): Some initialization or leftover from pipeline.
            initial (int): If 1, modifies shapes and transposes arrays.

        Returns:
            (np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray):
                - rot_matrix: 3x3 rotation matrix
                - tran_vector: 3x1 translation
                - image_point: Possibly subset of 2D inliers
                - obj_point: Possibly subset of 3D inliers
                - rot_vector: Possibly subset of rod vector?
        """
        # If initial == 1, reshape/transpose the input data
        if initial == 1:
            obj_point=obj_point[:,0,:]
            image_point = image_point.T
            rot_vector = rot_vector.T

        # Solve PnP with RANSAC to estimate camera pose    
        _, rot_vector_calc, tran_vector, inlier = cv2.solvePnPRansac(obj_point, image_point, K, dist_coeff, cv2.SOLVEPNP_ITERATIVE)
        rot_matrix, _ =cv2.Rodrigues(rot_vector_calc)

        # Filter inliers if available
        if inlier is not None:
            image_point=image_point[inlier[:,0]]
            obj_point=obj_point[inlier[:,0]]
            rot_vector = rot_vector[inlier[:,0]]
        return rot_matrix, tran_vector,image_point,obj_point,rot_vector
    
    def find_common_points(self, image_points_1, image_points_2, image_points_3) -> tuple:
        """
        Finds common points among overlapping sets of points.

        This is used to figure out which points are repeated (common) vs. new
        across consecutive or matching frames.

        Args:
            image_points_1 (np.ndarray): Nx2 points from previous matches
            image_points_2 (np.ndarray): Mx2 points from new matches
            image_points_3 (np.ndarray): Mx2 points from the new image

        Returns:
            (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
                - cm_points_1: Indices of common points in image_points_1
                - cm_points_2: Indices of common points in image_points_2
                - mask_array_1: Filtered array from image_points_2
                - mask_array_2: Filtered array from image_points_3
        """

        cm_points_1= []
        cm_points_2= []
        # Find common points between first and second set of image points
        for i in range(image_points_1.shape[0]):
            a= np.where(image_points_2 == image_points_1[i, :])
            if a[0].size !=0:
                cm_points_1.append(i)
                cm_points_2.append(a[0][0])

        # Mask arrays to remove common points
        mask_array_1 = np.ma.array(image_points_2, mask= False)
        mask_array_1.mask[cm_points_2] = True
        mask_array_1 = mask_array_1.compressed()
        mask_array_1=mask_array_1.reshape(int(mask_array_1.shape[0]/2),2)

        mask_array_2 = np.ma.array(image_points_3, mask=False)
        mask_array_2.mask[cm_points_2] = True
        mask_array_2 = mask_array_2.compressed()
        mask_array_2 = mask_array_2.reshape(int(mask_array_2.shape[0] / 2), 2)
        print(" Shape of New Array", mask_array_1.shape, mask_array_2.shape)
        return np.array(cm_points_1), np.array(cm_points_2), mask_array_1, mask_array_2

    def reproj_error(self, obj_points, image_points, transform_matrix, K, homogenity) -> tuple:
        """
        Computes the reprojection error for given 3D-2D correspondences and a camera transform.

        Args:
            obj_points (np.ndarray): Nx4 (homogeneous) or Nx3 points in 3D.
            image_points (np.ndarray): Nx2 array of 2D points.
            transform_matrix (np.ndarray): 3x4 extrinsic matrix [R | t].
            K (np.ndarray): 3x3 camera intrinsic matrix.
            homogenity (int): If 1, interpret obj_points as homogeneous.

        Returns:
            (float, np.ndarray):
                - total_error / len(image_points_calc): The average reprojection error.
                - obj_points: Possibly converted from homogeneous to Nx3 if homogenity == 1.
        """

        # Extract R, t from transform
        rot_matrix = transform_matrix[:3,:3]
        tran_vector = transform_matrix[:3, 3]
        rot_vector, _ = cv2.Rodrigues(rot_matrix)

        # Convert from homogeneous if needed
        if homogenity == 1:
            obj_points= cv2.convertPointsFromHomogeneous(obj_points.T)

        # Project 3D points back to 2D    
        image_points_calc, _ = cv2.projectPoints(obj_points, rot_vector, tran_vector, K, None)
        image_points_calc = np.float32(image_points_calc[:,0,:])

        # Calculate average L2 reprojection error
        total_error = cv2.norm(image_points_calc, np.float32(image_points.T) if homogenity == 1 else np.float32(image_points), cv2.NORM_L2)
        return total_error/ len(image_points_calc), obj_points
            
    def optimize_reproj_error(self, obj_points) -> np.array:
        """
        Cost function for bundle adjustment used by least_squares.

        Args:
            obj_points (np.ndarray): Flattened array of parameters:
                [3x4 transform_matrix, 3x3 K, 2D points, 3D points, etc.]

        Returns:
            (np.ndarray): Flattened reprojection error array for each point.
        """

        transform_matrix = obj_points[0:12].reshape((3,4)) #parse transform_matrix
        K = obj_points[12:21].reshape((3,3)) # parse K
        rest= int(len(obj_points[21:])* 0.4) # prase 2D subset
        p = obj_points[21:21 + rest].reshape((2, int(rest/2))).T
        obj_points = obj_points[21 + rest:].reshape((int(len(obj_points[21 + rest:])/3),3)) # parse 3D points

        rot_matrix = transform_matrix[:3,:3]
        tran_vector = transform_matrix[:3, 3]
        rot_vector , _ = cv2.Rodrigues(rot_matrix)

        #  project points
        image_points ,_ =cv2.projectPoints(obj_points, rot_vector, tran_vector, K, None)
        image_points = image_points[:,0,:]
        
        # compute squared error
        error = [(p[idx]- image_points[idx])**2 for idx in range(len(p))]
        return np.array(error).ravel()/len(p)
    
    def compute_bundle_adjustment(self, _3d_point, opt, transform_matrix_new, K, r_error) -> tuple:
        """
        Runs a least_squares optimizer to refine camera extrinsics, intrinsics, and 3D points.

        Args:
            _3d_point (np.ndarray): Nx3 array of 3D points.
            opt (np.ndarray): Nx2 array of 2D points.
            transform_matrix_new (np.ndarray): 3x4 extrinsic matrix to be refined.
            K (np.ndarray): 3x3 camera intrinsics to be refined.
            r_error (float): Convergence tolerance for least_squares.

        Returns:
            (np.ndarray, np.ndarray, np.ndarray):
                - Refined 3D points (Nx3).
                - Refined 2D points (2xN).
                - Refined extrinsic matrix (3x4).
        """

        # Concatenate all optimization variables
        opt_variables = np.hstack((transform_matrix_new.ravel(), K.ravel()))
        opt_variables= np.hstack((opt_variables, opt.ravel()))
        opt_variables= np.hstack((opt_variables, _3d_point.ravel()))

        # Perform least squares optimization
        values_corrected = least_squares(self.optimize_reproj_error, opt_variables, gtol= r_error).x
        
        K = values_corrected[12:21].reshape((3,3))
        rest = int(len(values_corrected[21:])* 0.4)
        return (values_corrected[21+rest:].reshape((int(len(values_corrected[21+rest:])/3),3)), 
                values_corrected[21:21 + rest].reshape((2, int(rest/2))).T, 
                values_corrected[0:12].reshape((3,4)))

    def save_to_ply(self, path, point_cloud, colors=None, bundle_adjustment_enabled=False,
                     binary_format=False, scaling_factor=1.0):
        
        """
        Saves the reconstructed 3D point cloud (and optional colors) to a PLY file.

        1) Creates a directory: 'Results' or 'Results with Bundle Adjustment'
        2) Normalizes & optionally outlier-removes the point cloud
        3) Writes ASCII or binary PLY depending on 'binary_format'.

        Args:
            path (str): Base path (e.g., self.img_obj.path).
            point_cloud (np.ndarray): Nx3 array of 3D points.
            colors (np.ndarray, optional): Nx3 color array (BGR or RGB) for each point.
            bundle_adjustment_enabled (bool): If True, saves under 'Results with Bundle Adjustment'.
            binary_format (bool): If True, writes PLY in binary_little_endian format.
            scaling_factor (float): Additional uniform scaling to apply to the point cloud.
        """

        # Create output directory
        sub_dir = 'Results with Bundle Adjustment' if bundle_adjustment_enabled else 'Results'
        output_dir = os.path.join(path, sub_dir)
        os.makedirs(output_dir, exist_ok=True)

        # Generate PLY filename based on dataset
        dataset_name = os.path.basename(os.path.normpath(self.img_obj.img_dir))
        ply_filename = os.path.join(output_dir, f"{dataset_name}.ply")

        # validate input arrays & scale if needed
        point_cloud= np.asarray(point_cloud).reshape(-1,3) * scaling_factor

        # If colors not provided, fill with some default
        if colors is not None:
            colors=np.asarray(colors).reshape(-1,3)
            colors=np.clip(colors,0,255).astype(np.uint8) # ensure valid RGB range
        else:
            colors = np.full_like(point_cloud, fill_value=105, dype=np.uint8) # default to white

        # Normalize the point cloud 
        mean= np.mean(point_cloud,axis=0)
        point_cloud-= mean # center the cloud
        scale_factor= np.max(np.linalg.norm(point_cloud,axis=1))
        point_cloud/=scale_factor 

        # optional outlier removal (eg, Z-score)
        distances =np.linalg.norm(point_cloud,axis=1)
        z_scores= (distances - np.mean(distances))/ np.std(distances)
        mask =np.abs(z_scores) < 2.5 # Remove the points with Z-scores > 2.5
        point_cloud =point_cloud[mask]
        point_cloud =point_cloud * scale_factor
        colors = colors[mask]

        # combine points and colors
        vertices= np.hstack([point_cloud,colors])

        # write the PLY file
        with open(ply_filename, 'wb' if binary_format else 'w') as f:
            # Write PLY header
            f.write(b'ply\n' if binary_format else 'ply\n')
            f.write(b'format binary_little_endian 1.0\n' if binary_format else 'format ascii 1.0\n')
            f.write(f'element vertex {len(vertices)}\n'.encode())
            f.write(b'property float x\nproperty float y\nproperty float z\n')
            f.write(b'property uchar red\nproperty uchar green\nproperty uchar blue\n')
            f.write(b'end_header\n')

            # Write vertices
            if binary_format:
                vertices_binary = np.zeros((len(vertices),), dtype=[
                    ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                    ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')
                ])
                vertices_binary['x'] = vertices[:, 0]
                vertices_binary['y'] = vertices[:, 1]
                vertices_binary['z'] = vertices[:, 2]
                vertices_binary['red'] = vertices[:, 3]
                vertices_binary['green'] = vertices[:, 4]
                vertices_binary['blue'] = vertices[:, 5]
                vertices_binary.tofile(f)
            else:
                # Write ASCII PLY
                np.savetxt(f, vertices, fmt='%f %f %f %d %d %d')

            print(f'Point cloud saved to {ply_filename}')

    def __call__(self, bundle_adjustment_enabled: bool = False):
        """
        The main entry point to run the entire SfM pipeline.

        1) Loads and downscales the first two images.
        2) Matches features & estimates initial pose using recoverPose.
        3) Incrementally adds images, solves PnP, triangulates new points.
        4) (Optionally) runs bundle adjustment.
        5) Saves final point cloud as .ply.

        Args:
            bundle_adjustment_enabled (bool): If True, runs compute_bundle_adjustment for each new view.
        """

        # cv2.namedWindow('image', cv2.WINDOW_NORMAL)

        # Pose array: store intrinsics + extrinsics in a linear stack
        pose_array = self.img_obj.K.ravel()
        transform_matrix_0 = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0]])
        transform_matrix_1 = np.empty((3,4))
        print('Camera Intrinsic Matrix:', self.img_obj.K)

        # Initial pose (camera extrinsic matrix)
        pose_0= np.matmul(self.img_obj.K, transform_matrix_0)
        pose_1 = np.empty((3,4))
        total_points= np.zeros((1,3))
        total_colors = np.zeros((1,3))

        # Load and downscale the first two images
        image_0 = self.img_obj.downscale_image(cv2.imread(self.img_obj.image_list[0]))
        image_1 = self.img_obj.downscale_image(cv2.imread(self.img_obj.image_list[1]))

        # Feature matching between the first two images
        features_0, features_1 = self.feature_matching(image_0, image_1)

        # Compute essential matrix and recover pose
        essential_matrix, em_mask = cv2.findEssentialMat(features_0, features_1, self.img_obj.K, method=cv2.RANSAC, prob=0.999, threshold=0.4, mask=None)
        features_0 = features_0[em_mask.ravel()==1]
        features_1 = features_1[em_mask.ravel()==1]

        _, rot_matrix, tran_matrix , em_mask = cv2.recoverPose(essential_matrix, features_0, features_1, self.img_obj.K)
        features_0 = features_0[em_mask.ravel()>0]
        features_1 = features_1[em_mask.ravel() > 0] 

        # Compose initial extrinsic: transform_matrix_1 = R0 * transform_matrix_0 + t
        transform_matrix_1[:3, :3]= np.matmul(rot_matrix, transform_matrix_0[:3,:3])
        transform_matrix_1[:3,3]= transform_matrix_0[:3, 3] + np.matmul(transform_matrix_0[:3,:3], tran_matrix.ravel())

        # Pose_1 = K * transform_matrix_1
        pose_1 = np.matmul(self.img_obj.K, transform_matrix_1)

        # Triangulate points between the first two images
        features_0, features_1, points_3d = self.triangulation(pose_0, pose_1, features_0, features_1)

         # Reproj error for the first two images
        error, points_3d= self.reproj_error(points_3d, features_1, transform_matrix_1, self.img_obj.K, homogenity=1)
        print("Reprojection error for first two images:", error)

        # SolvePnP for these initial points
        _,_, features_1, points_3d, _ = self.solve_PnP(points_3d, features_1, self.img_obj.K,
                                                       np.zeros((5,1), dtype=np.float32), features_0, initial=1)

        # Number of subsequent images after the first two
        total_images = len(self.img_obj.image_list) -2
        print('total_images', total_images)

        # Keep track of poses
        pose_array = np.hstack((np.hstack((pose_array, pose_0.ravel())), pose_1.ravel()))

        # RANSAC threshold for BA calls
        threshold = 0.75

        # after the first two images, start adding a single image to the group and repeat till last image is added.
        for i in tqdm(range(total_images)):
            # Load next image
            image_2 = self.img_obj.downscale_image(cv2.imread(self.img_obj.image_list[i+2]))

            # Feature matching
            features_cur, features_2 = self.feature_matching(image_1, image_2)
            
            # Retriangulate older pair if i != 0
            if i !=0:
                features_0 , features_1, points_3d = self. triangulation(pose_0, pose_1, features_0, features_1)
                features_1= features_1.T
                points_3d= cv2.convertPointsFromHomogeneous(points_3d.T)
                
            # Find common points across old + new matches
            cm_points_0, cm_points_1, cm_mask_0, cm_mask_1= self.find_common_points(features_1,features_cur, features_2)
            cm_points_2 = features_2[cm_points_1]
            cm_points_cur = features_cur[cm_points_1]
            
            # SolvePnP for new image
            rot_matrix, tran_matrix, cm_points_2, points_3d, cm_points_cur = self.solve_PnP(points_3d[cm_points_0], cm_points_2, self.img_obj.K, np.zeros((5, 1), dtype=np.float32), cm_points_cur, initial = 0)
            transform_matrix_1= np.hstack((rot_matrix, tran_matrix))
            pose_2 = np.matmul(self.img_obj.K, transform_matrix_1)

            # Reproj error for new image
            error, points_3d= self.reproj_error(points_3d, cm_points_2, transform_matrix_1, self.img_obj.K, homogenity=0)
            
            # Triangulate between pose_1 and the new pose_2
            cm_mask_0, cm_mask_1, points_3d= self.triangulation(pose_1, pose_2, cm_mask_0, cm_mask_1)
            error, points_3d = self.reproj_error(points_3d, cm_mask_1, transform_matrix_1, self.img_obj.K, homogenity=1)
            print("Reprojection error:", error)
            pose_array = np.hstack((pose_array, pose_2.ravel()))

            # If bundle adjustment is enabled, refine
            if bundle_adjustment_enabled:
                points_3d, cm_mask_1, transform_matrix_1 = self.compute_bundle_adjustment(points_3d, cm_mask_1,
                                                                                          transform_matrix_1, self.img_obj.K,
                                                                                          threshold)
                pose_2 = np.matmul(self.img_obj.K, transform_matrix_1)
                error, points_3d = self.reproj_error(points_3d, cm_mask_1, transform_matrix_1, self.img_obj.K, homogenity = 0)
                print("Reprojection error after Bundle Adjustment: ",error)

                # Accumulate 3D points & colors
                total_points = np.vstack((total_points, points_3d))
                points_left = np.array(cm_mask_1, dtype=np.int32)
                color_vector = np.array([image_2[l[1], l[0]] for l in points_left])
                total_colors = np.vstack((total_colors, color_vector))
            else:
                # No BA: just store points as Nx3
                total_points = np.vstack((total_points, points_3d[:, 0, :]))
                points_left = np.array(cm_mask_1, dtype=np.int32)
                color_vector = np.array([image_2[l[1], l[0]] for l in points_left.T])
                total_colors = np.vstack((total_colors, color_vector)) 

            # Update references for next iteration
            transform_matrix_0 = np.copy(transform_matrix_1)
            pose_0 = np.copy(pose_1)
            plt.scatter(i, error)
            plt.pause(0.05)

            image_0 = np.copy(image_1)
            image_1 = np.copy(image_2)
            features_0 = np.copy(features_cur)
            features_1 = np.copy(features_2)
            pose_1 = np.copy(pose_2)

            # Display current image
            cv2.imshow(self.img_obj.image_list[0].split('\\')[-2], image_2)
            if cv2.waitKey(1) & 0xff == ord('q'):
                break
        cv2.destroyAllWindows()

        # Save reprojection error plot
        if bundle_adjustment_enabled:
            plot_dir = os.path.join(self.img_obj.path, 'Results with Bundle Adjustment')
        else:
            plot_dir = os.path.join(self.img_obj.path, 'Results')

        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)

        plt.xlabel('Image Index')
        plt.ylabel('Reprojection Error')
        plt.title('Reprojection Error Plot')
        plt.savefig(os.path.join(plot_dir, 'reprojection_errors.png'))
        plt.close()

        # print("Saving to .ply file.......")

        # Ensure total_points and total_colors have valid data
        if total_points.size == 0 or total_colors.size == 0:
            print("Error: No points or colors to save. Skipping point cloud generation.")
        else:
            print(f"Total points to save: {total_points.shape[0]}")
            print(f"Total colors to save: {total_colors.shape[0]}")
        
        # Check if we actually have points
        scaling_factor=5000.0
        self.save_to_ply(self.img_obj.path, total_points, total_colors,
                         bundle_adjustment_enabled, binary_format=True, scaling_factor=scaling_factor)
        print("Saved the point cloud to .ply file!!!")

        # construct the output path for the pose array CSV
        results_dir = os.path.join(self.img_obj.path, 'Results Array')
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

        parent_folder= os.path.basename(os.path.dirname(self.img_obj.image_list[0]))
        pose_csv_name= f"{parent_folder}_pose_array.csv"
        pose_csv_path= os.path.join(results_dir, pose_csv_name)

        np.savetxt(pose_csv_path,pose_array, delimiter='\n')

if __name__ == '__main__':
     # Example usage:
    # Provide path to your dataset directory containing images + K.txt
    sfm = StructurefromMotion("Dataset/fountain-P11")
    sfm()
    # sfm(bundle_adjustment_enabled=True)