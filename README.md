# Image Stitching

**1. Corner Detection** ​​:
    Implementation Summary: Corner detection is implemented using cornerHarris from OpenCV, which
    returns a score for each pixel of the image (higher values indicate more strength as a corner).

**2. Adaptive Non-Maximal Suppression** ​​:
    Implementation Summary: We loop through each of the feature points (flattened to improve runtime by
    storing each i, j, and the value into a vector) and conduct pairwise comparison of strength with every
    other point (subject to within 0.9 magnitude). The minimal distance to a larger strength point is also kept
    track of. Then, the list of minimum radius to a stronger point is then sorted and the N top are chosen.



**3. Feature Descriptors:**
    Implementation Summary: For each point of interest previously found, we subsample a 40 x 40 image
    around it. We mirror the image if the patch goes out of image bounds. We then apply a Gaussian Blur in
    order to smooth the image. Then, we sample an 8 x 8 amount of points in a grid spaced 5 points apart in
    the sub-sampled image. Each of these 8 x 8 is flattened into a column with 64 descriptors, where each
    column represents a point of interest. Then, the mean and standard deviation of each column is found in
    order to standardize the points in each column vector.

**4. Feature Matching:**
    Implementation Summary: For each interest point in the first image, we find the two most similar
    neighbors in the second image using scipy.spatial’s KDTree. To find if the match is in fact good, we test
    if the ratio is smaller than the threshold to see if it's a good match. If it’s a good match, put the index of
    the best match descriptor into match, and if it’s not a good, match put -1 to indicate no good match
    found.
    
**5. RANSAC** ​​:
    Implementation Summary: From the matched points, we sample four pair of points and find the
    homography that relates them. We then apply the homography to the entire matched set. We consider a
    pair to be an inlier if the distance between the transformed point and target point are not farther from
    each other than a certain threshold. We find the homography that has the highest number of inliers and
    recalculate the homography on the entire inlier set.


**6. Mosaic** ​​:
    Implementation Summary: We use a recursive function to do the mosaicing. The function stitches
    together consecutive images and then calls the same function on the resulting reduced array of images
    till the entire set is stitched together into one image. The stitching is done in stitch.py where we
    calculate the size of the stitched image and the displacement of the target image. Then we loop through
    the empty stitched image and find the inverse homography to fill in the pixels.
We conducted the following stitching procedure:
First, we stitched the first two images together, (left and middle), and then stitched the last two images (right and
middle). The final image is produced by then stitching these stitch results together. 

Note: imageTests file is created to generate the images showing the outputs of the corner detection, adaptive
NMS, and RANSAC functions.



