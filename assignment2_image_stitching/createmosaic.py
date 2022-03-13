#Assignmnet_2 COL780  Image mosaicing : submission 2020SIY7566
import os
import cv2
import argparse
import numpy as np

# Initialize SIFT
sift = cv2.xfeatures2d.SIFT_create()
# Parameters for nearest-neighbor matching
FLANN_INDEX_KDTREE = 1
flann_params = dict(algorithm = FLANN_INDEX_KDTREE,
    trees = 5)
matcher = cv2.FlannBasedMatcher(flann_params, {})

# Use the keypoints to stitch the images
def get_stitched_image(img2, img1, M):

	# Get width and height of input images
	w1,h1 = img1.shape[:2]
	w2,h2 = img2.shape[:2]

	# Get the canvas dimesions
	img1_dims = np.float32([ [0,0], [0,w1], [h1, w1], [h1,0] ]).reshape(-1,1,2)
	img2_dims_temp = np.float32([ [0,0], [0,w2], [h2, w2], [h2,0] ]).reshape(-1,1,2)


	# Get relative perspective of second image
	img2_dims = cv2.perspectiveTransform(img2_dims_temp, M)

	# Resulting dimensions
	result_dims = np.concatenate( (img1_dims, img2_dims), axis = 0)

	# Getting images together
	# Calculate dimensions of match points
	[x_min, y_min] = np.int32(result_dims.min(axis=0).ravel() - 0.5)
	[x_max, y_max] = np.int32(result_dims.max(axis=0).ravel() + 0.5)

	# Create output array after affine transformation
	transform_dist = [-x_min,-y_min]
	transform_array = np.array([[1, 0, transform_dist[0]], [0, 1, transform_dist[1]], [0,0,1]])

	# Warp images to get the resulting image
	result_img = cv2.warpPerspective(img2, transform_array.dot(M), (x_max-x_min, y_max-y_min))
	result_img[transform_dist[1]:w1+transform_dist[1], transform_dist[0]:h1+transform_dist[0]] = img1

	# Return the result
	return result_img

# Find SIFT and return Homography Matrix
def get_sift_homography(img2, img1):

	# Extract keypoints and descriptors
	k1, d1 = sift.detectAndCompute(img1, None)
	k2, d2 = sift.detectAndCompute(img2, None)

	matches = matcher.knnMatch(d1, d2, k=2)

	# David Lowe's threshold of good matches
	check_matches = []
	for m1,m2 in matches:
		if m1.distance < 0.7 * m2.distance:
			if m1.queryIdx:
				check_matches.append(m1)

	
	min_m = 8 #threshold for minimum number of matches accepted
	if len(check_matches) > min_m:
		
		# Array to store matching points
		pts1 = []
		pts2 = []

		# Add matching points to array
		for match in check_matches:
			pts1.append(k1[match.queryIdx].pt)
			pts2.append(k2[match.trainIdx].pt)
		img1_pts = np.float32(pts1).reshape(-1,1,2)
		img2_pts = np.float32(pts2).reshape(-1,1,2)
		
		# Compute homography matrix
		M, mask = cv2.findHomography(img1_pts, img2_pts, cv2.RANSAC, 5.0)
		return M
	else:
		print ('Error: Case not handled too less matches')
		exit()
	

#parse arguments
def parse_arguments():
    parser = argparse.ArgumentParser(description='This implementation creates mosaics from given set of imagess ')
    parser.add_argument('-i', '--img_path', type=str, default='img', required=True, help="Path to the image folder")

    arguments = parser.parse_args()
    return arguments

# Equalize Histogram of Color Images
def equalize_histogram_color(img):
	img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
	img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])

	# Show the colored image
	img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
	return img

#calculates homography based on sift feature matching and returns stitched image
def stitch_images(Images):
	result_image = Images["eq"][0]
	result_image_raw = Images["raw"][0]
	del Images["eq"][0]
	del Images["raw"][0]

	for i in range(len(Images["eq"])) :
		# Use SIFT to find keypoints and return homography matrix
		M =  get_sift_homography(result_image, Images["eq"][i])

		# Stitch the images together using homography matrix
		result_image = get_stitched_image(Images["raw"][i], result_image_raw, M)

	return result_image

# Main function
def main():
	# parse arguments
	args = parse_arguments();
	dir = args.img_path

	# list of images
	Images = {}
	Images["raw"]=[]
	Images["eq"]=[]

	# read images and equalize histogram
	for filename in os.listdir(dir):
		img = cv2.imread(os.path.join(dir, filename))
		Images["raw"].append(img)
		img = cv2.GaussianBlur(img, (5,5), 0)
		img = equalize_histogram_color(img)
		Images["eq"].append(img)

	#returns stitched image
	pano = stitch_images(Images)

	save_path = "result_"+dir.split("/")[-1]+".jpg"
	# Show the resulting image
	# cv2.imshow ('Result', pano)
	# cv2.waitKey(0)
	cv2.imwrite(save_path, pano)
	# cv2.destroyAllWindows()

# Calling main function
if __name__=='__main__':
	main()
