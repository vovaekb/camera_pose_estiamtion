# camera_pose_estimation

Source code of final solution for the task of camera pose estimation in 3D scene using an image from camera.

## Description
Idea: given a set of images used for 3D scene registration and camera pose for each single image we need to estimate the pose of the camera where a query image was captured.
In this project we performed research on various techniques on image and point cloud matching was performed. 
Final solution: matching query image against all the original images used for 3D scene reconstruction. As keypoint detector AffNet algorithm was chosen as an innovative and efficient CNN based method for detecting keypoints on image robust to affine transformations which outperforms state-of-the-art approaches like Bag-of-Words on image matching and wide baseline stereo problems.

Stack: Python, OpenCV

## References
Demo of AffNet keypoint detection results on slides: [link](https://docs.google.com/presentation/d/17M39q3sez9UD4FPHgEoio43lXqD91nCf1ghuqn1ZPf8/edit#slide=id.g8433d5c4a0_0_106).

Article on Habr (rus): [link](https://habr.com/ru/articles/535162/)
