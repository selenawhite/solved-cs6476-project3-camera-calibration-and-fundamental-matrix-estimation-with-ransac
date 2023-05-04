Download Link: https://assignmentchef.com/product/solved-cs6476-project3-camera-calibration-and-fundamental-matrix-estimation-with-ransac
<br>



<ul>

 <li>Project materials including report template: <a href="https://cc.gatech.edu/~hays/compvision/proj3/proj3.zip">zip</a></li>

 <li>Required files: &lt;your_gt_username&gt;.zip, &lt;your_gt_username&gt;_proj3.pdf</li>

</ul>

Figure 1: An autonomous vehicle makes a right turn and captures two images a moment apart. The first image happens to contain an identical autonomous vehicle ahead of it. Epipolar lines show the camera locations given corresponding points in two views of a scene. Corresponding points in the two images are marked by circles of the same color. The two vehicles have the same hardware, and note that the epipole in the left image is located at the same height as the camera mounted on the other vehicle.

<h1>Overview</h1>

The goal of this project is to introduce you to camera and scene geometry. Specifically we will estimate the camera projection matrix, which maps 3D world coordinates to image coordinates, as well as the fundamental matrix, which relates points in one scene to epipolar lines in another. The camera projection matrix and fundamental matrix can each be estimated using point correspondences. To estimate the projection matrix (camera calibration), the input is corresponding 3D and 2D points. To estimate the fundamental matrix the input is corresponding 2D points across two images. You will start out by estimating the projection matrix and the fundamental matrix for a scene with ground truth correspondences. Then you will move on to estimating the fundamental matrix using point correspondences that are obtained using SIFT.

Remember these challenging images of Gaudi’s Episcopal Palace from project 2? By using RANSAC to find the fundamental matrix with the most inliers, we can filter away spurious matches and achieve near perfect point-to-point matching as shown below:

Figure 2: Gaudi’s Episcopal Palace.

<h1>Setup</h1>

<ol>

 <li>Install <a href="https://conda.io/miniconda.html">Miniconda</a><a href="https://conda.io/miniconda.html">.</a> It doesn’t matter whether you use Python 2 or 3 because we will create our own environment that uses python3 anyways.</li>

 <li>Download and extract the project starter code.</li>

 <li>Create a conda environment using the appropriate command. On Windows, open the installed “Conda prompt” to run the command. On MacOS and Linux, you can just use a terminal window to run the command, Modify the command based on your OS (linux, mac, or win): conda env create -f</li>

</ol>

proj3_env_&lt;OS&gt;.yml

<ol start="4">

 <li>This will create an environment named “cs6476 proj3”. Activate it using the Windows command, activate cs6476_proj3 or the MacOS / Linux command, conda activate cs6476_proj3 or source activate cs6476_proj3</li>

 <li>Install the project package, by running pip install -e . inside the repo folder.</li>

 <li>Run the notebook using jupyter notebook ./proj3_code/proj3.ipynb</li>

 <li>After implementing all functions, ensure that all sanity checks are passing by running pytest proj3_unit_tests inside the repo folder.</li>

 <li>Generate the zip folder for the code portion of your submission once you’ve finished the project using python zip_submission.py –gt_username &lt;your_gt_username&gt;</li>

</ol>

<h1>1           Part 1: Camera projection matrix</h1>

<h2>Introduction</h2>

The goal is to compute the projection matrix that goes from world 3D coordinates to 2D image coordinates. Recall that using homogeneous coordinates the equation for moving from 3D world to 2D camera coordinates is:




Another way of writing this equation is:




→ (<em>m</em>31<em>X </em>+ <em>m</em>32<em>Y </em>+ <em>m</em>33<em>Z </em>+ <em>m</em>34)<em>u </em>= <em>m</em>11<em>X </em>+ <em>m</em>12<em>Y </em>+ <em>m</em>13<em>Z </em>+ <em>m</em>14

→ 0 = <em>m</em>11<em>X </em>+ <em>m</em>12<em>Y </em>+ <em>m</em>13<em>Z </em>+ <em>m</em>14 − <em>m</em>31<em>uX </em>− <em>m</em>32<em>uY </em>− <em>m</em>33<em>uZ </em>− <em>m</em>34<em>u</em>




→ (<em>m</em>31<em>X </em>+ <em>m</em>32<em>Y </em>+ <em>m</em>33<em>Z </em>+ <em>m</em>34)<em>v </em>= <em>m</em>21<em>X </em>+ <em>m</em>22<em>Y </em>+ <em>m</em>23<em>Z </em>+ <em>m</em>24

→ 0 = <em>m</em>21<em>X </em>+ <em>m</em>22<em>Y </em>+ <em>m</em>23<em>Z </em>+ <em>m</em>24 − <em>m</em>31<em>vX </em>− <em>m</em>32<em>vY </em>− <em>m</em>33<em>vZ </em>− <em>m</em>34<em>v</em>

At this point, you’re almost able to set up your linear regression to find the elements of the matrix <em>M</em>. There’s only one problem–the matrix <em>M </em>is only defined up to a scale. Therefore, these equations have many different possible solutions (in particular <em>M </em>= all zeros is a solution, which is not very helpful in our context). The way around this is to first fix a scale, and then do the regression. There are several options for doing this: 1) You can fix the last element, <em>m</em><sub>34 </sub>= 1, and then find the remaining coefficients, or 2) you can use the singular value decomposition to directly solve the constrained optimization problem:

arg min     k<em>Ax</em>k

<em><sup>x                                                                                                                                          </sup></em>

s.t.               k<em>x</em>k = 1

To make sure that your code is correct, we are going to give you a set of “normalized points” in the files pts2d-norm-pic_a.txt and pts3d-norm.txt. If you solve for <em>M </em>using all the points, you should get a matrix that is a scaled equivalent of the following:

−0<em>.</em>4583               0<em>.</em>2947   0<em>.</em>0139   −0<em>.</em>0040

For example, this matrix will take the last normalized 3D point, <em>&lt; </em>1<em>.</em>2323<em>,</em>1<em>.</em>4421<em>,</em>0<em>.</em>4506<em>,</em>1<em>.</em>0 <em>&gt;</em>, and project it to <em>&lt; u,v &gt; </em>of <em>&lt; </em>0<em>.</em>1419<em>,</em>0<em>.</em>4518 <em>&gt;</em>, converting the homogeneous 2D point <em>&lt; us,vs,s &gt; </em>to its in homogeneous version (the transformed pixel coordinate in the image) by dividing by <em>s</em>.

First, you will need to implement the least squares regression to solve for <em>M </em>given the corresponding normalized points. The starter code will load 20 corresponding normalized 2D and 3D points. You have to write the code to set up the linear system of equations, solve for the unknown entries of <em>M</em>, and reshape it into the estimated projection matrix. To validate that you’ve found a reasonable projection matrix, we’ve provided evaluation code which computes the total “residual” between the projected 2D location of each 3D point and the actual location of that point in the 2D image. The residual is just the distance (square root of the sum of squared differences in <em>u </em>and <em>v</em>). This should be very small.

Once you have an accurate projection matrix <em>M</em>, it is possible to tease it apart into the more familiar and more useful matrix <em>K </em>of intrinsic parameters and matrix [<em>R</em>|<em>T</em>] of extrinsic parameters. For this project we will only ask you to estimate one particular extrinsic parameter: the camera center in world coordinates. Let us define <em>M </em>as being composed of a 3 × 3 matrix, <em>Q</em>, and a 4th column, <em>m</em><sub>4</sub>:

(5)

From class we said that the center of the camera <em>C </em>could be found by:

<em>C </em>= −<em>Q</em><sup>−1</sup><em>m</em><sub>4                                                                                                                                     </sub>(6)

To debug your code, if you use you the normalized 3D points to get the <em>M </em>given above, you would get a camera center of:

<em>C</em><sub>norm<em>A </em></sub>=<em>&lt; </em>−1<em>.</em>5125<em>,</em>−2<em>.</em>3515<em>,</em>0<em>.</em>2826 <em>&gt;</em>

We’ve also provided a visualization which will show the estimated 3D location of the camera with respect to the normalized 3D point coordinates.

In part1_projection_matrix.py, you will implement the following:

<ul>

 <li>projection(): Projects homogeneous world coordinates [<em>X,Y,Z,</em>1] to non-homogeneous image coordinates (<em>u,v</em>). Given projection matrix <em>M</em>, the equations that accomplish this are (2) and (3).</li>

 <li>calculate_projection_matrix(): Solves for the camera projection matrix using a system of equations set up from corresponding 2D and 3D points.</li>

 <li>calculate_camera_center(): Computes the camera center location in world coordinates.</li>

</ul>

<h1>2           Part 2: Fundamental matrix</h1>

Figure 3: Two-camera setup. Reference: Szeliski, p. 682.

The next part of this project is estimating the mapping of points in one image to lines in another by means of the fundamental matrix. This will require you to use similar methods to those in part I. We will make use of the corresponding point locations listed in pts2d-pic_a.txt and pts2d-pic_b.txt. Recall that the definition of the fundamental matrix is:

= 0

for a point (<em>u,v,</em>1) in image A, and a point (<em>u</em><sup>0</sup><em>,v</em><sup>0</sup><em>,</em>1) in image B. See Appendix A for the full derivation. Note: the fundamental matrix is sometimes defined as the transpose of the above matrix with the left and right image points swapped. Both are valid fundamental matrices, but the visualization functions in the starter code assume you use the above form.

Another way of writing this matrix equations is:

= 0

Which is the same as:

= 0                           (9)

Starting to see the regression equations? Given corresponding points you get one equation per point pair. With 8 or more points you can solve this (why 8?). Similar to part I, there’s an issue here where the matrix is only defined up to scale and the degenerate zero solution solves these equations. So you need to solve using the same method you used in part I of first fixing the scale and then solving the regression.

The least squares estimate of <em>F </em>is full rank; however, a proper fundamental matrix is a rank 2. As such we must reduce its rank. In order to do this, we can decompose <em>F </em>using singular value decomposition into the matrices <em>U</em>Σ<em>V </em><sup>0 </sup>= <em>F</em>. We can then construct a rank 2 matrix by setting the smallest singular value in Σ to zero thus generating Σ<sub>2 </sub>. The fundamental matrix is then easily calculated as <em>F </em>= <em>U</em>Σ<sub>2</sub><em>V </em><sup>0</sup>. You can check your fundamental matrix estimation by plotting the epipolar lines using the plotting function provided in the starter code.

<h2>Coordinate normalization</h2>

As discussed in lecture, your estimate of the fundamental matrix can be improved by normalizing the coordinates before computing the fundamental matrix (see [1] by Hartley). It is suggested for this project you perform the normalization through linear transformations as described below to make the mean of the points zero and the average magnitude 1.0.




The transform matrix <em>T </em>is the product of the scale and offset matrices. <em>c<sub>u </sub></em>and <em>c<sub>v </sub></em>are the mean coordinates. To compute a scale <em>s </em>you could estimate the standard deviation after subtracting the means. Then the scale factor <em>s </em>would be the reciprocal of whatever estimate of the scale you are using. You could use one scale matrix based on the statistics of the coordinates from both images or you could do it per image.

In part2_fundamental_matrix.py you will need to use the scaling matrices to adjust your fundamental matrix so that it can operate on the original pixel coordinates. This is performed as follows:

<em>F</em><em>orig </em>= <em>T</em><em>bT </em>∗ <em>F</em><em>norm </em>∗ <em>T</em><em>a                                                                                                           </em>

In part2_fundamental_matrix.py, you will implement the following:

<ul>

 <li>normalize_points(): Normalizes the 2D coordinates.</li>

 <li>unnormalize_F(): Adjusts the fundamental matrix to account for the normalized coordinates. See Appendix B.</li>

 <li>estimate_fundamental_matrix(): Calculates the fundamental matrix.</li>

</ul>

<h1>3           Part 3: Fundamental matrix with RANSAC</h1>

For two photographs of a scene it’s unlikely that you’d have perfect point correspondence with which to do the regression for the fundamental matrix. So, next you are going to compute the fundamental matrix with point correspondences computed using SIFT. As discussed in class, least squares regression alone is not appropriate in this scenario due to the presence of multiple outliers. In order to estimate the fundamental matrix from this noisy data you’ll need to use RANSAC in conjunction with your fundamental matrix estimation.

You’ll use these putative point correspondences and RANSAC to find the “best” fundamental matrix. You will iteratively choose some number of point correspondences (8, 9, or some small number), solve for the fundamental matrix using the function you wrote for part II, and then count the number of inliers. Inliers in this context will be point correspondences that “agree” with the estimated fundamental matrix. In order to count how many inliers a fundamental matrix has, you’ll need a distance metric based on the fundamental matrix. (Hint: For a point correspondence (<em>x,x</em><sup>0</sup>) what properties does the fundamental matrix have?). You’ll need to pick a threshold between inliers and outliers and your results are very sensitive to this threshold, so explore a range of values. You don’t want to be too permissive about what you consider an inlier, nor do you want to be too conservative. Return the fundamental matrix with the most inliers.

Recall from lecture the expected number of iterations of RANSAC to find the “right” solution in the presence of outliers. For example, if half of your input correspondences are wrong, then you have a (0<em>.</em>5)<sup>8 </sup>= 0<em>.</em>39% chance to randomly pick 8 correspondences when estimating the fundamental matrix. Hopefully that correct fundamental matrix will have more inliers than one created from spurious matches, but to even find it you should probably do thousands of iterations of RANSAC.

For many real images, the fundamental matrix that is estimated will be “wrong” (as in it implies a relationship between the cameras that must be wrong, e.g., an epipole in the center of one image when the cameras actually have a large translation parallel to the image plane). The estimated fundamental matrix can be wrong because the points are co-planar or because the cameras are not actually pinhole cameras and have lens distortions. Still, these “incorrect” fundamental matrices tend to do a good job at removing incorrect SIFT matches (and, unfortunately, many correct ones).

For this part, you will implement the following methods in part3_ransac.py:

<ul>

 <li>calculate_num_ransac_iterations(): Calculates the number of RANSAC iterations needed for a given guarantee of success.</li>

 <li>ransac_fundamental_matrix(): Uses RANSAC to find the best fundamental matrix.</li>

</ul>

<h1>4           Part 4: Performance comparison</h1>

In this part, you will compare the performance of using the direct least squares method against RANSAC. You can use the code in the notebook for visualization of the fundamental matrix estimation. The first one uses a subset of the matches and the direct linear method (which corresponds to the estimate_fundamental_matrix () in part 2) to compute the fundamental matrix. The second method uses RANSAC, which corresponds to ransac_fundamental_matrix() in part 3. Based on your output visualizations, answer the reflection questions in the report.

<h1>5           Part 5: Visual odometry</h1>

Visual odometry (VO) is an important part of the simultaneous localization and mapping (SLAM) problem. In this part, you can observe the implementation of VO on a real-world example from Argoverse. VO will allow us to recreate most of the ego-motion of a camera mounted on a robot – the relative translation (but only up to an unknown scale), and the relative rotation. See [2] for a more detailed understanding. Based on the output and your understanding of the process, answer the reflection questions in the report.

<h1>6           Writeup</h1>

For this project (and all other projects), you must do a project report using the template slides provided to you. Do <em>not </em>change the order of the slides or remove any slides, as this will affect the grading process on Gradescope and you will be deducted points. In the report you will describe your algorithm and any decisions you made to write your algorithm a particular way. Then you will show and discuss the results of your algorithm. The template slides provide guidance for what you should include in your report. A good writeup doesn’t just show results – it tries to draw some conclusions from the experiments. You must convert the slide deck into a PDF for your submission, and then assign each PDF page to the relevant question number on Gradescope.

If you choose to do anything extra, add slides <em>after the slides given in the template deck </em>to describe your implementation, results, and analysis. Adding slides in between the report template will cause issues with Gradescope, and you will be deducted points. You will not receive full credit for your extra credit implementations if they are not described adequately in your writeup.

<h2>Potentially useful NumPy (Python library) functions</h2>

<ul>

 <li>linalg.svd() – This function returns the singular value decomposition of a matrix. Useful for solving the linear systems of equations you build and reducing the rank of the fundamental matrix.</li>

 <li>linalg.inv() – This function returns the inverse of a matrix.</li>

 <li>random.randint() – Lets you pick integers from a range. Useful for RANSAC.</li>

</ul>

<h2>Forbidden functions</h2>

(You can use these for testing, but not in your final code). You may not use the SciPy constrained least squares function scipy.optimize.lsq_linear() or any similar function. You may also not use anyone else’s code that estimates the fundamental matrix or performs RANSAC for you. If it feels like you’re sidestepping the work, then it’s probably not allowed. Ask the TAs if you have any doubts.

<h1>Testing</h1>

We have provided a set of tests for you to evaluate your implementation. We have included tests inside proj3.ipynb so you can check your progress as you implement each section. When you’re done with the entire project, you can call additional tests by running pytest proj3_unit_tests inside the root directory of the project, as well as checking against the tests on Gradescope. <em>Your grade on the coding portion of the project will be further evaluated with a set of tests not provided to you.</em>

<h1>Bells &amp; whistles (extra points)</h1>

We don’t have good suggestions for extra credit on this project. If you have ideas, come talk to us! If you choose to do extra credit, you should add slides <em>at the end </em>of your report further explaining your implementation, results, and analysis. You will not be awarded credit if this is missing from your submission.

<h1>Rubric</h1>

<ul>

 <li>+20 pts: Implementation of projection matrix estimation in py</li>

 <li>+25 pts: Implementation of fundamental matrix estimation in py</li>

 <li>+25 pts: Implementation of RANSAC in py</li>

 <li>+30 pts: Report</li>

 <li>-5*n pts: Lose 5 points for every time you do not follow the instructions for the hand-in format</li>

</ul>

<h1>Submission format</h1>

This is very important as you will lose 5 points for every time you do not follow the instructions. You will submit two items to Gradescope:

<ol>

 <li>&lt;your_gt_username&gt;.zip containing:

  <ul>

   <li>proj3_code/ – directory containing all your code for this assignment</li>

   <li>additional_data/ – (optional) if you use any data other than the images we provide, please include them here</li>

  </ul></li>

 <li>&lt;your_gt_usernamme&gt;_proj3.pdf – your report</li>

</ol>

Do <em>not </em>install any additional packages inside the conda environment. The TAs will use the same environment as defined in the config files we provide you, so anything that’s not in there by default will probably cause your code to break during grading. Do <em>not </em>use absolute paths in your code or your code will break. Use relative paths like the starter code already does. Failure to follow any of these instructions will lead to point deductions. Create the zip file using python zip_submission.py –gt_username &lt;your_gt_username&gt; (it will zip up the appropriate directories/files for you!) and hand it in with your report PDF through Gradescope (please remember to mark which parts of your report correspond to each part of the rubric).

<h1>Credits</h1>

Assignment developed by James Hays, Cusuh Ham, Arvind Krishnakumar, Jing Wu, John Lambert, Samarth Brahmbhatt, Grady Williams, and Henry Hu, based on a similar project by Aaron Bobick.

<h1>References</h1>

<ul>

 <li>Richard I Hartley. “In defense of the eight-point algorithm”. In: <em>IEEE Transactions on pattern analysis and machine intelligence </em>6 (1997), pp. 580–593.</li>

 <li>Scaramuzza and F. Fraundorfer. “Visual Odometry [Tutorial]”. In: <em>IEEE Robotics Automation Magazine </em>18.4 (2011), pp. 80–92. doi: <a href="https://doi.org/10.1109/MRA.2011.943233">10.1109/MRA.2011.943233</a><a href="https://doi.org/10.1109/MRA.2011.943233">.</a></li>

</ul>

<h1>Appendix A           Fundamental matrix derivation</h1>

Recall that the definition of the fundamental matrix is:

= 0                                                                   (12)

Where does this equation come from? Szeliski shows that a 3D point <strong>p </strong>being viewed from two cameras (see Figure 3) can be modeled as:

<em>d</em>1<strong>x</strong>ˆ1 = <strong>p</strong>1 = 1<strong>R</strong>0<strong>p</strong>0 + 1<strong>t</strong>0 = 1<strong>R</strong>0(<em>d</em>0<strong>x</strong>ˆ0) + 1<strong>t</strong>0                                                                                        (13)

where <strong>x</strong>ˆ<em><sub>j </sub></em>= <strong>K</strong><sup>−</sup><em><sub>j </sub></em><sup>1</sup><strong>x</strong><em><sub>j </sub></em>are the (local) ray direction vectors. Note that <sup>1</sup><strong>R</strong><sub>0 </sub>and <sup>1</sup><strong>t</strong><sub>0 </sub>define an SE(3) ‘1 T 0‘ object that transforms <strong>p</strong><sub>0 </sub>from camera 0’s frame to camera 1’s frame. We’ll refer to these just as <strong>R </strong>and <strong>t </strong>for brevity in the following derivation.

We can eliminate the +<strong>t </strong>term by a cross-product. This can be achieved by multiplying with a skewsymmetric matrix as [<strong>t</strong>]<sub>×</sub><strong>t </strong>= 0. Then:

<em>d</em><sub>1</sub>[<strong>t</strong>]<sub>×</sub><strong>x</strong>ˆ<sub>1 </sub>= <em>d</em><sub>0</sub>[<strong>t</strong>]<sub>×</sub><strong>Rx</strong>ˆ<sub>0</sub><em>.                                                                            </em>(14)

Swapping sides and taking the dot product of both sides with <strong>x</strong>ˆ<sub>1 </sub>yields

<em>,                                                               </em>(15)

Since the cross product [<strong>t</strong>]<em><sub>x </sub></em>returns 0 when pre- and post-multiplied by the same vector, we arrive at the familiar epipolar constraint, where <strong>E </strong>= [<strong>t</strong>]<sub>×</sub><strong>R</strong>:

<strong>x</strong>ˆ = 0                                                                                   (16)

The fundamental matrix is defined as<strong>EK</strong><sup>−</sup><sub>0 </sub><sup>1</sup>. Thus,

<strong>EK</strong> = 0                                                               (17)

We can write this as:

= 0                                                           (18)

<h1>Appendix B           Unnormalizing normalized coordinates</h1>

The main idea of coordinate normalization is to replace coordinates <strong>u</strong><em><sub>a </sub></em>in image <em>a </em>with <strong>uˆ</strong><em><sub>a </sub></em>= <em>T<sub>a</sub></em><strong>u</strong><em><sub>a</sub></em>, and coordinates <strong>u</strong><em><sub>b </sub></em>in image <em>b </em>with <strong>uˆ</strong><em><sub>b </sub></em>= <em>T<sub>b</sub></em><strong>u</strong><em><sub>b</sub></em>. If <em>T </em>is chosen to be invertible, then we can recover the original coordinates from the transformed ones, as

<table width="367">

 <tbody>

  <tr>

   <td width="343"><strong>uˆ</strong><em>a </em>= <em>T</em><em>a</em><strong>u</strong><em>a</em></td>

   <td width="24"> </td>

  </tr>

  <tr>

   <td width="343"><em>T</em><em>a</em>−1<strong>uˆ</strong><em>a </em>= <em>T</em><em>a</em>−1<em>T</em><em>a</em><strong>u</strong><em>a</em><em>T</em><em>a</em>−1<strong>uˆ</strong><em>a </em>= <strong>u</strong><em>a</em></td>

   <td width="24">(19)</td>

  </tr>

 </tbody>

</table>

Substituting in the equation <strong>u</strong><strong>      </strong>= 0, we derive the equation

<table width="377">

 <tbody>

  <tr>

   <td width="353"></td>

   <td width="24"> </td>

  </tr>

  <tr>

   <td width="353">(<em>T</em><em>b</em>−1<strong>uˆ</strong><em>b</em>)<em>TFT</em><em>a</em>−1<strong>uˆ</strong><em>a </em>= 0</td>

   <td width="24">(20)</td>

  </tr>

 </tbody>

</table>

<strong>uˆ</strong>

If we use the normalized points <strong>u</strong> when fitting the fundamental matrix, then we will end up estimating

. In other words, <strong>u</strong><em><sup>T</sup><sub>b </sub>F</em><strong>u</strong><em><sub>a </sub></em>= <strong>uˆ</strong><em><sup>T</sup><sub>b </sub>F</em><sup>ˆ</sup><strong>uˆ</strong><em><sub>a</sub></em>. If we want to find out the original <em>F </em>that corresponded

to raw (unnormalized) point coordinates, than we need to transform backwards:

<em>F</em>ˆ = <em>T</em><em>b</em>−<em>T</em><em>FT</em><em>a</em>−1 <em>T</em><em>bTF</em>ˆ = <em>T</em><em>bTT</em><em>b</em>−<em>T</em><em>FT</em><em>a</em>−1

(21)

<em>T</em><em>bTFT</em>ˆ <em>a </em>= <em>FT</em><em>a</em>−1<em>T</em><em>a</em>