import cv2
import numpy as np

class LoopClosureDetector:
    def __init__(self, detector, matcher, match_threshold=30, min_loop_interval=30):
        """
        Args:
            detector: OpenCV feature detector (e.g., SIFT).
            matcher: OpenCV descriptor matcher (e.g., BFMatcher).
            match_threshold (int): Minimum number of good matches to accept a loop.
            min_loop_interval (int): Minimum index difference between keyframes to consider.
        """
        self.detector = detector
        self.matcher = matcher
        self.match_threshold = match_threshold
        self.min_loop_interval = min_loop_interval
        self.keyframes = []  # list of (img_gray, keypoints, descriptors)

    def add_keyframe(self, img_gray):
        kp, des = self.detector.detectAndCompute(img_gray, None)
        self.keyframes.append((img_gray, kp, des))

    def detect_loop(self, img_gray, curr_idx):
        """
        Args:
            img_gray (np.ndarray): Current grayscale image.
            curr_idx (int): Current keyframe index.

        Returns:
            matched_idx (int or None): Index of matched keyframe, or None if no loop found.
        """
        kp_curr, des_curr = self.detector.detectAndCompute(img_gray, None)

        for idx, (img_prev, kp_prev, des_prev) in enumerate(self.keyframes):
            if curr_idx - idx < self.min_loop_interval:
                continue

            matches = self.matcher.knnMatch(des_curr, des_prev, k=2)

            # Apply Lowe's ratio test
            good_matches = []
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)

            if len(good_matches) >= self.match_threshold:
                return idx  # Detected loop to keyframe idx

        return None

class LoopClosureConstraint:
    def __init__(self, curr_idx, matched_idx, relative_pose):
        """
        Args:
            curr_idx (int): Index of the current frame.
            matched_idx (int): Index of the matched (loop) frame.
            relative_pose (tuple): (R, t) relative transformation.
        """
        self.curr_idx = curr_idx
        self.matched_idx = matched_idx
        self.relative_pose = relative_pose  # (R, t)