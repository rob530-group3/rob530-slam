import cv2

def initialize_feature_detector(settings):
    algo = settings.get("feature_algorithm", "ORB")
    matcher_combo = settings.get("matcher_type", "BF_crosscheck")

    # === Select feature extractor ===
    if algo == "ORB":
        detector = cv2.ORB_create(2000)
        descriptor_type = "binary"
        norm_type = cv2.NORM_HAMMING
    elif algo == "SIFT":
        detector = cv2.SIFT_create()
        descriptor_type = "float"
        norm_type = cv2.NORM_L2
    elif algo == "SURF":
        detector = cv2.xfeatures2d.SURF_create()
        descriptor_type = "float"
        norm_type = cv2.NORM_L2
    elif algo == "AKAZE":
        detector = cv2.AKAZE_create()
        descriptor_type = "binary"
        norm_type = cv2.NORM_HAMMING
    else:
        raise ValueError(f"Unsupported feature_algorithm: {algo}")

    # === Select matcher and strategy ===
    if matcher_combo == "BF_crosscheck":
        matcher = cv2.BFMatcher(norm_type, crossCheck=True)
        strategy = "crosscheck"
    elif matcher_combo == "BF_KNN":
        matcher = cv2.BFMatcher(norm_type, crossCheck=False)
        strategy = "KNN"
    elif matcher_combo == "FLANN_KNN":
        if descriptor_type == "float":
            index_params = dict(algorithm=1, trees=5)  # KDTree
        else:  # binary descriptor: use LSH
            index_params = dict(algorithm=6, table_number=6, key_size=12, multi_probe_level=1)
        matcher = cv2.FlannBasedMatcher(index_params, dict())
        strategy = "KNN"
    else:
        raise ValueError(f"Unsupported matcher_type: {matcher_combo}")

    return detector, matcher, strategy

def match_features(matcher, des1, des2, strategy):
    if strategy == "KNN":
        matches = matcher.knnMatch(des1, des2, k=2)
        good_matches = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)
        return good_matches
    elif strategy == "crosscheck":
        matches = matcher.match(des1, des2)
        return sorted(matches, key=lambda x: x.distance)
    else:
        raise ValueError(f"Unknown matching strategy: {strategy}")

