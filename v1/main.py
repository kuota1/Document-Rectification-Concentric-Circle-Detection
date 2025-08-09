
import argparse, json, os, math
import numpy as np
import cv2 as cv

def order_quad(pts):
    # pts: Nx2
    rect = np.zeros((4,2), dtype=np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).reshape(-1)
    rect[0] = pts[np.argmin(s)]      # top-left
    rect[2] = pts[np.argmax(s)]      # bottom-right
    rect[1] = pts[np.argmin(diff)]   # top-right
    rect[3] = pts[np.argmax(diff)]   # bottom-left
    return rect

def find_doc_quad(gray):
    # Canny + contours -> largest 4-pt polygon
    edges = cv.Canny(gray, 0, 0)  # thresholds adapted later
    contours, _ = cv.findContours(edges, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    best = None; best_area = 0
    for c in contours:
        peri = cv.arcLength(c, True)
        approx = cv.approxPolyDP(c, 0.02*peri, True)
        if len(approx) == 4:
            area = cv.contourArea(approx)
            if area > best_area:
                best_area = area; best = approx
    if best is None:
        return None
    quad = best.reshape(-1,2).astype(np.float32)
    return order_quad(quad)

def warp_to_rect(img, quad, target_w=None):
    # Compute destination rectangle preserving aspect from quad
    (tl, tr, br, bl) = quad
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    w = int(max(widthA, widthB))
    h = int(max(heightA, heightB))
    if target_w:
        scale = target_w / w
        w = int(target_w); h = int(h*scale)
    dst = np.array([[0,0],[w-1,0],[w-1,h-1],[0,h-1]], dtype=np.float32)
    H = cv.getPerspectiveTransform(quad, dst)
    rectified = cv.warpPerspective(img, H, (w,h), flags=cv.INTER_CUBIC)
    return rectified, H

def deskew_by_lines(gray):
    edges = cv.Canny(gray, 0, 0)
    lines = cv.HoughLines(edges, 1, np.pi/180, 150)
    if lines is None: return 0.0
    angles = []
    for rho_theta in lines:
        rho, theta = rho_theta[0]
        angle = theta - np.pi/2.0
        angles.append(angle)
    if not angles: return 0.0
    angle = np.median(angles)
    return angle

def rotate_image(img, angle_rad):
    angle = angle_rad * 180/np.pi
    (h,w) = img.shape[:2]
    M = cv.getRotationMatrix2D((w//2,h//2), angle, 1.0)
    return cv.warpAffine(img, M, (w,h), flags=cv.INTER_CUBIC, borderMode=cv.BORDER_REPLICATE)

def suppress_grid_for_circles(gray):
    # Remove long straight lines using morphological opening with linear structuring elements
    edges = cv.Canny(gray, 0, 0)
    # Horizontal
    kernel_h = cv.getStructuringElement(cv.MORPH_RECT, (25,1))
    lines_h = cv.morphologyEx(edges, cv.MORPH_OPEN, kernel_h)
    # Vertical
    kernel_v = cv.getStructuringElement(cv.MORPH_RECT, (1,25))
    lines_v = cv.morphologyEx(edges, cv.MORPH_OPEN, kernel_v)
    lines = cv.bitwise_or(lines_h, lines_v)
    no_lines = cv.bitwise_and(edges, cv.bitwise_not(lines))
    return no_lines

def adaptive_canny(gray):
    v = np.median(gray)
    lo = int(max(0, 0.66*v))
    hi = int(min(255, 1.33*v))
    return cv.Canny(gray, lo, hi)

def detect_concentric(gray, overlay_img):
    # Pre-emphasis
    blur = cv.GaussianBlur(gray, (5,5), 0)
    edges = adaptive_canny(blur)
    edges = suppress_grid_for_circles(gray)

    # HoughCircles seed
    circles = cv.HoughCircles(blur, cv.HOUGH_GRADIENT, dp=1.2, minDist=20,
                              param1=100, param2=30,
                              minRadius=10, maxRadius=0)
    result = {"center": None, "radii": [], "confidence": 0.0}
    if circles is None:
        return result, overlay_img
    circles = np.uint16(np.around(circles[0]))
    # Cluster centers by median
    cx = np.median(circles[:,0]); cy = np.median(circles[:,1])
    dists = np.sqrt((circles[:,0]-cx)**2 + (circles[:,1]-cy)**2)
    inliers = circles[dists < np.percentile(dists, 50)]
    if len(inliers) < 1:
        inliers = circles
    cx = float(np.median(inliers[:,0])); cy = float(np.median(inliers[:,1]))
    radii = sorted([float(r) for r in inliers[:,2]])
    # Draw overlay
    ov = overlay_img.copy()
    for r in radii:
        cv.circle(ov, (int(cx), int(cy)), int(r), (0,255,0), 2)
    cv.circle(ov, (int(cx), int(cy)), 3, (0,0,255), -1)
    # Confidence: ratio inliers/total
    conf = len(inliers)/len(circles)
    result["center"] = [cx, cy]
    result["radii"] = radii
    result["confidence"] = float(conf)
    return result, ov

def process(image_path, outdir):
    os.makedirs(outdir, exist_ok=True)
    img = cv.imread(image_path, cv.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(image_path)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # Try document quad warp; else deskew
    quad = find_doc_quad(gray)
    if quad is not None:
        rectified, H = warp_to_rect(img, quad)
    else:
        angle = deskew_by_lines(gray)
        rectified = rotate_image(img, -angle)
    rect_gray = cv.cvtColor(rectified, cv.COLOR_BGR2GRAY)

    # TODO: affine residual (lattice fitting) – stubbed for PoC
    rectified_out = rectified
    cv.imwrite(os.path.join(outdir, "rectified.png"), rectified_out)

    # Detect concentric circles
    result, overlay = detect_concentric(rect_gray, rectified_out)
    cv.imwrite(os.path.join(outdir, "overlay.png"), overlay)
    with open(os.path.join(outdir, "result.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    return result

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("image", help="input image path")
    ap.add_argument("--outdir", default="out", help="output directory")
    args = ap.parse_args()
    res = process(args.image, args.outdir)
    print(json.dumps(res, indent=2))
