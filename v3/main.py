
import argparse, json, os, math, glob
import numpy as np
import cv2 as cv

SUPPORTED_EXT = (".jpg",".jpeg",".png",".bmp",".tif",".tiff",".webp")

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def save_if(flag, path, img):
    if flag:
        cv.imwrite(path, img)

def to_gray(img):
    if img.ndim == 2: return img
    return cv.cvtColor(img, cv.COLOR_BGR2GRAY)

def illumination_correct(gray):
    gray_f = gray.astype(np.float32)
    k = max(31, int(min(gray.shape[:2]) * 0.1) | 1)
    bg = cv.GaussianBlur(gray_f, (k,k), 0)
    norm = (gray_f / (bg + 1e-6))
    norm = cv.normalize(norm, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX)
    norm = norm.astype(np.uint8)
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    eq = clahe.apply(norm)
    return eq

def adaptive_canny(gray):
    v = np.median(gray)
    low = int(max(0, 0.66*v))
    high = int(min(255, 1.33*v))
    return cv.Canny(gray, low, high, L2gradient=True)

def angle_from_lines(lines):
    if lines is None: return 0.0
    angs = []
    for l in lines:
        x1,y1,x2,y2 = l[0]
        ang = math.atan2(y2-y1, x2-x1)
        if ang < -math.pi/2: ang += math.pi
        if ang >=  math.pi/2: ang -= math.pi
        angs.append(ang)
    if not angs: return 0.0
    return float(np.median(angs))

def rotate_image(img, angle_rad):
    angle_deg = angle_rad * 180/np.pi
    (h,w) = img.shape[:2]
    M = cv.getRotationMatrix2D((w//2,h//2), angle_deg, 1.0)
    return cv.warpAffine(img, M, (w,h), flags=cv.INTER_CUBIC, borderMode=cv.BORDER_REPLICATE)

def rectify_with_grid(img, save_steps=False, outdir=None):
    gray = to_gray(img)
    g = illumination_correct(gray)
    edges = adaptive_canny(g)
    save_if(save_steps and outdir, os.path.join(outdir, "debug_edges.png"), edges)
    min_len = max(20, min(img.shape[:2])//5)
    lines = cv.HoughLinesP(edges, 1, np.pi/180, threshold=120, minLineLength=min_len, maxLineGap=12)
    ang = angle_from_lines(lines)
    rect = rotate_image(img, -ang)
    return rect, -ang

def suppress_grid_for_circles(gray):
    edges = adaptive_canny(gray)
    k = max(15, min(gray.shape[:2])//35)
    kernel_h = cv.getStructuringElement(cv.MORPH_RECT, (k,1))
    kernel_v = cv.getStructuringElement(cv.MORPH_RECT, (1,k))
    lines_h = cv.morphologyEx(edges, cv.MORPH_OPEN, kernel_h)
    lines_v = cv.morphologyEx(edges, cv.MORPH_OPEN, kernel_v)
    lines = cv.bitwise_or(lines_h, lines_v)
    no_lines = cv.bitwise_and(edges, cv.bitwise_not(lines))
    return no_lines, edges

def score_circle_by_gradient(gray, circle):
    cx, cy, r = circle
    h, w = gray.shape[:2]
    gx, gy = np.gradient(cv.GaussianBlur(gray, (5,5), 0).astype(np.float32))
    mag = np.hypot(gx, gy)
    theta = np.linspace(0, 2*np.pi, 360, endpoint=False)
    xs = np.clip((cx + r*np.cos(theta)).astype(int), 0, w-1)
    ys = np.clip((cy + r*np.sin(theta)).astype(int), 0, h-1)
    vals = mag[ys, xs]
    score = float(np.mean(vals))
    support = float(np.mean(vals > (np.percentile(vals, 60))))
    return score, support

def hough_multi_try(gray, minr, maxr):
    blur = cv.GaussianBlur(gray, (5,5), 0)
    tries = [
        (1.2, 120, 30),
        (1.2, 110, 28),
        (1.2, 100, 26),
        (1.4, 120, 28),
    ]
    best = None
    best_score = -1
    best_support = 0.0
    chosen = None
    for dp, p1, p2 in tries:
        circles = cv.HoughCircles(blur, cv.HOUGH_GRADIENT, dp=dp, minDist=int(0.8*min(gray.shape[:2])),
                                  param1=p1, param2=p2, minRadius=minr, maxRadius=maxr)
        if circles is None: continue
        for c in np.around(circles[0]).astype(np.float32):
            s, sup = score_circle_by_gradient(gray, c)
            if s > best_score:
                best_score = s
                best_support = sup
                best = c
                chosen = (dp, p1, p2)
    return best, best_score, best_support, chosen

def refine_hough(gray, seed_circle):
    if seed_circle is None: return None, None, None, None
    cx, cy, r = seed_circle
    minr = int(max(10, r*0.7))
    maxr = int(r*1.3)
    c, sc, sup, ch = hough_multi_try(gray, minr, maxr)
    if c is None:
        return seed_circle, None, None, None
    return c, sc, sup, ch

def contour_fallback(gray, edges_wo_grid):
    cnts, _ = cv.findContours(edges_wo_grid, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if not cnts: return None, 0.0
    best = None; best_score = 0.0
    h, w = gray.shape[:2]
    for c in cnts:
        area = cv.contourArea(c)
        if area < (0.001*h*w): continue
        perim = cv.arcLength(c, True)+1e-6
        circ = 4*np.pi*area/(perim*perim)
        (cx, cy), r = cv.minEnclosingCircle(c)
        score = circ * np.sqrt(area)
        if score > best_score:
            best_score = score
            best = (float(cx), float(cy), float(r))
    return best, best_score

def detect_circle_robust(gray, minr=None, maxr=None, save_steps=False, outdir=None):
    grayc = illumination_correct(gray)
    mask_no_lines, edges = suppress_grid_for_circles(grayc)
    if save_steps and outdir:
        cv.imwrite(os.path.join(outdir, "debug_edges_no_lines.png"), mask_no_lines)

    h, w = gray.shape[:2]
    if minr is None: minr = max(10, int(0.05*min(h,w)))
    if maxr is None: maxr = int(0.48*min(h,w))

    c0, sc0, sup0, ch0 = hough_multi_try(grayc, minr, maxr)
    if c0 is not None:
        c1, sc1, sup1, ch1 = refine_hough(grayc, c0)
        if c1 is not None:
            conf = float(min(1.0, 0.5*sup1 + 0.5*(sc1/(sc0+1e-6+sc1))))
            return {"circle": tuple(map(float,c1)), "confidence": conf, "method": "hough_refined"}
        else:
            conf = float(min(1.0, 0.4*sup0 + 0.6))
            return {"circle": tuple(map(float,c0)), "confidence": conf, "method": "hough_coarse"}

    cF, sF = contour_fallback(grayc, mask_no_lines)
    if cF is not None:
        cx, cy, r = cF
        theta = np.linspace(0, 2*np.pi, 360, endpoint=False)
        xs = np.clip((cx + r*np.cos(theta)).astype(int), 0, w-1)
        ys = np.clip((cy + r*np.sin(theta)).astype(int), 0, h-1)
        support = float(np.mean(mask_no_lines[ys, xs] > 0))
        conf = float(min(0.85, 0.5*support + 0.35))
        return {"circle": (float(cx), float(cy), float(r)), "confidence": conf, "method": "contour_fallback"}

    return {"circle": None, "confidence": 0.0, "method": "none"}

def process_one(image_path, outdir, minr=None, maxr=None, scale=1.0, save_steps=False):
    ensure_dir(outdir)
    img = cv.imread(image_path, cv.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(image_path)
    if scale != 1.0:
        img = cv.resize(img, (int(img.shape[1]*scale), int(img.shape[0]*scale)), interpolation=cv.INTER_AREA)

    rect, angle = rectify_with_grid(img, save_steps=save_steps, outdir=outdir)
    cv.imwrite(os.path.join(outdir, "rectified.png"), rect)

    gray = to_gray(rect)
    det = detect_circle_robust(gray, minr=minr, maxr=maxr, save_steps=save_steps, outdir=outdir)

    result = {"center": None, "radius": None, "confidence": float(det["confidence"]), "method": det["method"]}
    overlay = rect.copy()

    if det["circle"] is not None:
        cx, cy, r = det["circle"]
        cv.circle(overlay, (int(cx), int(cy)), int(r), (0,255,0), 2)
        cv.circle(overlay, (int(cx), int(cy)), 3, (0,0,255), -1)
        result["center"] = [float(cx), float(cy)]
        result["radius"] = float(r)

    cv.imwrite(os.path.join(outdir, "overlay.png"), overlay)
    with open(os.path.join(outdir, "result.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    return result

def process_path(path, outdir, **kwargs):
    if os.path.isdir(path):
        ensure_dir(outdir)
        images = [os.path.join(path, f) for f in os.listdir(path) if os.path.splitext(f.lower())[1] in SUPPORTED_EXT]
        summary = []
        for p in images:
            name = os.path.splitext(os.path.basename(p))[0]
            sub = os.path.join(outdir, name)
            ensure_dir(sub)
            res = process_one(p, sub, **kwargs)
            res["image"] = p
            summary.append(res)
        with open(os.path.join(outdir, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        return {"batch": True, "count": len(images)}
    else:
        return process_one(path, outdir, **kwargs)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("image", help="Path to image OR folder")
    ap.add_argument("--outdir", default="out")
    ap.add_argument("--minr", type=int, default=None)
    ap.add_argument("--maxr", type=int, default=None)
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--save-steps", action="store_true", help="Save debug intermediate steps")
    args = ap.parse_args()
    res = process_path(args.image, args.outdir, minr=args.minr, maxr=args.maxr, scale=args.scale, save_steps=args.save_steps)
    print(json.dumps(res, indent=2))
