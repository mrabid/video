import cv2, numpy as np
im = cv2.imread('assets/original.png')
g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
white = (g >= 249).astype(np.uint8)
n, lab, stats, _ = cv2.connectedComponentsWithStats(white)
bg = np.zeros_like(white)
for i in range(1, n):
    if stats[i, cv2.CC_STAT_AREA] > 20000:
        bg[lab == i] = 1
obj = (bg == 0).astype(np.uint8)
obj = cv2.morphologyEx(obj, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
n, lab, stats, _ = cv2.connectedComponentsWithStats(obj)
keep = np.zeros_like(obj)
for i in range(1, n):
    if stats[i, cv2.CC_STAT_AREA] > 8000:
        keep[lab == i] = 1
keep = cv2.morphologyEx(keep, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11)))
keep = cv2.dilate(keep, np.ones((3, 3), np.uint8), iterations=1)
out = im.copy()
out[keep == 0] = 255
cv2.imwrite('assets/original_clean.png', out)
cv2.imwrite('/tmp/mask.png', keep * 255)

out = cv2.imread('assets/original_clean.png')
g2 = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(g2, (0, 0), 5)
m = np.zeros_like(g2)
y0, y1, x0, x1 = 636, 676, 385, 500
d = g2.astype(int) - blur.astype(int)
sub = (np.abs(d[y0:y1, x0:x1]) > 4) & (keep[y0:y1, x0:x1] == 1)
m[y0:y1, x0:x1][sub] = 255
m = cv2.dilate(m, np.ones((3, 3), np.uint8), iterations=2)
out = cv2.inpaint(out, m, 5, cv2.INPAINT_TELEA)
cv2.imwrite('assets/original_clean.png', out)
cv2.imwrite('/tmp/c2.png', cv2.resize(out[600:700, 340:540], None, fx=4, fy=4))

out = cv2.imread('assets/original_clean.png')
g3 = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
for (ya, yb, xa, xb) in [(676, 710, 340, 470), (655, 690, 400, 480)]:
    sub = g3[ya:yb, xa:xb] > 170
    sub = cv2.dilate(sub.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    out[ya:yb, xa:xb][sub] = 255
cv2.imwrite('assets/original_clean.png', out)
cv2.imwrite('/tmp/zoom_cup3.png', cv2.resize(out[560:720, 300:620], None, fx=2.5, fy=2.5))
