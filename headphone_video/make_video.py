import cv2, numpy as np, subprocess, wave, math, sys
from PIL import Image, ImageDraw, ImageFont

W = H = 1080
FPS = 30
INTRO = 0.8
SEG = 2.2
NSEG = 4
ROT_END = INTRO + SEG * NSEG
IRIS_END = ROT_END + 1.0
CARTOON_END = IRIS_END + 5.2
TOTAL = CARTOON_END + 1.3
NFRAMES = int(TOTAL * FPS)

FB = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FS = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FSER = "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf"

def load_product(path, target_h=720, mirror=False):
    im = cv2.imread(path).astype(np.float32)
    im = np.clip(im * (255.0 / 245.0), 0, 255)
    g = im.mean(axis=2)
    ys, xs = np.where(g < 235)
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    im = im[y0:y1 + 1, x0:x1 + 1]
    if mirror:
        im = im[:, ::-1]
    s = target_h / im.shape[0]
    max_w = 780
    if im.shape[1] * s > max_w:
        s = max_w / im.shape[1]
    im = cv2.resize(im, (int(im.shape[1] * s), int(im.shape[0] * s)), interpolation=cv2.INTER_LANCZOS4)
    return im / 255.0

orig = load_product('assets/original_clean.png', 640)
side = load_product('assets/view_side.jpg', 680)
back = load_product('assets/view_back.jpg', 640)
side_m = load_product('assets/view_side.jpg', 680, mirror=True)
KEYS = [orig, side, back, side_m, orig]

cartoon = cv2.imread('assets/cartoon_wearing.jpg')
cartoon = cv2.cvtColor(cv2.resize(cartoon, (W, H), interpolation=cv2.INTER_LANCZOS4), cv2.COLOR_BGR2RGB).astype(np.float32)

yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
r = np.sqrt((xx - W / 2) ** 2 + (yy - H * 0.47) ** 2) / (W * 0.75)
inner = np.array([246, 244, 240], np.float32)
outer = np.array([196, 205, 222], np.float32)
BG = (inner[None, None] * (1 - np.clip(r, 0, 1)[..., None] ** 1.3) + outer[None, None] * np.clip(r, 0, 1)[..., None] ** 1.3) / 255.0
FLOOR = np.exp(-(((yy - 905) / 90) ** 2) - (((xx - W / 2) / 420) ** 2)).astype(np.float32)
BG = np.clip(BG + FLOOR[..., None] * 0.06, 0, 1)

def smooth(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)

def ease_out_back(x):
    x = np.clip(x, 0, 1)
    c1, c3 = 1.70158, 2.70158
    return 1 + c3 * (x - 1) ** 3 + c1 * (x - 1) ** 2

def place(canvas_white, img, cx, cy, sx=1.0, sy=1.0, alpha=1.0):
    """Multiply-blend product (white bg) onto canvas at center with scales."""
    h, w = img.shape[:2]
    nw, nh = max(2, int(w * sx)), max(2, int(h * sy))
    r_ = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA if sx < 1 else cv2.INTER_LINEAR)
    x0, y0 = int(cx - nw / 2), int(cy - nh / 2)
    xa, ya = max(0, x0), max(0, y0)
    xb, yb = min(W, x0 + nw), min(H, y0 + nh)
    if xb <= xa or yb <= ya:
        return
    patch = r_[ya - y0:yb - y0, xa - x0:xb - x0]
    dst = canvas_white[ya:yb, xa:xb]
    mixed = 1 - alpha * (1 - patch)
    dst *= mixed

def font(path, size):
    return ImageFont.truetype(path, size)

def text_c(d, xy, s, f, fill, anchor="mm"):
    d.text(xy, s, font=f, fill=fill, anchor=anchor)

def product_scene(t):
    prod = np.ones((H, W, 3), np.float32)
    cy_base = 540
    if t < INTRO:
        k = t / INTRO
        sc = 0.55 + 0.45 * ease_out_back(k)
        place(prod, orig, W / 2, cy_base, sc, sc, alpha=smooth(k * 1.6))
        w_sq = sc
        prog = 0.0
    else:
        tr = min(t - INTRO, SEG * NSEG - 1e-6)
        seg = int(tr // SEG)
        u = (tr - seg * SEG) / SEG
        e = smooth(u)
        w_sq = 1 - 0.38 * math.sin(math.pi * e)
        a_new = smooth((e - 0.30) / 0.40)
        bob = 10 * math.sin(2 * math.pi * (t - INTRO) / SEG * 0.5)
        A, B = KEYS[seg], KEYS[seg + 1]
        shift = 70 * math.sin(math.pi * e)
        place(prod, A, W / 2 - shift * e, cy_base + bob, w_sq, 1.0, alpha=1 - a_new)
        place(prod, B, W / 2 + shift * (1 - e), cy_base + bob, w_sq, 1.0, alpha=a_new)
        prog = (tr / (SEG * NSEG))
    shadow = np.zeros((H, W), np.float32)
    cv2.ellipse(shadow, (W // 2, 905), (int(250 * (0.55 + 0.45 * w_sq)), 26), 0, 0, 360, 1.0, -1)
    shadow = cv2.GaussianBlur(shadow, (0, 0), 22)
    frame = BG * (1 - 0.28 * shadow[..., None])
    frame = frame * prod
    return frame, prog

def overlay_product_ui(frame, t, prog):
    img = Image.fromarray(np.clip(frame * 255, 0, 255).astype(np.uint8))
    S = 2
    layer = Image.new("RGBA", (W * S, H * S), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    fa = int(255 * smooth(t / 0.5))
    text_c(d, (W * S // 2, 95 * S), "hoco.", font(FSER, 84 * S), (30, 30, 34, fa))
    text_c(d, (W * S // 2, 170 * S), "WIRELESS  HEADPHONE", font(FB, 28 * S), (90, 96, 110, fa))
    if t >= INTRO:
        cx, cy, rad = W * S // 2, 975 * S, 52 * S
        d.ellipse([cx - rad, cy - rad, cx + rad, cy + rad], outline=(170, 178, 195, 255), width=7 * S)
        d.arc([cx - rad, cy - rad, cx + rad, cy + rad], -90, -90 + 360 * prog, fill=(40, 120, 255, 255), width=7 * S)
        text_c(d, (cx, cy), "360\u00b0", font(FB, 30 * S), (30, 30, 34, 255))
        names = ["FRONT VIEW", "SIDE VIEW", "BACK VIEW", "OTHER SIDE", "FRONT VIEW"]
        trr = min(t - INTRO, SEG * NSEG - 1e-6)
        lbl = names[int(round(trr / SEG))]
        text_c(d, (cx + 330 * S, cy), lbl, font(FB, 26 * S), (90, 96, 110, 255))
        text_c(d, (cx - 330 * S, cy), "360\u00b0 VIEW", font(FB, 26 * S), (90, 96, 110, 255))
    layer = layer.resize((W, H), Image.LANCZOS)
    img.paste(layer, (0, 0), layer)
    return np.asarray(img).astype(np.float32)

BPM = 120.0
def beat_pulse(t, t0):
    ph = ((t - t0) * BPM / 60.0) % 1.0
    return math.exp(-5.0 * ph)

def cartoon_scene(t):
    tc = t - IRIS_END
    pulse = beat_pulse(t, IRIS_END)
    ent = ease_out_back(min(1, (t - ROT_END) / 1.1)) if t >= ROT_END else 0
    sc = 0.86 * (1.0 + 0.03 * pulse) * (1.18 - 0.18 * smooth((t - ROT_END) / 1.2))
    ang = 2.4 * math.sin(2 * math.pi * tc / 2.0)
    ty = -10 * pulse
    M = cv2.getRotationMatrix2D((W / 2, H * 0.95), ang, sc)
    M[1, 2] += ty
    fr = cv2.warpAffine(cartoon, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return fr

NOTES = [(0.14, 0.0, "\u266a", (255, 255, 255)), (0.86, 0.35, "\u266b", (255, 214, 64)),
         (0.07, 0.7, "\u266b", (255, 120, 150)), (0.93, 1.05, "\u266a", (255, 255, 255)),
         (0.22, 1.4, "\u266a", (120, 210, 255)), (0.80, 1.75, "\u266b", (255, 255, 255))]

def overlay_cartoon_ui(frame, t):
    img = Image.fromarray(np.clip(frame, 0, 255).astype(np.uint8))
    S = 2
    layer = Image.new("RGBA", (W * S, H * S), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    tc = t - IRIS_END
    if tc > 0:
        for i, (fx, off, ch, col) in enumerate(NOTES):
            life = ((tc * 0.55 + off) % 2.2) / 2.2
            x = fx * W + 22 * math.sin(life * 9 + i)
            y = H * 0.62 - life * H * 0.5
            a = int(255 * math.sin(math.pi * life))
            text_c(d, (x * S, y * S), ch, font(FS, int((70 + 20 * math.sin(life * 6 + i)) * S)), col + (a,))
        k = smooth(tc / 0.6)
        pop = ease_out_back(tc / 0.7)
        fs = int(70 * S * max(0.01, pop))
        if fs > 4:
            text_c(d, (W * S // 2 + 3 * S, 95 * S + 4 * S), "Feel The Music", font(FB, fs), (20, 40, 80, int(110 * k)))
            text_c(d, (W * S // 2, 95 * S), "Feel The Music", font(FB, fs), (255, 255, 255, int(255 * k)))
        text_c(d, (W * S // 2 + 2 * S, 1012 * S + 3 * S), "hoco.  |  Wear it. Love it.", font(FB, 34 * S), (60, 40, 10, int(150 * k)))
        text_c(d, (W * S // 2, 1012 * S), "hoco.  |  Wear it. Love it.", font(FB, 34 * S), (255, 255, 255, int(255 * k)))
    layer = layer.resize((W, H), Image.LANCZOS)
    img.paste(layer, (0, 0), layer)
    return np.asarray(img).astype(np.float32)

def render(t):
    if t < ROT_END + 0.0:
        f, prog = product_scene(t)
        return overlay_product_ui(f, t, prog)
    f_prod, prog = product_scene(ROT_END - 1e-3)
    f_prod = overlay_product_ui(f_prod, ROT_END - 1e-3, 1.0)
    if t >= IRIS_END:
        out = overlay_cartoon_ui(cartoon_scene(t), t)
    else:
        k = smooth((t - ROT_END) / (IRIS_END - ROT_END))
        # product scene gently zooms as the iris opens onto the cartoon
        zs = 1 + 0.12 * k
        Mz = cv2.getRotationMatrix2D((W / 2, H / 2), 0, zs)
        f_prod = cv2.warpAffine(f_prod, Mz, (W, H), borderMode=cv2.BORDER_REPLICATE)
        c = overlay_cartoon_ui(cartoon_scene(t), t)
        rad = k * (W * 0.78)
        mask = np.zeros((H, W), np.float32)
        cv2.circle(mask, (W // 2, int(H * 0.45)), int(rad), 1.0, -1)
        mask = cv2.GaussianBlur(mask, (0, 0), 3)[..., None]
        out = f_prod * (1 - mask) + c * mask
        ring = np.zeros((H, W), np.float32)
        cv2.circle(ring, (W // 2, int(H * 0.45)), int(rad), 1.0, 10)
        ring = cv2.GaussianBlur(ring, (0, 0), 2)[..., None] * (1 - k) * 0.9
        out = out * (1 - ring) + 255 * ring
    if t > CARTOON_END:
        k = smooth((t - CARTOON_END) / 1.0)
        out = out * (1 - 0.35 * k)
        img = Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))
        d = ImageDraw.Draw(img)
        text_c(d, (W // 2, H // 2 - 20), "hoco.", font(FSER, 190), (255, 255, 255))
        text_c(d, (W // 2, H // 2 + 110), "Wireless Headphone", font(FB, 46), (255, 255, 255))
        a = np.asarray(img).astype(np.float32)
        out = out * (1 - k) + a * k
    return out

def make_audio(path, dur, sr=44100):
    n = int(dur * sr)
    t = np.arange(n) / sr
    y = np.zeros(n)
    beat = 60.0 / BPM
    def add(start, sig):
        i = int(start * sr)
        j = min(n, i + len(sig))
        if i < n:
            y[i:j] += sig[:j - i]
    kt = np.arange(int(0.25 * sr)) / sr
    kick = np.sin(2 * np.pi * (50 + 110 * np.exp(-kt * 28)) * kt) * np.exp(-kt * 11)
    ht = np.arange(int(0.06 * sr)) / sr
    rng = np.random.default_rng(1)
    hat = rng.standard_normal(len(ht)) * np.exp(-ht * 90) * 0.25
    hat = np.diff(hat, prepend=0)
    def tone(f, d, vol, wave='tri'):
        tt = np.arange(int(d * sr)) / sr
        if wave == 'tri':
            s = 2 / np.pi * np.arcsin(np.sin(2 * np.pi * f * tt))
        else:
            s = np.sin(2 * np.pi * f * tt)
        env = np.minimum(1, tt * 80) * np.exp(-tt * 5)
        return s * env * vol
    bass = [55.0, 55.0, 65.41, 73.42]
    arp_notes = [[220.0, 261.63, 329.63], [220.0, 261.63, 329.63], [261.63, 329.63, 392.0], [293.66, 349.23, 440.0]]
    nb = int(dur / beat) + 1
    for b in range(nb):
        st = b * beat
        if st >= 0.8:
            add(st, kick * 0.9)
        add(st + beat / 2, hat)
        bar = (b // 4) % 4
        add(st, tone(bass[bar], beat * 0.9, 0.35, 'sin'))
        for k in range(2):
            f = arp_notes[bar][(b * 2 + k) % 3] * 2
            add(st + k * beat / 2, tone(f, beat * 0.5, 0.16))
        if st >= IRIS_END - 0.5:
            mel = [880, 784, 659.25, 784][b % 4] if (b // 4) % 2 == 0 else [659.25, 784, 880, 987.77][b % 4]
            add(st, tone(mel, beat * 0.9, 0.12, 'sin'))
    fade = np.minimum(1, t / 0.4) * np.minimum(1, (dur - t) / 1.0)
    y *= fade
    y = np.tanh(y * 1.3) * 0.8
    with wave.open(path, 'wb') as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(sr)
        wf.writeframes((y * 32767).astype(np.int16).tobytes())

if __name__ == "__main__":
    only = None
    if len(sys.argv) > 1 and sys.argv[1] == "stills":
        for tt in [0.5, 2.0, 3.3, 4.6, 6.6, 8.8, 10.1, 10.6, 12.5, 15.8]:
            cv2.imwrite(f"/tmp/still_{tt:05.2f}.png", cv2.cvtColor(np.clip(render(tt), 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
        sys.exit()
    make_audio("/tmp/music.wav", TOTAL)
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", str(FPS),
           "-i", "-", "-i", "/tmp/music.wav", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "17", "-preset", "medium",
           "-c:a", "aac", "-b:a", "160k", "-shortest", "-movflags", "+faststart", "headphone_360_promo.mp4"]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for i in range(NFRAMES):
        fr = render(i / FPS)
        p.stdin.write(cv2.cvtColor(np.clip(fr, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR).tobytes())
    p.stdin.close(); p.wait()
    print("done", NFRAMES, "frames")
