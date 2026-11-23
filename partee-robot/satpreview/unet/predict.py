# predict.py : class map for a satellite photo with the trained model (sliding window, overlapping tiles)
import os, numpy as np, cv2, torch, segmentation_models_pytorch as smp
UMPP = 0.5; _m = None
def model():
    global _m
    if _m is None:
        _m = smp.Unet('resnet18', encoder_weights=None, classes=8)
        _m.load_state_dict({k: v.float() for k, v in torch.load(os.environ['MODEL'], map_location='cpu').items()}); _m.eval(); torch.set_num_threads(os.cpu_count())
    return _m
def classify(im_bgr, mpp):
    H, W = im_bgr.shape[:2]; im = cv2.resize(im_bgr, (int(W * mpp / UMPP), int(H * mpp / UMPP)), interpolation=cv2.INTER_AREA)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB).astype(np.float32); h, w = im.shape[:2]
    x = (im - np.array([.485, .456, .406]) * 255) / (np.array([.229, .224, .225]) * 255)
    T, O = 512, 64; P = np.zeros((8, h, w), np.float32); N = np.zeros((h, w), np.float32)
    ph, pw = max(h, T), max(w, T); xp = np.zeros((ph, pw, 3), np.float32); xp[:h, :w] = x
    ys = list(range(0, ph - T + 1, T - 2 * O)) or [0]; xs = list(range(0, pw - T + 1, T - 2 * O)) or [0]
    if ys[-1] != ph - T: ys.append(ph - T)
    if xs[-1] != pw - T: xs.append(pw - T)
    with torch.no_grad():
        for y in ys:
            for x0 in xs:
                t = torch.tensor(xp[y:y + T, x0:x0 + T].transpose(2, 0, 1)[None].copy())
                p = sum(torch.rot90(torch.softmax(model()(torch.rot90(t, r, (2, 3))), 1), -r, (2, 3)) for r in range(2))[0].numpy()
                yy, xx = min(T, h - y), min(T, w - x0)
                P[:, y:y + yy, x0:x0 + xx] += p[:, :yy, :xx]; N[y:y + yy, x0:x0 + xx] += 1
    c = (P / np.maximum(N, 1)).argmax(0).astype(np.uint8)
    return cv2.resize(c, (W, H), interpolation=cv2.INTER_NEAREST)
