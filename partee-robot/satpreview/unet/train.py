# train.py DATA OUT.pt : U-Net (resnet18, ImageNet start) on satellite crops of our checked courses
import sys, os, glob, time, random, numpy as np, cv2, torch, segmentation_models_pytorch as smp
data, outp = sys.argv[1:3]; torch.set_num_threads(os.cpu_count()); random.seed(0)
keys = sorted(os.path.basename(p)[:-4] for p in glob.glob(data + '/*.png')); random.shuffle(keys)
val = keys[:8]; tr = keys[8:]; print(len(tr), 'train', len(val), 'val', flush=True)
IM = {k: cv2.cvtColor(cv2.imread(f'{data}/{k}.jpg'), cv2.COLOR_BGR2RGB) for k in keys}; LB = {k: cv2.imread(f'{data}/{k}.png', 0) for k in keys}
MEAN = np.array([.485, .456, .406]) * 255; STD = np.array([.229, .224, .225]) * 255; S = int(os.environ.get('CROP', '256'))
def crop(k, aug=True):
    im, lb = IM[k], LB[k]; H, W = lb.shape
    for _ in range(20):
        y, x = random.randrange(0, H - S), random.randrange(0, W - S); l = lb[y:y + S, x:x + S]
        if (l != 255).mean() > .3: break
    i = im[y:y + S, x:x + S].astype(np.float32)
    if aug:
        r = random.randrange(4); i = np.rot90(i, r); l = np.rot90(l, r)
        if random.random() < .5: i = i[:, ::-1]; l = l[:, ::-1]
        i = i * random.uniform(.85, 1.15) + random.uniform(-12, 12)   # brightness/contrast: different photo dates
    return ((i - MEAN) / STD).transpose(2, 0, 1).copy(), l.astype(np.int64).copy()
def batch(ks, n, aug=True):
    xs, ys = zip(*[crop(random.choice(ks), aug) for _ in range(n)]); return torch.tensor(np.stack(xs), dtype=torch.float32), torch.tensor(np.stack(ys))
m = smp.Unet('resnet18', encoder_weights='imagenet', classes=8)
w = torch.tensor([1, 2, 3, 3, 2, 1, 3, 1], dtype=torch.float32)   # small things (greens, bunkers, tees) count more
lossf = torch.nn.CrossEntropyLoss(weight=w, ignore_index=255)
opt = torch.optim.AdamW(m.parameters(), 1e-3, weight_decay=1e-4); IT = int(os.environ.get('ITERS', '2500'))
sch = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=IT); t0 = time.time(); random.seed(1); VX, VY = batch(val, 48, False)
for it in range(IT):
    m.train(); x, y = batch(tr, 8); loss = lossf(m(x), y); opt.zero_grad(); loss.backward(); opt.step(); sch.step()
    if it % 250 == 0 or it == IT - 1:
        m.eval()
        with torch.no_grad(): p = torch.cat([m(VX[i:i + 8]).argmax(1) for i in range(0, 48, 8)])
        ok = VY != 255; iou = []
        for c in range(8):
            a, b = (p == c) & ok, (VY == c) & ok; u = (a | b).sum().item(); iou.append(round((a & b).sum().item() / u, 2) if u else None)
        print(it, round(loss.item(), 3), 'val IoU', dict(zip(['rough', 'fairway', 'green', 'bunker', 'water', 'trees', 'tee', 'built'], iou)), int(time.time() - t0), 's', flush=True)
        torch.save({k: v.half() for k, v in m.state_dict().items()}, outp)
