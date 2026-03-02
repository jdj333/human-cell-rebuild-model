import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# Live-updating "Human Cell" procedural renderer with status UI
# ------------------------------------------------------------
# What it does:
# - Uses one Matplotlib figure that updates repeatedly.
# - Shows a "Processing..." status between stages.
# - Renders a stylized cell in layered stages (membrane, nucleus, ER, Golgi,
#   mitochondria network-ish, vesicles/ribosomes).
#
# Works in:
# - Google Colab / Jupyter (best with `%matplotlib inline` or `%matplotlib notebook`)
# - Local Python (matplotlib with interactive backend)
#
# Tip:
# - In Jupyter: run `plt.ion()` is enough for most setups.
# - In Colab: it will still render updates, but sometimes appears as successive frames.

def rng(seed=0):
    return np.random.default_rng(seed)

def gaussian_splat(canvas, x, y, amp=1.0, sigma=2.0):
    """Add a gaussian blob to a 2D canvas at (x,y)."""
    H, W = canvas.shape
    x0 = int(round(x))
    y0 = int(round(y))
    rad = int(max(2, sigma * 3))
    x1 = max(0, x0 - rad); x2 = min(W, x0 + rad + 1)
    y1 = max(0, y0 - rad); y2 = min(H, y0 + rad + 1)
    if x1 >= x2 or y1 >= y2:
        return

    yy, xx = np.mgrid[y1:y2, x1:x2]
    d2 = (xx - x)**2 + (yy - y)**2
    blob = amp * np.exp(-d2 / (2 * sigma**2))
    canvas[y1:y2, x1:x2] += blob

def capsule_mask(W, H, cx, cy, length, radius):
    """Capsule (stadium) mask aligned along x-axis."""
    y, x = np.mgrid[0:H, 0:W]
    x = x - cx
    y = y - cy
    half = length / 2.0
    rect = (np.abs(x) <= half) & (np.abs(y) <= radius)
    left = (x < -half) & ((x + half) ** 2 + y ** 2 <= radius ** 2)
    right = (x > half) & ((x - half) ** 2 + y ** 2 <= radius ** 2)
    return rect | left | right

def circle_mask(W, H, cx, cy, r):
    y, x = np.mgrid[0:H, 0:W]
    return (x - cx)**2 + (y - cy)**2 <= r**2

def draw_polyline(canvas, pts, value=1.0, thickness=2):
    """Rasterize polyline to a float canvas."""
    H, W = canvas.shape
    pts = np.asarray(pts, float)
    if len(pts) < 2:
        return
    r = max(1, int(thickness))
    offsets = [(dy, dx) for dy in range(-r, r+1) for dx in range(-r, r+1)]
    offsets = np.array(offsets, int)

    for i in range(len(pts) - 1):
        x0, y0 = pts[i]
        x1, y1 = pts[i + 1]
        seg_len = np.hypot(x1-x0, y1-y0)
        n = int(max(12, seg_len * 1.8))
        ts = np.linspace(0, 1, n)
        xs = x0 + (x1-x0) * ts
        ys = y0 + (y1-y0) * ts

        for x, y in zip(xs, ys):
            ix = int(round(x)); iy = int(round(y))
            if not (0 <= ix < W and 0 <= iy < H):
                continue
            for dy, dx in offsets:
                xx = ix + dx; yy = iy + dy
                if 0 <= xx < W and 0 <= yy < H:
                    d2 = dx*dx + dy*dy
                    w = np.exp(-d2 / (2 * (r*0.8 + 0.6)**2))
                    canvas[yy, xx] = max(canvas[yy, xx], value * w)

def fractal_branch(polylines, start, direction, step, depth, rstate,
                   turn_sigma=0.35, branch_prob=0.40, shrink=0.78):
    """Recursive branching for mitochondrial-ish tubules."""
    x, y = start
    ang = direction

    seg_pts = [(x, y)]
    seg_len = int(max(8, step * 2.2))
    for _ in range(seg_len):
        ang += rstate.normal(0, turn_sigma) * 0.12
        x += np.cos(ang) * step
        y += np.sin(ang) * step
        seg_pts.append((x, y))

    polylines.append(seg_pts)

    if depth <= 0:
        return

    if rstate.random() < branch_prob:
        bx, by = seg_pts[int(len(seg_pts) * rstate.uniform(0.45, 0.85))]
        delta = rstate.uniform(0.55, 1.05)
        for sign in (-1, 1):
            if rstate.random() < 0.78:
                fractal_branch(
                    polylines, (bx, by), ang + sign*delta,
                    step * shrink, depth - 1, rstate,
                    turn_sigma=turn_sigma,
                    branch_prob=branch_prob * 0.86,
                    shrink=shrink
                )

    if rstate.random() < 0.95:
        fractal_branch(
            polylines, seg_pts[-1], ang + rstate.normal(0, 0.08),
            step * shrink, depth - 1, rstate,
            turn_sigma=turn_sigma,
            branch_prob=branch_prob * 0.90,
            shrink=shrink
        )

def render_cell_layers(W=1200, H=850, seed=11):
    """
    Returns a dict of layers (float canvases) that can be combined progressively.
    """
    rstate = rng(seed)

    cx, cy = W//2, H//2
    # Cell boundary (round-ish)
    cell_r = int(min(W, H) * 0.40)
    cell = circle_mask(W, H, cx, cy, cell_r)
    cell_inner = circle_mask(W, H, cx, cy, int(cell_r * 0.95))
    membrane = cell.astype(float) - cell_inner.astype(float)
    membrane[membrane < 0] = 0

    # Base cytoplasm shading
    base = np.zeros((H, W), float)
    base += cell_inner.astype(float) * 0.06
    base += membrane * 0.95

    # Nucleus
    nuc_r = int(cell_r * 0.22)
    nuc = circle_mask(W, H, int(cx - cell_r*0.08), int(cy - cell_r*0.06), nuc_r)
    nuc_inner = circle_mask(W, H, int(cx - cell_r*0.08), int(cy - cell_r*0.06), int(nuc_r*0.85))
    nucleus_ring = nuc.astype(float) - nuc_inner.astype(float)
    nucleus_ring[nucleus_ring < 0] = 0
    nucleus_fill = nuc_inner.astype(float) * 0.11

    # Nucleolus blobs
    nucleolus = np.zeros((H, W), float)
    for _ in range(3):
        gaussian_splat(
            nucleolus,
            (cx - cell_r*0.08) + rstate.normal(0, nuc_r*0.18),
            (cy - cell_r*0.06) + rstate.normal(0, nuc_r*0.18),
            amp=0.45,
            sigma=rstate.uniform(8, 14)
        )
    nucleolus *= nuc_inner.astype(float)

    # ER network (wavy polylines around nucleus)
    er = np.zeros((H, W), float)
    for _ in range(14):
        ang0 = rstate.uniform(0, 2*np.pi)
        rad0 = rstate.uniform(nuc_r*1.05, cell_r*0.85)
        x0 = (cx - cell_r*0.08) + np.cos(ang0)*rad0
        y0 = (cy - cell_r*0.06) + np.sin(ang0)*rad0
        pts = [(x0, y0)]
        ang = ang0 + rstate.normal(0, 0.35)
        step = rstate.uniform(7, 12)
        nseg = rstate.integers(35, 70)
        for _i in range(nseg):
            ang += rstate.normal(0, 0.18)
            x0 += np.cos(ang) * step
            y0 += np.sin(ang) * step
            pts.append((x0, y0))
        draw_polyline(er, pts, value=0.65, thickness=2)
    er *= cell_inner.astype(float)
    # Keep ER out of nucleus
    er *= (1.0 - nuc.astype(float))

    # Golgi (stacked arcs near nucleus)
    golgi = np.zeros((H, W), float)
    gx = int(cx + cell_r*0.22)
    gy = int(cy - cell_r*0.04)
    for k in range(7):
        # arc polyline
        rr = nuc_r * (0.75 + 0.09*k)
        t0, t1 = 0.2*np.pi, 1.15*np.pi
        ts = np.linspace(t0, t1, 120)
        pts = [(gx + np.cos(t)*rr*1.2, gy + np.sin(t)*rr*0.55) for t in ts]
        draw_polyline(golgi, pts, value=0.85, thickness=2)
    golgi *= cell_inner.astype(float)

    # Mitochondria: several capsules + fractal tubules inside each
    mito = np.zeros((H, W), float)
    mito_inner = np.zeros((H, W), float)

    mito_count = 10
    for _ in range(mito_count):
        mx = rstate.normal(cx + cell_r*0.10, cell_r*0.28)
        my = rstate.normal(cy + cell_r*0.10, cell_r*0.23)
        # ensure inside cell
        if (mx - cx)**2 + (my - cy)**2 > (cell_r*0.85)**2:
            continue

        length = rstate.uniform(140, 260)
        radius = rstate.uniform(38, 62)

        # outer capsule mask
        outer = capsule_mask(W, H, int(mx), int(my), length=length, radius=radius)
        inner = capsule_mask(W, H, int(mx), int(my), length=length*0.90, radius=radius*0.82)
        ring = outer.astype(float) - inner.astype(float)
        ring[ring < 0] = 0

        mito += ring * 1.0
        mito_inner += inner.astype(float) * 0.07

        # fractal tubules inside each mito
        polylines = []
        ang = rstate.uniform(-0.35, 0.35)
        for _s in range(2):
            sx = mx + rstate.normal(0, length*0.10)
            sy = my + rstate.normal(0, radius*0.12)
            fractal_branch(polylines, (sx, sy), ang, step=rstate.uniform(5.5, 7.8), depth=5, rstate=rstate)

        tmp = np.zeros((H, W), float)
        for poly in polylines:
            draw_polyline(tmp, poly, value=1.0, thickness=2)
        tmp *= inner.astype(float)
        mito_inner += tmp * 0.85

    mito *= cell_inner.astype(float)
    mito_inner *= cell_inner.astype(float)

    # Vesicles / lysosomes / ribosomes (dots)
    dots = np.zeros((H, W), float)
    for _ in range(550):
        # sample points within cell
        for __ in range(6):
            x = rstate.uniform(cx - cell_r, cx + cell_r)
            y = rstate.uniform(cy - cell_r, cy + cell_r)
            if (x-cx)**2 + (y-cy)**2 <= (cell_r*0.92)**2 and not nuc[int(y), int(x)]:
                break
        sigma = rstate.uniform(1.2, 2.6)
        amp = rstate.uniform(0.12, 0.30)
        gaussian_splat(dots, x, y, amp=amp, sigma=sigma)
    dots *= cell_inner.astype(float)

    return {
        "base": base,
        "nucleus_ring": nucleus_ring * 0.95,
        "nucleus_fill": nucleus_fill,
        "nucleolus": nucleolus,
        "er": er,
        "golgi": golgi,
        "mito_membrane": mito * 0.95,
        "mito_inner": mito_inner,
        "dots": dots,
        "cell_mask": cell_inner.astype(float),
    }

def composite(layers, keys):
    img = np.zeros_like(next(iter(layers.values())))
    for k in keys:
        img += layers[k]
    img = np.clip(img, 0, 1)
    # subtle gamma
    return img ** 0.88

def live_render_human_cell(seed=11, W=1200, H=850, pause=0.35):
    plt.ion()  # interactive mode

    layers = render_cell_layers(W=W, H=H, seed=seed)

    # Define progressive stages
    stages = [
        ("Initializing canvas…", ["base"]),
        ("Building nucleus…", ["base", "nucleus_fill", "nucleus_ring"]),
        ("Adding nucleolus…", ["base", "nucleus_fill", "nucleus_ring", "nucleolus"]),
        ("Growing endoplasmic reticulum…", ["base", "nucleus_fill", "nucleus_ring", "nucleolus", "er"]),
        ("Stacking Golgi apparatus…", ["base", "nucleus_fill", "nucleus_ring", "nucleolus", "er", "golgi"]),
        ("Generating mitochondria membranes…", ["base", "nucleus_fill", "nucleus_ring", "nucleolus", "er", "golgi", "mito_membrane"]),
        ("Filling mitochondria / cristae-like tubules…", ["base", "nucleus_fill", "nucleus_ring", "nucleolus", "er", "golgi", "mito_membrane", "mito_inner"]),
        ("Distributing vesicles & ribosomes…", ["base", "nucleus_fill", "nucleus_ring", "nucleolus", "er", "golgi", "mito_membrane", "mito_inner", "dots"]),
        ("Complete ✅", ["base", "nucleus_fill", "nucleus_ring", "nucleolus", "er", "golgi", "mito_membrane", "mito_inner", "dots"]),
    ]

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_axis_off()

    # initial
    img0 = composite(layers, stages[0][1])
    im = ax.imshow(img0, cmap="magma", interpolation="bilinear")

    # status overlay
    status_text = ax.text(
        0.02, 0.98, "",
        transform=ax.transAxes,
        va="top", ha="left",
        fontsize=14,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="black", alpha=0.35, edgecolor="none"),
        color="white"
    )

    # progress overlay
    progress_text = ax.text(
        0.98, 0.02, "",
        transform=ax.transAxes,
        va="bottom", ha="right",
        fontsize=12,
        bbox=dict(boxstyle="round,pad=0.30", facecolor="black", alpha=0.35, edgecolor="none"),
        color="white"
    )

    fig.canvas.draw()
    plt.show()

    total = len(stages)
    for i, (label, keys) in enumerate(stages, start=1):
        # Show "processing status" BETWEEN frames
        status_text.set_text(f"{label}")
        progress_text.set_text(f"Stage {i}/{total}")
        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(pause * 0.6)

        # Update the image
        img = composite(layers, keys)
        im.set_data(img)

        # brief "rendered" beat
        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(pause)

    return fig, ax

if __name__ == "__main__":
    # Run once
    live_render_human_cell(seed=14, pause=0.45)
    # Keep window open in some environments
    plt.ioff()
    plt.show()