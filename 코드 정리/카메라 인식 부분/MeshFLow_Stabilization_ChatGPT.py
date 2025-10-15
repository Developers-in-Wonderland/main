# MeshFLow_Stabilization_ChatGPT.py
# -------------------------------------------------------------------
# VS Code "Run ▶" 버튼으로 바로 실행 가능한 MeshFlow-style 안정화 전체 코드
# - 명령줄 인자 없이도 실행되도록 CONFIG 값으로 입력/출력/파라미터 설정
# - 필요시, 명령줄 인자 사용 모드도 지원(아래 RUN_WITH_ARGS=True 로 변경)
# -------------------------------------------------------------------

import sys
import os
import math
from collections import defaultdict

# ===== 안전한 OpenCV/Numpy import (설치 안내 메세지) =====
try:
    import cv2
except ModuleNotFoundError:
    print("[ERROR] OpenCV(cv2)가 설치되어 있지 않습니다. 아래 명령으로 설치하세요:")
    print("        pip install opencv-python")
    sys.exit(1)

try:
    import numpy as np
except ModuleNotFoundError:
    print("[ERROR] NumPy가 설치되어 있지 않습니다. 아래 명령으로 설치하세요:")
    print("        pip install numpy")
    sys.exit(1)

# ====== [CONFIG] VS Code 'Run ▶'로 실행 시 사용할 기본값 ======
# * 아래 경로/옵션만 수정하고 ▶ 누르면 바로 동작합니다.
CONFIG = {
    # 🔸 입력/출력 파일 경로
    "input_path":  r"C:\\Users\\82109\\Desktop\\stab\\output_12.avi",
    "output_path": r"C:\\Users\\82109\\Desktop\\stab\\output_12_stab.mp4",

    # 🔸 메쉬/Grid 해상도 (가로x세로)
    "grid":        (20, 12),

    # 🔸 특징점/크롭/재검출 주기
    "max_features": 900,
    "crop_ratio":   0.0,#0.08,     # 0.0~0.2 추천
    "refit_every":  8,        # N프레임마다 특징 재검출

    # 🔸 적응형 EMA 파라미터 (부드러움/반응성)
    "alpha_hi":   0.93,       # 모션 작을 때 사용할 큰 알파(부드러움↑)
    "alpha_lo":   0.70,       # 모션 클 때 사용할 작은 알파(반응↑)
    "mag_scale":  12.0,       # 모션 크기에 따른 알파 감소 민감도

    # 🔸 전역 앵커(호모그래피) 혼합(드리프트 방지)
    "anchor_w":    0.12,      # 0~0.3 권장
    "anchor_auto": True,      # 텍스처 빈곤/포인트 부족 시 자동 가중↑

    # 🔸 디버그(벡터 시각화 로그)
    "debug": True
}

# ▶ True 로 바꾸면 명령줄 인자 모드(예: 터미널에서 --input ... 사용)
RUN_WITH_ARGS = False

# ===================================================================
# 유틸
# ===================================================================

def l1_mag(field: np.ndarray) -> float:
    """2D 벡터장 평균 L1 크기"""
    return float(np.nanmean(np.abs(field[..., 0])) + np.nanmean(np.abs(field[..., 1])))

def robust_median(vecs):
    if not vecs:
        return None
    arr = np.asarray(vecs, dtype=np.float32)
    return np.median(arr, axis=0)

def iqr_mask_1d(x, k=1.5):
    q1, q3 = np.percentile(x, [25, 75])
    iqr = q3 - q1
    lo, hi = q1 - k * iqr, q3 + k * iqr
    return (x >= lo) & (x <= hi)

def reject_outliers_xy(vecs, k=1.5):
    if not vecs:
        return None
    arr = np.asarray(vecs, dtype=np.float32)
    dx, dy = arr[:, 0], arr[:, 1]
    mask = iqr_mask_1d(dx, k) & iqr_mask_1d(dy, k)
    arr = arr[mask]
    if len(arr) == 0:
        return None
    return np.median(arr, axis=0)

def parse_grid_str(s: str):
    s = s.lower().strip()
    if "x" in s:
        a, b = s.split("x")
        return int(a), int(b)
    v = int(s)
    return v, v

# ===================================================================
# Stabilizer (MeshFlow-Style, 강화 버전)
# ===================================================================

class MeshFlowPro:
    def __init__(
        self,
        frame_size,                 # (W, H)
        grid_wh=(20, 12),
        max_features=900,
        gftt_quality=0.01,
        gftt_min_dist=8,
        lk_win=21,
        lk_max_level=3,
        outlier_iqr_k=1.5,
        refit_every=8,
        crop_ratio=0.08,
        # 적응형 EMA
        alpha_hi=0.93,
        alpha_lo=0.70,
        mag_scale=12.0,
        # 전역 앵커
        anchor_w=0.12,
        anchor_auto=True,
        debug=False
    ):
        self.W, self.H = frame_size
        self.grid_w, self.grid_h = grid_wh
        self.cell_w = self.W / self.grid_w
        self.cell_h = self.H / self.grid_h
        self.vW = self.grid_w + 1
        self.vH = self.grid_h + 1

        self.max_features = max_features
        self.gftt_quality = gftt_quality
        self.gftt_min_dist = gftt_min_dist
        self.lk_params = dict(
            winSize=(lk_win, lk_win),
            maxLevel=lk_max_level,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        )
        self.outlier_iqr_k = outlier_iqr_k
        self.refit_every = refit_every
        self.crop_ratio = crop_ratio

        self.alpha_hi = alpha_hi
        self.alpha_lo = alpha_lo
        self.mag_scale = mag_scale

        self.anchor_w0  = anchor_w
        self.anchor_auto = anchor_auto

        self.debug = debug

        self.prev_gray = None
        self.prev_pts  = None
        self.frame_idx = 0
        self.prev_disp = np.zeros((self.vH, self.vW, 2), np.float32)

        self.good_mask = np.ones((self.H, self.W), np.uint8) * 255

    # --- Feature ---
    def detect_features(self, gray):
        pts = cv2.goodFeaturesToTrack(
            gray,
            maxCorners=self.max_features,
            qualityLevel=self.gftt_quality,
            minDistance=self.gftt_min_dist,
            mask=self.good_mask,
            useHarrisDetector=False
        )
        return pts if pts is not None else np.empty((0, 1, 2), np.float32)

    def track_features(self, prev_gray, gray, prev_pts):
        if prev_pts is None or len(prev_pts) == 0:
            return None, None
        nxt, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_pts, None, **self.lk_params)
        if nxt is None:
            return None, None
        good_new = nxt[st.squeeze() == 1]
        good_old = prev_pts[st.squeeze() == 1]
        return good_old, good_new

    # --- Mesh aggregation ---
    def assign_to_vertices(self, pts_old, pts_new):
        # --- 🔧 shape 보정 추가 ---
        pts_old = np.squeeze(pts_old)
        pts_new = np.squeeze(pts_new)

        buckets = defaultdict(list)  # (vy,vx) -> list of (dx,dy)
        for (x0, y0), (x1, y1) in zip(pts_old, pts_new):
            dx, dy = (x1 - x0), (y1 - y0)
            vx = int(round(x0 / self.cell_w))
            vy = int(round(y0 / self.cell_h))
            vx = np.clip(vx, 0, self.vW - 1)
            vy = np.clip(vy, 0, self.vH - 1)
            buckets[(vy, vx)].append((dx, dy))
        return buckets


    def initial_vertex_disp(self, buckets):
        disp = np.full((self.vH, self.vW, 2), np.nan, np.float32)
        for (vy, vx), vecs in buckets.items():
            med = reject_outliers_xy(vecs, k=self.outlier_iqr_k)
            if med is not None:
                disp[vy, vx] = med
        return disp

    def spatial_median_pass(self, disp_in, radius=1):
        H, W = disp_in.shape[:2]
        out = np.zeros_like(disp_in)
        for y in range(H):
            y0, y1 = max(0, y - radius), min(H, y + radius + 1)
            for x in range(W):
                x0, x1 = max(0, x - radius), min(W, x + radius + 1)
                patch = disp_in[y0:y1, x0:x1, :]
                ch0 = patch[..., 0][~np.isnan(patch[..., 0])]
                ch1 = patch[..., 1][~np.isnan(patch[..., 1])]
                vx = np.nanmedian(ch0) if ch0.size else disp_in[y, x, 0]
                vy = np.nanmedian(ch1) if ch1.size else disp_in[y, x, 1]
                if np.isnan(disp_in[y, x, 0]) or np.isnan(disp_in[y, x, 1]):
                    out[y, x] = (vx, vy)
                else:
                    out[y, x] = (0.5 * disp_in[y, x, 0] + 0.5 * vx,
                                 0.5 * disp_in[y, x, 1] + 0.5 * vy)
        return out

    def two_pass_spatial_median(self, disp):
        return self.spatial_median_pass(self.spatial_median_pass(disp, radius=1), radius=1)

    # --- Global anchor (weak homography) ---
    def estimate_global_disp_vertices(self, pts_old, pts_new):
        if pts_old is None or pts_new is None or len(pts_old) < 8:
            return np.full((self.vH, self.vW, 2), np.nan, np.float32)
        Hm, mask = cv2.findHomography(pts_old.reshape(-1, 1, 2), pts_new.reshape(-1, 1, 2),
                                      method=cv2.RANSAC, ransacReprojThreshold=3.0)
        if Hm is None:
            return np.full((self.vH, self.vW, 2), np.nan, np.float32)
        disp_g = np.zeros((self.vH, self.vW, 2), np.float32)
        for vy in range(self.vH):
            for vx in range(self.vW):
                x = vx * self.cell_w
                y = vy * self.cell_h
                src = np.array([[x, y, 1.0]], dtype=np.float32).T  # 3x1
                dst = Hm @ src
                dx = float(dst[0] / dst[2] - x)
                dy = float(dst[1] / dst[2] - y)
                disp_g[vy, vx] = (dx, dy)
        return disp_g

    # --- Temporal EMA (adaptive) ---
    def temporal_ema_adaptive(self, disp_vertices):
        mag = l1_mag(disp_vertices)
        alpha = self.alpha_hi - (mag / self.mag_scale)
        alpha = float(np.clip(alpha, self.alpha_lo, self.alpha_hi))
        m_hat = alpha * self.prev_disp + (1.0 - alpha) * disp_vertices
        self.prev_disp = m_hat.copy()
        return m_hat, alpha, mag

    # --- Warp + Crop ---
    def dense_remap_from_vertices(self, disp_vertices):
        dense = cv2.resize(disp_vertices, (self.W, self.H), interpolation=cv2.INTER_LINEAR)
        gx, gy = np.meshgrid(np.arange(self.W), np.arange(self.H))
        map_x = (gx + dense[..., 0]).astype(np.float32)
        map_y = (gy + dense[..., 1]).astype(np.float32)
        return map_x, map_y

    def safe_crop(self, frame):
        if self.crop_ratio <= 0:
            return frame
        h, w = frame.shape[:2]
        dx = int(w * self.crop_ratio)
        dy = int(h * self.crop_ratio)
        return frame[dy:h - dy, dx:w - dx]

    # --- One step ---
    def stabilize_step(self, frame_bgr):
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        need_refit = (self.prev_pts is None) or (len(self.prev_pts) < self.max_features * 0.35) \
                     or (self.frame_idx % self.refit_every == 0)

        if need_refit:
            self.prev_pts = self.detect_features(gray if self.prev_gray is None else self.prev_gray)

        if self.prev_gray is None:
            self.prev_gray = gray
            self.frame_idx += 1
            return frame_bgr, dict(alpha=None, mag=None, feats=len(self.prev_pts) if self.prev_pts is not None else 0)

        old, new = self.track_features(self.prev_gray, gray, self.prev_pts)

        self.prev_gray = gray
        self.prev_pts = new.reshape(-1, 1, 2) if new is not None and len(new) > 0 else None
        self.frame_idx += 1

        if old is None or new is None or len(old) < 12:
            return frame_bgr, dict(alpha=None, mag=None, feats=0)

        buckets = self.assign_to_vertices(old, new)
        disp_local = self.initial_vertex_disp(buckets)
        disp_local = self.two_pass_spatial_median(disp_local)

        disp_global = self.estimate_global_disp_vertices(old, new)

        anchor_w = self.anchor_w0
        if self.anchor_auto:
            nfeat = len(old)
            if nfeat < 300:
                factor = np.clip((300 - nfeat) / 300.0, 0.0, 1.0)
                anchor_w = self.anchor_w0 * (1.0 + 1.5 * factor)

        if not np.isnan(disp_global).all():
            disp_vertices = (1.0 - anchor_w) * disp_local + anchor_w * disp_global
        else:
            disp_vertices = disp_local

        disp_smooth, alpha, mag = self.temporal_ema_adaptive(disp_vertices)
        map_x, map_y = self.dense_remap_from_vertices(disp_smooth)
        warped = cv2.remap(frame_bgr, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_REFLECT)
        cropped = self.safe_crop(warped)

        if self.debug:
            # 필요하면 여기서 시각화 이미지를 저장하거나 로그를 출력할 수 있음
            print(f"[{self.frame_idx:05d}] feats={len(old):4d} alpha={alpha if alpha is not None else -1:.3f} mag={mag if mag is not None else -1:.2f}")

        return cropped, dict(alpha=alpha, mag=mag, feats=len(old))

# ===================================================================
# 실행 함수 (CONFIG 또는 명령줄 인자)
# ===================================================================

def run_with_config(cfg: dict):
    in_path  = cfg["input_path"]
    out_path = cfg["output_path"]
    grid_w, grid_h = cfg["grid"]

    if not os.path.exists(in_path):
        print(f"[ERROR] 입력 파일이 존재하지 않습니다:\n  {in_path}")
        return

    cap = cv2.VideoCapture(in_path)
    if not cap.isOpened():
        print(f"[ERROR] 영상을 열 수 없습니다: {in_path}")
        return

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    stab = MeshFlowPro(
        frame_size=(W, H),
        grid_wh=(grid_w, grid_h),
        max_features=cfg["max_features"],
        refit_every=cfg["refit_every"],
        crop_ratio=cfg["crop_ratio"],
        alpha_hi=cfg["alpha_hi"],
        alpha_lo=cfg["alpha_lo"],
        mag_scale=cfg["mag_scale"],
        anchor_w=cfg["anchor_w"],
        anchor_auto=cfg["anchor_auto"],
        debug=cfg["debug"]
    )

    out_w = int(W * (1 - 2 * cfg["crop_ratio"]))
    out_h = int(H * (1 - 2 * cfg["crop_ratio"]))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (out_w, out_h))

    print("[INFO] Stabilization start...")
    print(f"       input : {in_path}")
    print(f"       output: {out_path}")
    print(f"       grid  : {grid_w}x{grid_h}, features={cfg['max_features']}, crop={cfg['crop_ratio']}")
    print(f"       alpha : {cfg['alpha_lo']} ~ {cfg['alpha_hi']} (mag_scale={cfg['mag_scale']})")
    print(f"       anchor: w={cfg['anchor_w']} (auto={cfg['anchor_auto']})")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        stabilized, _ = stab.stabilize_step(frame)
        writer.write(stabilized)

    cap.release()
    writer.release()
    print(f"[OK] Saved -> {out_path}")

def run_with_args():
    import argparse
    ap = argparse.ArgumentParser(description="MeshFlow-style Video Stabilization (VS Code/CLI)")
    ap.add_argument("--input",  required=True, help="입력 영상 경로")
    ap.add_argument("--output", required=True, help="출력 영상 경로")
    ap.add_argument("--grid",   default="20x12", help="격자: 예) 20x12")
    ap.add_argument("--feat",   type=int, default=900, help="최대 특징점 수")
    ap.add_argument("--crop",   type=float, default=0.08, help="안전 크롭 비율 (0~0.2)")
    ap.add_argument("--refit",  type=int, default=8, help="N프레임마다 특징 재검출")
    ap.add_argument("--alpha_hi", type=float, default=0.93)
    ap.add_argument("--alpha_lo", type=float, default=0.70)
    ap.add_argument("--mag_scale", type=float, default=12.0)
    ap.add_argument("--anchor_w",  type=float, default=0.12)
    ap.add_argument("--no_anchor_auto", action="store_true")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    in_path  = args.input
    out_path = args.output
    gw, gh   = parse_grid_str(args.grid)

    if not os.path.exists(in_path):
        print(f"[ERROR] 입력 파일이 존재하지 않습니다:\n  {in_path}")
        return

    cap = cv2.VideoCapture(in_path)
    if not cap.isOpened():
        print(f"[ERROR] 영상을 열 수 없습니다: {in_path}")
        return

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    stab = MeshFlowPro(
        frame_size=(W, H),
        grid_wh=(gw, gh),
        max_features=args.feat,
        refit_every=args.refit,
        crop_ratio=args.crop,
        alpha_hi=args.alpha_hi,
        alpha_lo=args.alpha_lo,
        mag_scale=args.mag_scale,
        anchor_w=args.anchor_w,
        anchor_auto=(not args.no_anchor_auto),
        debug=args.debug
    )

    out_w = int(W * (1 - 2 * args.crop))
    out_h = int(H * (1 - 2 * args.crop))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (out_w, out_h))

    print("[INFO] Stabilization start (CLI mode)...")
    print(f"       input : {in_path}")
    print(f"       output: {out_path}")
    print(f"       grid  : {gw}x{gh}, features={args.feat}, crop={args.crop}")
    print(f"       alpha : {args.alpha_lo} ~ {args.alpha_hi} (mag_scale={args.mag_scale})")
    print(f"       anchor: w={args.anchor_w} (auto={not args.no_anchor_auto})")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        stabilized, _ = stab.stabilize_step(frame)
        writer.write(stabilized)

    cap.release()
    writer.release()
    print(f"[OK] Saved -> {out_path}")

# ===================================================================
# 엔트리포인트: VS Code 'Run ▶'에서 바로 실행되도록 구성
# ===================================================================

if __name__ == "__main__":
    # VS Code 'Run ▶' 기본: CONFIG 사용
    if not RUN_WITH_ARGS:
        run_with_config(CONFIG)
    else:
        # 터미널/디버그 설정에서 인자 넘길 때 사용
        run_with_args()
