"""
Compute optical flow between two images and optionally save results.

Provides a simple, dependency-light interface using OpenCV (cv2).

Functions:
    compute_optical_flow(img1, img2, method='farneback', save_path=None, save_viz=True, save_raw=True, resize=False, params=None)

The function accepts file paths or numpy arrays for img1/img2. It returns a tuple
    (flow, flow_vis)
where `flow` is a float32 array of shape (H,W,2) containing the horizontal and vertical
flow components, and `flow_vis` is a uint8 BGR image visualising flow (suitable for cv2.imwrite).

Save behaviour (save_path):
 - If save_path is a directory, two files are written there: flow.npy and flow_vis.png
 - If save_path has an extension matching .npy/.npz the raw flow is written to that path;
   the visualization is written alongside with .png extension.
 - If save_path has an image extension (.png/.jpg/...) the visualization is written there;
   the raw flow is written alongside as .npy if save_raw is True.

Default optical flow method is Farneback. TV-L1 is attempted if method='tvl1' and the
installed OpenCV build exposes the implementation.

Requires: opencv-python

Example::
    from utils.optical_flow import compute_optical_flow
    flow, vis = compute_optical_flow('frame0.png', 'frame1.png', save_path='out_dir')

"""
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np

import cv2


def _read_gray(img: Union[str, Path, np.ndarray]):
    """Read an image path or accept a numpy array and return a grayscale float32 image."""
    if isinstance(img, (str, Path)):
        arr = cv2.imread(str(img), cv2.IMREAD_COLOR)
        if arr is None:
            raise FileNotFoundError(f"Could not read image: {img}")
        gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
        return gray.astype(np.uint8)
    if isinstance(img, np.ndarray):
        if img.ndim == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            return gray.astype(np.uint8)
        elif img.ndim == 2:
            return img.astype(np.uint8)
        else:
            raise ValueError("Unsupported numpy image shape: {}".format(img.shape))
    raise TypeError("img must be a file path or numpy.ndarray")


def _flow_to_bgr(flow: np.ndarray, max_magnitude: Optional[float] = None) -> np.ndarray:
    """Convert optical flow to a BGR image for visualization.

    Hue encodes direction, value encodes magnitude.
    """
    h, w = flow.shape[:2]
    fx, fy = flow[..., 0], flow[..., 1]
    mag, ang = cv2.cartToPolar(fx, fy, angleInDegrees=True)

    # Normalize magnitude to [0,255]
    if max_magnitude is None:
        # robust estimate: use 95th percentile to avoid outlier saturation
        max_magnitude = np.percentile(mag, 95)
        if max_magnitude <= 0:
            max_magnitude = 1.0

    v = np.clip((mag / max_magnitude) * 255.0, 0, 255).astype(np.uint8)

    # OpenCV Hue range: [0,180], use angle/2 because ang in degrees [0,360)
    h_img = (ang / 2).astype(np.uint8)
    s_img = np.full((h, w), 255, dtype=np.uint8)

    hsv = np.stack([h_img, s_img, v], axis=-1)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr


def compute_optical_flow(
    img1: Union[str, Path, np.ndarray],
    img2: Union[str, Path, np.ndarray],
    method: str = "farneback",
    save_path: Optional[Union[str, Path]] = None,
    save_viz: bool = True,
    save_raw: bool = True,
    resize: bool = False,
    params: Optional[dict] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute optical flow between img1 and img2.

    Args:
        img1, img2: paths or numpy arrays of the two images (previous, next)
        method: 'farneback' (default) or 'tvl1'
        save_path: optional path or directory to save outputs
        save_viz: whether to save a visualization image (PNG)
        save_raw: whether to save the raw flow array (npy)
        resize: if True, resize img2 to img1's shape when shapes differ
        params: dict of method-specific parameters

    Returns:
        (flow, flow_vis)
          flow: np.ndarray float32 shape (H,W,2)
          flow_vis: np.ndarray uint8 shape (H,W,3) BGR visualization
    """
    p = Path(save_path) if save_path is not None else None

    im1 = _read_gray(img1)
    im2 = _read_gray(img2)

    if im1.shape != im2.shape:
        if resize:
            im2 = cv2.resize(im2, (im1.shape[1], im1.shape[0]), interpolation=cv2.INTER_LINEAR)
        else:
            raise ValueError(f"Image shapes differ: {im1.shape} vs {im2.shape}. Set resize=True to resize the second image.")

    params = params or {}

    if method.lower() == "farneback":
        # sensible defaults, can be overridden via params
        fb_defaults = dict(pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
        fb_defaults.update(params)
        flow = cv2.calcOpticalFlowFarneback(im1, im2, None, **fb_defaults)
    elif method.lower() in ("tvl1", "dual_tvl1"):
        # attempt to create TV-L1 optical flow (may not be available in all cv2 builds)
        try:
            # prefer cv2.optflow interface
            tvl1 = cv2.optflow.DualTVL1OpticalFlow_create()
        except Exception:
            try:
                tvl1 = cv2.DualTVL1OpticalFlow_create()
            except Exception:
                raise RuntimeError("TV-L1 optical flow is not available in this OpenCV build")
        # TV-L1 accepts single-channel float32 images
        f1 = im1.astype(np.float32) / 255.0
        f2 = im2.astype(np.float32) / 255.0
        flow = tvl1.calc(f1, f2, None)
    else:
        raise ValueError(f"Unknown optical flow method: {method}")

    # flow is (H,W,2) float32
    if not isinstance(flow, np.ndarray):
        flow = np.array(flow, dtype=np.float32)
    else:
        flow = flow.astype(np.float32)

    # visualization
    flow_vis = _flow_to_bgr(flow)

    # save if requested
    if p is not None:
        if p.exists() and p.is_file():
            ext = p.suffix.lower()
        else:
            ext = p.suffix.lower()

        # if path looks like a directory or has no meaningful extension, create directory
        if (p.exists() and p.is_dir()) or (ext == ""):
            out_dir = p if p.exists() else p
            out_dir.mkdir(parents=True, exist_ok=True)
            raw_path = out_dir / "flow.npy"
            vis_path = out_dir / "flow_vis.png"
        else:
            # path is a file-like path
            if ext in (".npy", ".npz"):
                raw_path = p
                vis_path = p.with_suffix('.png')
            elif ext in ('.png', '.jpg', '.jpeg', '.bmp'):
                vis_path = p
                raw_path = p.with_suffix('.npy')
            else:
                # unknown extension, use as directory
                out_dir = p
                out_dir.mkdir(parents=True, exist_ok=True)
                raw_path = out_dir / "flow.npy"
                vis_path = out_dir / "flow_vis.png"

        if save_raw:
            try:
                np.save(raw_path, flow)
            except Exception as e:
                print(f"Warning: failed to save raw flow to {raw_path}: {e}")
        if save_viz:
            try:
                cv2.imwrite(str(vis_path), flow_vis)
            except Exception as e:
                print(f"Warning: failed to save visualization to {vis_path}: {e}")

    return flow, flow_vis


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compute optical flow between two images")
    parser.add_argument("img1", help="Previous/first image path")
    parser.add_argument("img2", help="Next/second image path")
    parser.add_argument("--method", default="farneback", choices=["farneback", "tvl1"], help="Optical flow method")
    parser.add_argument("--out", default=None, help="Directory or file path to save outputs")
    parser.add_argument("--no-viz", dest="save_viz", action="store_false", help="Do not save visualization image")
    parser.add_argument("--no-raw", dest="save_raw", action="store_false", help="Do not save raw flow array")
    parser.add_argument("--resize", action="store_true", help="Resize second image to first if shapes differ")
    args = parser.parse_args()

    flow, vis = compute_optical_flow(args.img1, args.img2, method=args.method, save_path=args.out, save_viz=args.save_viz, save_raw=args.save_raw, resize=args.resize)
    print("Computed optical flow. Flow shape:", flow.shape)
    if args.out is None:
        # show visualization in a simple OpenCV window (blocking)
        try:
            cv2.imshow('flow', vis)
            print('Press any key in the image window to exit')
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except Exception:
            # falling back to not showing if running headless
            pass
