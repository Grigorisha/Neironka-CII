#!/usr/bin/env python3
"""
Офлайн-калибровка камеры по фото шахматной доски (OpenCV).

"""
import argparse
import glob
import os

import cv2
import numpy as np


def main():
    parser = argparse.ArgumentParser(
        description="Camera calibration using chessboard images (no ArUco)."
    )
    parser.add_argument(
        "--images",
        required=True,
        help='Glob pattern, e.g. "/path/to/calib_photos/*.jpg"',
    )
    parser.add_argument(
        "--pattern_size",
        type=str,
        default="9x6",
        help="Chessboard inner corners as WxH, e.g. 9x6 (INNER corners, not squares).",
    )
    parser.add_argument(
        "--square_size",
        type=float,
        default=0.04,
        help="Square size in meters (or any unit, but be consistent).",
    )
    parser.add_argument(
        "--out",
        default="camera_calib.yml",
        help="Output YAML file (OpenCV FileStorage format).",
    )
    parser.add_argument(
        "--min_images",
        type=int,
        default=10,
        help="Minimum number of valid frames for calibration.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show detections while processing (press key to advance).",
    )
    parser.add_argument(
        "--show_delay_ms",
        type=int,
        default=200,
        help="Delay between preview frames in --show mode (ms). Use 0 for fastest non-blocking refresh.",
    )
    args = parser.parse_args()

    try:
        w, h = args.pattern_size.lower().split("x")
        pattern_size = (int(w), int(h))
    except Exception:
        raise SystemExit('Invalid --pattern_size. Use format like "9x6".')

    images = sorted(glob.glob(args.images))
    if not images:
        print(f"Нет изображений по маске: {args.images}")
        return 1

    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0 : pattern_size[0], 0 : pattern_size[1]].T.reshape(-1, 2)
    objp *= args.square_size

    objpoints = []
    imgpoints = []
    imsize = None

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-4)

    good = 0
    for fname in images:
        img = cv2.imread(fname)
        if img is None:
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if imsize is None:
            imsize = (gray.shape[1], gray.shape[0])

        flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
        found, corners = cv2.findChessboardCorners(gray, pattern_size, flags)

        if not found:
            if args.show:
                vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                cv2.putText(
                    vis, "NOT FOUND", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2
                )
                cv2.imshow("chessboard", vis)
                key = cv2.waitKey(max(0, int(args.show_delay_ms))) & 0xFF
                if key in (ord("q"), 27):
                    print("Остановлено пользователем в режиме --show.")
                    break
            continue

        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        objpoints.append(objp.copy())
        imgpoints.append(corners2)
        good += 1

        if args.show:
            vis = cv2.drawChessboardCorners(img.copy(), pattern_size, corners2, found)
            cv2.putText(
                vis,
                os.path.basename(fname),
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2,
            )
            cv2.imshow("chessboard", vis)
            key = cv2.waitKey(max(0, int(args.show_delay_ms))) & 0xFF
            if key in (ord("q"), 27):
                print("Остановлено пользователем в режиме --show.")
                break

    if args.show:
        cv2.destroyAllWindows()

    print(f"Всего изображений: {len(images)}")
    print(f"Подходящих (углы найдены): {good}")

    if good < args.min_images:
        print(f"Недостаточно кадров для калибровки (нужно минимум {args.min_images})")
        return 1

    ret, K, D, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, imsize, None, None
    )

    print("RMS:", ret)
    print("K (camera matrix):\n", K)
    print("D (dist coeffs):\n", D.ravel())

    mean_error = 0.0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], K, D)
        err = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        mean_error += err
    mean_error /= len(objpoints)
    print("Mean reprojection error:", mean_error)

    fs = cv2.FileStorage(args.out, cv2.FILE_STORAGE_WRITE)
    fs.write("camera_matrix", K)
    fs.write("dist_coeffs", D)
    fs.write("image_width", imsize[0])
    fs.write("image_height", imsize[1])
    fs.write("rms", float(ret))
    fs.write("mean_reprojection_error", float(mean_error))
    fs.write("pattern_size_w", pattern_size[0])
    fs.write("pattern_size_h", pattern_size[1])
    fs.write("square_size", float(args.square_size))
    fs.release()

    print("Saved:", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
