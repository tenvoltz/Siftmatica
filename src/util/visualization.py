import cv2
import numpy as np
import matplotlib.pyplot as plt
from src.util.geometry import shift_point, draw_infinite_line, line_to_endpoints
from src.util.logger import get_logger

logger = get_logger(__name__)


class Visualizer:
    @staticmethod
    def _ensure_bgr(img):
        """Convert image to BGR format if needed."""
        if img.ndim == 2:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 4:
            return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        return img

    @staticmethod
    def _draw_line(canvas, line, color, vp_shift=None):
        p1, p2 = line
        h, w = canvas.shape[:2]

        if vp_shift is not None:
            (x1, y1), (x2, y2) = line_to_endpoints(line, h, w)
            mx, my = (x1 + x2) // 2, (y1 + y2) // 2
            draw_infinite_line(canvas, (mx, my), vp_shift, color)
        else:
            cv2.line(canvas, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), color, 2)

    @staticmethod
    def plot_polar_histogram(lines, output_path, title, num_bins=36):
        dirs = np.array([line[1] - line[0] for line in lines], dtype=float)
        norms = np.linalg.norm(dirs, axis=1)
        dirs = dirs[norms > 0] / norms[norms > 0][:, None]
        angles = np.arctan2(dirs[:, 1], dirs[:, 0]) % np.pi

        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection="polar")
        ax.hist(angles, bins=num_bins, weights=norms, color="#3182bd", alpha=0.75, edgecolor="black", linewidth=0.5)
        ax.set_title(title, pad=15)
        ax.set_theta_zero_location("E")
        ax.set_theta_direction(-1)
        ax.set_thetagrids(range(0, 180, 15))
        ax.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close(fig)
        logger.trace("visualization", "Plotted polar histogram", {"output": output_path})

    @staticmethod
    def plot_vanishing_points(img, lines, vps, inlier_line_indices_list, output_path, title, plot_line_extensions=True):
        min_x, max_x, min_y, max_y = Visualizer._compute_canvas_bounds(img, vps)
        canvas = np.zeros((max_y - min_y, max_x - min_x, 3), dtype=np.uint8)
        offset = (min_x, min_y)
        img_shift = (-min_x, -min_y)

        img = Visualizer._ensure_bgr(img)
        canvas[img_shift[1]:img_shift[1] + img.shape[0], img_shift[0]:img_shift[0] + img.shape[1]] = img

        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
        for i, (vp, line_indices) in enumerate(zip(vps, inlier_line_indices_list)):
            color = colors[i % len(colors)]
            vp_shift = shift_point(vp, offset)
            cv2.circle(canvas, vp_shift, 6, color, -1)

            for line_idx in line_indices:
                if line_idx < len(lines):
                    # Shift line endpoints by offset
                    p1, p2 = lines[line_idx]
                    p1_shift = shift_point(p1, offset)
                    p2_shift = shift_point(p2, offset)
                    shifted_line = (p1_shift, p2_shift)

                    vp_for_line = vp_shift if plot_line_extensions else None
                    Visualizer._draw_line(canvas, shifted_line, color, vp_for_line)

        plt.figure(figsize=(10, 10))
        plt.imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
        plt.title(title)
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        logger.trace("visualization", "Plotted vanishing points", {"output": output_path})

    @staticmethod
    def plot_hough_lines(img, lines, output_path, title):
        canvas = Visualizer._ensure_bgr(img).copy()

        for line in lines:
            Visualizer._draw_line(canvas, line, (0, 255, 0))

        plt.figure(figsize=(10, 8))
        plt.imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
        plt.title(title)
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        logger.trace("visualization", "Plotted Hough lines", {"output": output_path})

    @staticmethod
    def plot_contours(img, contours, output_path, title):
        canvas = img.copy()
        cv2.drawContours(canvas, contours, -1, (0, 255, 0), 2)

        plt.figure(figsize=(10, 8))
        plt.imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
        plt.title(title)
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        logger.trace("visualization", "Plotted contours", {"output": output_path})

    @staticmethod
    def plot_filtered_img(img, output_path, title):
        plt.figure(figsize=(10, 8))
        if img.ndim == 2:
            plt.imshow(img, cmap='gray')
        else:
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title(title)
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        logger.trace("visualization", "Plotted filtered image", {"output": output_path})

    @staticmethod
    def _compute_canvas_bounds(img, vps, pad=200):
        h, w = img.shape[:2]
        xs, ys = [0, w], [0, h]
        for vp in vps:
            xs.append(int(vp[0]))
            ys.append(int(vp[1]))
        return int(min(xs)) - pad, int(max(xs)) + pad, int(min(ys)) - pad, int(max(ys)) + pad
