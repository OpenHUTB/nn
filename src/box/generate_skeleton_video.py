import os
import shutil
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import imageio


def make_sphere(center, r=0.12, n=24):
    u = np.linspace(0, 2 * np.pi, n)
    v = np.linspace(0, np.pi, n)
    x = center[0] + r * np.outer(np.cos(u), np.sin(v))
    y = center[1] + r * np.outer(np.sin(u), np.sin(v))
    z = center[2] + r * np.outer(np.ones_like(u), np.cos(v))
    return x, y, z


def set_axes_equal(ax):
    # Workaround to set equal aspect on 3D axes
    limits = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
    centers = np.mean(limits, axis=1)
    max_range = 0.5 * np.max(limits[:, 1] - limits[:, 0])
    ax.set_xlim3d(centers[0] - max_range, centers[0] + max_range)
    ax.set_ylim3d(centers[1] - max_range, centers[1] + max_range)
    ax.set_zlim3d(centers[2] - max_range, centers[2] + max_range)


def render_frame(theta_shoulder, theta_elbow, ribs_twist, eye_bob, figsize=(6, 6), dpi=100):
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor((0.85, 0.93, 0.99))
    fig.patch.set_facecolor((0.85, 0.93, 0.99))

    # Draw ribs (semi-circular arcs)
    rib_count = 9
    for i in range(rib_count):
        t = np.linspace(-np.pi / 2, np.pi / 2, 120)
        radius_x = 0.6 + i * 0.02
        radius_y = 0.18 + i * 0.015
        z = 0.0 + (i - rib_count // 2) * 0.03
        x = radius_x * np.cos(t) * math.cos(ribs_twist)
        y = radius_y * np.sin(t) * math.cos(ribs_twist)
        ax.plot(x - 0.2, y, z, color='sandybrown', linewidth=3)

    # Shoulder position (near ribcage)
    shoulder = np.array([0.0, 0.0, 0.0])
    # Upper arm
    upper_len = 0.7
    elbow = shoulder + np.array([
        -upper_len * math.cos(theta_shoulder),
        -upper_len * math.sin(theta_shoulder) * 0.6,
        -upper_len * math.sin(theta_shoulder) * 0.4,
    ])
    # Forearm
    fore_len = 0.6
    wrist = elbow + np.array([
        -fore_len * math.cos(theta_shoulder + theta_elbow),
        -fore_len * math.sin(theta_shoulder + theta_elbow) * 0.6,
        -fore_len * math.sin(theta_shoulder + theta_elbow) * 0.4,
    ])

    # Draw bones
    ax.plot([shoulder[0], elbow[0]], [shoulder[1], elbow[1]], [shoulder[2], elbow[2]],
            color='saddlebrown', linewidth=5)
    ax.plot([elbow[0], wrist[0]], [elbow[1], wrist[1]], [elbow[2], wrist[2]],
            color='saddlebrown', linewidth=4)

    # Draw simple hand bones
    hand_tip = wrist + (wrist - elbow) * 0.3
    ax.plot([wrist[0], hand_tip[0]], [wrist[1], hand_tip[1]], [wrist[2], hand_tip[2]],
            color='saddlebrown', linewidth=3)

    # Muscles (red lines near bones)
    def offset_point(p, q, off=0.06):
        v = q - p
        n = np.array([-v[1], v[0], 0.0])
        n = n / (np.linalg.norm(n) + 1e-8)
        return p + n * off, q + n * off

    a1, a2 = offset_point(shoulder, elbow, off=0.07)
    ax.plot([a1[0], a2[0]], [a1[1], a2[1]], [a1[2], a2[2]], color='crimson', linewidth=3)
    b1, b2 = offset_point(elbow, wrist, off=0.05)
    ax.plot([b1[0], b2[0]], [b1[1], b2[1]], [b1[2], b2[2]], color='crimson', linewidth=3)

    # Small clavicle
    clav_r = 0.18
    t = np.linspace(-0.6, 0.2, 20)
    clav_x = clav_r * np.cos(t) - 0.1
    clav_y = clav_r * np.sin(t) * 0.5 + 0.05
    clav_z = 0.08 + 0.02 * np.sin(t * 3 + ribs_twist)
    ax.plot(clav_x, clav_y, clav_z, color='saddlebrown', linewidth=3)

    # Eye (a sphere floating above)
    eye_center = np.array([0.0, 0.9 + 0.06 * math.sin(eye_bob), 0.6])
    xs, ys, zs = make_sphere(eye_center, r=0.12, n=32)
    ax.plot_surface(xs, ys, zs, color='white', linewidth=0, shade=True)
    # iris
    xs2, ys2, zs2 = make_sphere(eye_center + np.array([0.0, 0.02, 0.02]), r=0.04, n=18)
    ax.plot_surface(xs2, ys2, zs2, color='dodgerblue', linewidth=0, shade=True)
    xs3, ys3, zs3 = make_sphere(eye_center + np.array([0.0, 0.02, 0.03]), r=0.015, n=10)
    ax.plot_surface(xs3, ys3, zs3, color='black', linewidth=0, shade=True)

    # Camera / view
    ax.view_init(elev=10, azim= -30 + 30 * math.sin(ribs_twist * 0.5))
    ax.set_xlim(-1.4, 1.0)
    ax.set_ylim(-1.4, 1.4)
    ax.set_zlim(-0.8, 1.2)
    set_axes_equal(ax)
    ax.axis('off')

    # Ensure no margins so saved PNG matches requested pixel size
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    # Render to numpy array via PNG buffer (robust across backends)
    from io import BytesIO
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=fig.dpi, bbox_inches=None, pad_inches=0)
    buf.seek(0)
    img = imageio.imread(buf)
    plt.close(fig)
    return img


def generate_video(outfile='skeleton_arm.mp4', frames=240, fps=30, width=800, height=800, dpi=100):
    tmpdir = 'frames_tmp_generate_skeleton'
    if os.path.exists(tmpdir):
        shutil.rmtree(tmpdir)
    os.makedirs(tmpdir, exist_ok=True)

    filenames = []
    print('Rendering frames...')
    # compute figure size in inches from desired pixel width/height and dpi
    fig_inches = (width / dpi, height / dpi)
    for i in range(frames):
        t = i / frames * 2 * math.pi
        theta_shoulder = 0.6 * math.sin(t * 1.0)
        theta_elbow = 0.4 * math.sin(t * 1.3 + 0.5)
        ribs_twist = 0.25 * math.sin(t * 0.7)
        eye_bob = 0.2 * math.sin(t * 1.6)
        # pass figsize and dpi so final PNG matches requested pixel dimensions
        img = render_frame(theta_shoulder, theta_elbow, ribs_twist, eye_bob, figsize=fig_inches, dpi=dpi)
        fname = os.path.join(tmpdir, f'frame_{i:04d}.png')
        imageio.imwrite(fname, img)
        filenames.append(fname)
        if (i + 1) % 40 == 0:
            print(f'  rendered {i+1}/{frames} frames')

    print('Composing video...')
    # Use macro_block_size=1 if strict codec compatibility issues arise; here we keep defaults.
    with imageio.get_writer(outfile, fps=fps, codec='libx264', quality=8) as writer:
        for fname in filenames:
            frame = imageio.imread(fname)
            writer.append_data(frame)

    # cleanup frames
    shutil.rmtree(tmpdir)
    print('Saved video to', outfile)


if __name__ == '__main__':
    import argparse

    p = argparse.ArgumentParser(description='Generate a 3D skeleton arm animation video')
    p.add_argument('--out', default='skeleton_arm.mp4')
    p.add_argument('--frames', type=int, default=240)
    p.add_argument('--fps', type=int, default=30)
    p.add_argument('--width', type=int, default=800, help='output width in pixels')
    p.add_argument('--height', type=int, default=800, help='output height in pixels')
    p.add_argument('--dpi', type=int, default=100, help='DPI used to set figure size (width/dpi inches)')
    args = p.parse_args()
    generate_video(outfile=args.out, frames=args.frames, fps=args.fps, width=args.width, height=args.height, dpi=args.dpi)
