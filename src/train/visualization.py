import logging

import numpy as np

MATPLOTLIB_FLAG = False


def _ensure_matplotlib_agg():
    global MATPLOTLIB_FLAG
    if MATPLOTLIB_FLAG:
        return
    import matplotlib

    matplotlib.use("Agg")
    MATPLOTLIB_FLAG = True
    logging.getLogger("matplotlib").setLevel(logging.WARNING)


def _matplotlib_pyplot():
    _ensure_matplotlib_agg()
    import matplotlib.pylab as plt

    return plt


def _figure_canvas_to_rgb_array(fig):
    fig.canvas.draw()
    if hasattr(fig.canvas, "tostring_rgb"):
        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        return data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()


def _render_figure_to_rgb_array(fig, plt):
    data = _figure_canvas_to_rgb_array(fig)
    plt.close(fig)
    return data


def _plot_matrix_to_rgb(
    matrix,
    *,
    figsize,
    xlabel,
    ylabel,
    title=None,
    info=None,
    transpose=False,
    cmap=None,
    vmin=None,
    vmax=None,
):
    plt = _matplotlib_pyplot()
    fig, ax = plt.subplots(figsize=figsize)
    image = matrix.transpose() if transpose else matrix
    im = ax.imshow(
        image,
        aspect="auto",
        origin="lower",
        interpolation="none",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    fig.colorbar(im, ax=ax)
    if title is not None:
        ax.set_title(title)
    if info is not None:
        xlabel = f"{xlabel}\n\n{info}"
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    return _render_figure_to_rgb_array(fig, plt)


def plot_spectrogram_to_numpy(spectrogram):
    return _plot_matrix_to_rgb(
        spectrogram,
        figsize=(10, 2),
        xlabel="Frames",
        ylabel="Channels",
    )


def _render_validation_panel(fig, ax, title, mel, cmap, vmin, vmax):
    im = ax.imshow(
        mel,
        aspect="auto",
        origin="lower",
        interpolation="none",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    fig.colorbar(im, ax=ax)
    ax.set_title(title)
    ax.set_ylabel("Channels")


def plot_validation_mels_to_numpy(gt_mel, pred_mel, diff_mel):
    plt = _matplotlib_pyplot()

    gt_mel = np.asarray(gt_mel)
    pred_mel = np.asarray(pred_mel)
    diff_mel = np.asarray(diff_mel)

    mel_min = float(min(np.min(gt_mel), np.min(pred_mel)))
    mel_max = float(max(np.max(gt_mel), np.max(pred_mel)))
    if mel_min == mel_max:
        mel_max = mel_min + 1e-6

    diff_abs_max = float(np.max(np.abs(diff_mel)))
    if diff_abs_max == 0.0:
        diff_abs_max = 1e-6

    fig, axes = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
    panels = (
        ("GT", gt_mel, "viridis", mel_min, mel_max),
        ("PRED", pred_mel, "viridis", mel_min, mel_max),
        ("DIFF", diff_mel, "seismic", -diff_abs_max, diff_abs_max),
    )

    for ax, (title, mel, cmap, vmin, vmax) in zip(axes, panels):
        _render_validation_panel(fig, ax, title, mel, cmap, vmin, vmax)

    axes[-1].set_xlabel("Frames")
    plt.tight_layout()
    return _render_figure_to_rgb_array(fig, plt)


def plot_alignment_to_numpy(alignment, info=None):
    return _plot_matrix_to_rgb(
        alignment,
        figsize=(6, 4),
        xlabel="Decoder timestep",
        ylabel="Encoder timestep",
        info=info,
        transpose=True,
    )
