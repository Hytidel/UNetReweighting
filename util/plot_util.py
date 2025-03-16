from util.logger import logger

from typing import Optional, Tuple, Union, Dict, List, Set, Callable, TypeVar, Generic, NewType, Protocol

from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
from PIL import Image

from util.numpy_util import tsfm_to_1d_array


def save_plot(
    fig, 
    save_plot_root_path: Union[str, Path], 
    save_plot_filename: str
):
    if not isinstance(save_plot_root_path, Path):
        save_plot_root_path = Path(save_plot_root_path)
    
    save_plot_root_path.mkdir(parents = True, exist_ok = True)

    save_plot_path = save_plot_root_path / save_plot_filename
    fig.savefig(save_plot_path)

def get_line_chart(
    figsize: Optional[Union[float, Tuple[float, float]]] = (10, 5), 

    x_list: Optional[Union[List[float], np.ndarray]] = None, 
    y_list_list: Union[List[List[float]], List[np.ndarray], np.ndarray] = None, 
    y_label_list: Union[str, List[str]] = None, 

    marker_list: Union[str, List[str]] = [
        'o',  # circle
        's',  # square
        '*',  # star
        '+',  # plus
        'x',  # x

        'd',  # diamond
        'D',  # thin diamond

        '^',  # up-pointing triangle
        'v',  # down-pointing triangle
        '<',  # left-pointing triangle
        '>',  # right-pointing triangle

        'p',  # pentagon
        'h',  # hexagon 1
        'H',  # hexagon 2
    ], 
    color_list: Union[str, List[str]] = None,  # color name or hex
    alpha_list: Union[float, List[float]] = None,

    num_sample: int = None, 
    std_list_list: Union[float, List[List[float]], np.ndarray] = None, 
    confidence_level: float = None, 

    fill_alpha_list: Union[float, List[float]] = None, 
    face_color_list: Union[str, List[str]] = None,  # color name or hex

    plot_title: Optional[str] = None, 
    plot_x_label: Optional[str] = None, 
    plot_y_label: Optional[str] = None, 

    show_grid: Optional[bool] = True, 
    show_legend: Optional[bool] = True
) -> Tuple:
    # get `figsize`
    if isinstance(figsize, tuple):
        pass
    elif isinstance(figsize, float):
        figsize = (figsize, figsize)
    else:
        raise ValueError(
            f"Unsupported type of `figsize`, got `{type(figsize)}`. "
        )

    # get `num_plot`, `max_len_y_list`
    if isinstance(y_list_list, np.ndarray):
        num_plot = y_list_list.shape[0]
        max_len_y_list = y_list_list.shape[1]
    elif isinstance(y_list_list, list):
        num_plot = len(y_list_list)
        
        # y_list_list: List[List[float]]
        if isinstance(y_list_list[0], list):
            max_len_y_list = max(
                [len(y_list) for y_list in y_list_list]
            )
        # y_list_list: List[np.ndarray]
        else:
            max_len_y_list = max(
                [y_list.shape[0] for y_list in y_list_list]
            )
    else:
        raise ValueError(
            f"Unsupported type of `y_list_list`, got `{type(y_list_list)}`. "
        )

    # get `x_list`
    if x_list is None:
        x_list = [i for i in range(max_len_y_list)]
    
    # get `y_label_list`
    y_label_list = tsfm_to_1d_array(y_label_list, num_plot)

    if marker_list is None:
        marker_list = ['o']
    
    if color_list is not None:
        color_list = tsfm_to_1d_array(color_list, num_plot)

    if alpha_list is not None:
        alpha_list = tsfm_to_1d_array(alpha_list, num_plot)

    if fill_alpha_list is not None:
        fill_alpha_list = tsfm_to_1d_array(fill_alpha_list, num_plot)

    if face_color_list is not None:
        face_color_list = tsfm_to_1d_array(face_color_list, num_plot)

    fig, ax = plt.subplots(figsize = figsize)

    # plot
    for plot_idx, y_list in enumerate(y_list_list):
        ax.plot(
            x_list, y_list, 
            marker = marker_list[plot_idx % len(marker_list)], 
            label = y_label_list[plot_idx], 
            color = None if (color_list is None) else color_list[plot_idx], 
            alpha = None if (alpha_list is None) else alpha_list[plot_idx]
        )

    # fill between
    if confidence_level is not None:
        y_list_list = np.asarray(y_list_list)

        std_list_list = np.asarray(std_list_list)

        # TODO: update
        y_down_list_list = y_list_list - 1.96 * std_list_list / np.sqrt(num_sample)
        y_up_list_list = y_list_list + 1.96 * std_list_list / np.sqrt(num_sample)

        # plot
        for (
            plot_idx, 
            (y_down_list, y_up_list)
        ) in enumerate(
            zip(y_down_list_list, y_up_list_list)
        ):
            ax.fill_between(
                x_list, 
                y_up_list, y_down_list, 

                alpha = fill_alpha_list[plot_idx], 
                facecolor = face_color_list[plot_idx]
            )

    # set title
    if plot_title is not None:
        ax.set_title(plot_title)
    
    # set x-label
    if plot_x_label is not None:
        ax.set_xlabel(plot_x_label)

    # set y-label
    if plot_y_label is not None:
        ax.set_ylabel(plot_y_label)

    # determine whether to show grid
    ax.grid(show_grid)

    # determine whether to show legend
    if show_legend:
        ax.legend()

    fig.tight_layout()

    return fig, ax

# TODO: add more param
def merge_line_chart_list(
    chart_list: List[Tuple], 

    figsize: Optional[Union[float, Tuple[float]]] = (10, 5), 

    num_row: Optional[int] = 1, 
    num_col: Optional[int] = 1, 

    show_grid_list: Optional[Union[bool, List[bool]]] = True, 
    show_legend_list: Optional[Union[bool, List[bool]]] = True
) -> Tuple:
    # get `figsize`
    if isinstance(figsize, tuple):
        pass
    elif isinstance(figsize, float):
        figsize = (figsize, figsize)
    else:
        raise ValueError(
            f"Unsupported type of `figsize`, got `{type(figsize)}`. "
        )

    # check num_chart = num_row * num_col, if not, raise warning
    num_chart = len(chart_list)
    if num_chart != num_row * num_col:
        if num_chart < num_row * num_col:
            logger(
                f"The amount of chart(s) is less than `num_row` * `num_col`, "
                f"the grid(s) for the unprovided chart(s) will remain blank. ", 
                log_type = "warning"
            )
        else:
            logger(
                f"The amount of chart(s) is greater than `num_row` * `num_col`, "
                f"only the first `num_row` * `num_col` chart(s) will be plotted. ", 
                log_type = "warning"
            )

    # get `show_grid_list`
    if isinstance(show_grid_list, bool):
        show_grid_list = [show_grid_list] * num_chart
    elif isinstance(show_grid_list, List[bool]):
        if num_chart != len(show_grid_list):
            raise ValueError(
                f"The length of `show_grid_list` doesn't match `num_chart`, "
                f"got {len(show_grid_list)} and {num_chart}. "
            )
    else:
        raise ValueError(
            f"Unsupported type of `show_grid_list`, got `{type(show_grid_list)}`. "
        )

    # get `show_legend_list`
    if isinstance(show_legend_list, bool):
        show_legend_list = [show_legend_list] * num_chart
    elif isinstance(show_legend_list, List[bool]):
        if num_chart != len(show_legend_list):
            raise ValueError(
                f"The length of `show_legend_list` doesn't match `num_chart`, "
                f"got {len(show_legend_list)} and {num_chart}. "
            )
    else:
        raise ValueError(
            f"Unsupported type of `show_legend_list`, got `{type(show_legend_list)}`. "
        )

    new_fig, new_ax_matrix = plt.subplots(
        num_row, num_col, 
        figsize = figsize
    )

    # ensure new_ax_matrix.shape = [num_row, num_col]
    if new_ax_matrix.ndim == 0:
        new_ax_matrix = np.asarray([new_ax_matrix])
    if new_ax_matrix.ndim == 1:
        new_ax_matrix = np.asarray([new_ax_matrix])
    
    for row_idx in range(num_row):
        break_flag = False

        for col_idx in range(num_col):
            chart_idx = row_idx * num_col + col_idx
            if chart_idx >= num_chart:
                break_flag = True
                break

            _, ax = chart_list[chart_idx]

            # add lines
            for line in ax.get_lines():
                new_ax_matrix[row_idx][col_idx].plot(
                    line.get_xdata(), line.get_ydata(), 
                    marker = line.get_marker(), 
                    color = line.get_color(), 
                    label = line.get_label(), 
                )
            
            # add titles and labels
            new_ax_matrix[row_idx][col_idx].set_title(ax.get_title())
            new_ax_matrix[row_idx][col_idx].set_xlabel(ax.get_xlabel())
            new_ax_matrix[row_idx][col_idx].set_ylabel(ax.get_ylabel())

            # determine whether to show grid
            new_ax_matrix[row_idx][col_idx].grid(show_grid_list[chart_idx])
            
            # determine whether to show legend
            if show_legend_list[chart_idx]:
                new_ax_matrix[row_idx][col_idx].legend()

        if break_flag:
            break

    # adjust interval between subplots
    plt.tight_layout()

    return new_fig, new_ax_matrix

def get_scatter(
    figsize: Optional[Union[float, Tuple[float, float]]] = (10, 5), 

    point_list: Optional[List[Tuple[float]]] = None, 

    label_list: Optional[List[str]] = None, 

    marker_list: Union[str, List[str]] = [
        'o',  # circle
        's',  # square
        '*',  # star
        '+',  # plus
        'x',  # x

        'd',  # diamond
        'D',  # thin diamond

        '^',  # up-pointing triangle
        'v',  # down-pointing triangle
        '<',  # left-pointing triangle
        '>',  # right-pointing triangle

        'p',  # pentagon
        'h',  # hexagon 1
        'H',  # hexagon 2
    ], 

    color_list: Optional[List[str]] = None, 

    area_list: Optional[List[float]] = None, 

    plot_title: Optional[str] = None, 
    plot_x_label: Optional[str] = None, 
    plot_y_label: Optional[str] = None, 

    show_grid: Optional[bool] = True, 

    # legend
    show_legend: Optional[bool] = True, 
    legend_num_col: Optional[int] = 1
) -> Tuple: 
    # get `figsize`
    if isinstance(figsize, tuple):
        pass
    elif isinstance(figsize, float):
        figsize = (figsize, figsize)
    else:
        raise ValueError(
            f"Unsupported type of `figsize`, got `{type(figsize)}`. "
        )
    
    fig, ax = plt.subplots(figsize = figsize)

    for i, point in enumerate(point_list):
        ax.scatter(
            x = point[0], y = point[1], 

            label = label_list[i], 

            marker = None if (marker_list is None) else marker_list[i], 

            c = None if (color_list is None) else color_list[i], 

            s = None if (area_list is None) else area_list[i], 
        )

    # show labels
    if plot_x_label is not None:
        ax.set_xlabel(plot_x_label)
    if plot_y_label is not None:
        ax.set_ylabel(plot_y_label)

    # show title
    if plot_title is not None:
        ax.set_title(plot_title)
    
    # determine whether to show grid
    ax.grid(show_grid)

    # determine whether to show legend
    if show_legend:
        ax.legend(ncol = legend_num_col)

    fig.tight_layout()

    return fig, ax

def get_heatmap(
    figsize: Optional[Union[float, Tuple[float, float]]] = (10, 5), 

    matrix: Union[List[List], np.ndarray] = None, 

    # color mapping
    color_mapping: Optional[str] = "viridis",  # ["viridis", "plasma", "hot"]
    map_min_val: Optional[float] = None, 
    map_max_val: Optional[float] = None, 

    # text annotation
    show_text_annotation: Optional[bool] = False, 
    text_annotation_horizontal_alignment: Optional[str] = "center",  # ["center"]
    text_annotation_vertical_alignment: Optional[str] = "center",  # ["center"]
    text_annotation_color: Optional[str] = "white", 
    
    # x label
    x_label_list: Optional[List[str]] = None, 
    x_label_rotation: Optional[float] = None, 
    x_label_horizontal_alignment: Optional[str] = "right",  # ["right"]
    x_label_rotation_mode: Optional[str] = None,  # ["anchor"]

    # y label
    y_label_list: Optional[List[str]] = None, 
    y_label_rotation: Optional[float] = None, 
    y_label_horizontal_alignment: Optional[str] = "right",  # ["right"]
    y_label_rotation_mode: Optional[str] = None,  # ["anchor"]

    # color bar
    show_color_bar: Optional[bool] = False, 
    color_bar_orientation: Optional[str] = "vertical",  # ["vertical", "horizontal"]
    color_bar_pad: Optional[float] = 0.1, 
    color_bar_label: Optional[str] = None, 

    # labels
    plot_x_label: Optional[str] = None, 
    plot_y_label: Optional[str] = None, 

    plot_title: Optional[str] = None, 
) -> Tuple:
    # get `figsize`
    if isinstance(figsize, tuple):
        pass
    elif isinstance(figsize, float):
        figsize = (figsize, figsize)
    else:
        raise ValueError(
            f"Unsupported type of `figsize`, got `{type(figsize)}`. "
        )

    if not isinstance(matrix, np.ndarray):
        matrix = np.asarray(matrix)
    
    num_row, num_col = matrix.shape

    fig, ax = plt.subplots(figsize = figsize)

    # show heatmap
    plot = ax.imshow(
        matrix, 
        cmap = color_mapping, 

        vmin = map_min_val, 
        vmax = map_max_val
    )

    # show text annotations
    if show_text_annotation:
        for i in range(num_row):
            for j in range(num_col):
                ax.text(
                    j, i, 
                    f"{matrix[i][j]:.4f}", 
                    
                    ha = text_annotation_horizontal_alignment, 
                    va = text_annotation_vertical_alignment, 
                    color = text_annotation_color
                )
    
    # show x labels
    if x_label_list is not None:
        ax.set_xticks(
            range(len(x_label_list)), 
            labels = x_label_list, 

            rotation = x_label_rotation, 
            ha = x_label_horizontal_alignment, 
            rotation_mode = x_label_rotation_mode
        )

    # show y labels
    if y_label_list is not None:
        ax.set_yticks(
            range(len(y_label_list)), 
            labels = y_label_list, 

            rotation = y_label_rotation, 
            ha = y_label_horizontal_alignment, 
            rotation_mode = y_label_rotation_mode
        )

    # color bar
    if show_color_bar:
        color_bar = fig.colorbar(
            plot, 
            ax = ax, 
            orientation = color_bar_orientation, 
            pad = color_bar_pad
        )

        if color_bar_label is not None:
            color_bar.ste_label(color_bar_label)

    # show labels
    if plot_x_label is not None:
        ax.set_xlabel(plot_x_label)
    if plot_y_label is not None:
        ax.set_ylabel(plot_y_label)

    # show title
    if plot_title is not None:
        ax.set_title(plot_title)
    
    fig.tight_layout()

    return fig, ax

def get_bar_chart(
    figsize: Optional[Union[float, Tuple[float, float]]] = (10, 5), 

    y_list: List[float] = None, 
    y_upper_lim: Optional[float] = None, 

    x_label_list: Optional[List[str]] = None, 

    bar_color_list: Optional[List[str]] = None, 

    # text annotation
    show_text_annotation: Optional[bool] = False, 
    text_annotation_color: Optional[str] = "black", 
    text_annotation_fontsize: Optional[float] = 14, 

    # labels
    plot_x_label: Optional[str] = None, 
    plot_y_label: Optional[str] = None, 

    plot_title: Optional[str] = None, 

    show_legend: Optional[bool] = True
) -> Tuple:
    # get `figsize`
    if isinstance(figsize, tuple):
        pass
    elif isinstance(figsize, float):
        figsize = (figsize, figsize)
    else:
        raise ValueError(
            f"Unsupported type of `figsize`, got `{type(figsize)}`. "
        )

    fig, ax = plt.subplots(figsize = figsize)

    bar_list = ax.bar(
        x = x_label_list, 
        height = y_list, 

        color = bar_color_list
    )

    if y_upper_lim is not None:
        cur_y_lim = ax.get_ylim()
        
        ax.set_ylim(cur_y_lim[0], y_upper_lim)

    if show_text_annotation:
        for i, bar in enumerate(bar_list):
            ax.annotate(
                int(y_list[i]), 
                xy = (
                    bar.get_x() + bar.get_width() / 2, 
                    bar.get_height() + 1
                ), 

                color = text_annotation_color, 
                fontsize = text_annotation_fontsize, 

                ha = "center", 
                va = "bottom"
            )

    # show label
    if plot_x_label is not None:
        ax.set_xlabel(plot_x_label)
    if plot_y_label is not None:
        ax.set_ylabel(plot_y_label)

    if plot_title is not None:
        ax.set_title(plot_title)

    if show_legend:
        ax.legend()

    fig.tight_layout()

    return fig, ax
