import functools
import itertools
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
#from mpl_toolkits.basemap import Basemap
from matplotlib.offsetbox import OffsetImage, AnnotationBbox


def neighbours_8(x, y, x_max, y_max):
    deltas_x = (-1, 0, 1)
    deltas_y = (-1, 0, 1)
    for (dx, dy) in itertools.product(deltas_x, deltas_y):
        x_new, y_new = x + dx, y + dy
        if 0 <= x_new < x_max and 0 <= y_new < y_max and (dx, dy) != (0, 0):
            yield x_new, y_new


def neighbours_4(x, y, x_max, y_max):
    for (dx, dy) in [(1, 0), (0, 1), (0, -1), (-1, 0)]:
        x_new, y_new = x + dx, y + dy
        if 0 <= x_new < x_max and 0 <= y_new < y_max and (dx, dy) != (0, 0):
            yield x_new, y_new


def get_neighbourhood_func(neighbourhood_fn):
    if neighbourhood_fn == "4-grid":
        return neighbours_4
    elif neighbourhood_fn == "8-grid":
        return neighbours_8
    else:
        raise Exception(f"neighbourhood_fn of {neighbourhood_fn} not possible")


def edges_from_vertex(x, y, N, neighbourhood_fn):
    v = (x, y)
    neighbours = get_neighbourhood_func(neighbourhood_fn)(*v, x_max=N, y_max=N)
    v_edges = [
        (*v, *vn) for vn in neighbours if vertex_index(v, N) < vertex_index(vn, N)
    ]  # Enforce ordering on vertices
    return v_edges


def vertex_index(v, dim):
    x, y = v
    return x * dim + y


@functools.lru_cache(32)
def edges_from_grid(N, neighbourhood_fn):
    all_vertices = itertools.product(range(N), range(N))
    all_edges = [edges_from_vertex(x, y, N, neighbourhood_fn=neighbourhood_fn) for x, y in all_vertices]
    all_edges_flat = sum(all_edges, [])
    all_edges_flat_unique = list(set(all_edges_flat))
    return np.asarray(all_edges_flat_unique)


def perfect_matching_vis(grid_img, grid_dim, labels, color=(0, 255, 255), width=2, offset=0):
    edges = edges_from_grid(grid_dim, neighbourhood_fn='4-grid')
    pixels_per_cell = int(grid_img.shape[0] / grid_dim)

    img = Image.fromarray(np.uint8(grid_img.squeeze())).convert("RGB")
    for i, (y1, x1, y2, x2) in enumerate(edges):
        if labels[i]:
            draw = ImageDraw.Draw(img)
            if x1 == x2:
                draw.line(
                    (x1 * pixels_per_cell + pixels_per_cell / 2, y1 * pixels_per_cell + pixels_per_cell / 2 + offset,
                     x2 * pixels_per_cell + pixels_per_cell / 2, y2 * pixels_per_cell + pixels_per_cell / 2 - offset),
                    fill=color, width=width)
            else:
                draw.line(
                    (x1 * pixels_per_cell + pixels_per_cell / 2 + offset, y1 * pixels_per_cell + pixels_per_cell / 2,
                     x2 * pixels_per_cell + pixels_per_cell / 2 - offset, y2 * pixels_per_cell + pixels_per_cell / 2),
                    fill=color, width=width)
            del draw

    return np.asarray(img, dtype=np.uint8)


def plot_tsp_path(gps, flags, tsp_tour):
    plt.figure(figsize=(24, 12))
    m = Basemap(projection='ortho', lon_0=20.0, lat_0=20.0, resolution=None)
    m.shadedrelief()

    for (lon, lat), flag in zip(gps, flags):
        x, y = m(lon, lat)
        im = OffsetImage(flag[..., ::-1], zoom=0.8)
        ab = AnnotationBbox(im, (x, y), xycoords='data', frameon=False)
        m._check_ax().add_artist(ab)

    num_countries = len(tsp_tour)
    last_country = 0
    current_country = 0
    path_indices = [0]
    for _ in range(num_countries):
        for j in range(num_countries):
            if tsp_tour[current_country][j] and j != last_country:
                last_country = current_country
                current_country = j
                path_indices.append(current_country)
                break

    lat = [gps[i][1] for i in path_indices]
    lon = [gps[i][0] for i in path_indices]

    x, y = m(lon, lat)
    m.plot(x, y, 'o-', markersize=5, linewidth=3)

    plt.title("Country locations with TSP solution")
    plt.show()


# helper functions, you need to install tqdm for progress bar feature
import urllib.request
import numpy as np
from matplotlib import pyplot as plt

try:

    from tqdm import tqdm
    class DownloadProgressBar(tqdm):
        def update_to(self, b=1, bsize=1, tsize=None):
            if tsize is not None:
                self.total = tsize
            self.update(b * bsize - self.n)


    def download_url(url, output_path):
        with DownloadProgressBar(unit='B', unit_scale=True,
                                miniters=1, desc=url.split('/')[-1]) as t:
            urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)
except ModuleNotFoundError as e:
    print("Not using progress bar")
    def download_url(url, output_path):
        urllib.request.urlretrieve(url, filename=output_path)
