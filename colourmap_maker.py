#%%
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

from matplotlib.colors import LinearSegmentedColormap


def rgb_colour_map(rgb_list, N_bins):

    N_segments = len(rgb_list) - 1

    r_cmap_array = np.zeros(N_segments*N_bins)
    g_cmap_array = np.zeros(N_segments*N_bins)
    b_cmap_array = np.zeros(N_segments*N_bins)

    for i in range(N_segments):
        rgb_start = rgb_list[i]
        rgb_stop = rgb_list[i + 1]

        r_cmap_array[i*N_bins:(i + 1)*N_bins] = np.linspace(rgb_start[0], rgb_stop[0], num = N_bins + 1, dtype = np.float32)[:-1]
        g_cmap_array[i*N_bins:(i + 1)*N_bins] = np.linspace(rgb_start[1], rgb_stop[1], num = N_bins + 1, dtype = np.float32)[:-1]
        b_cmap_array[i*N_bins:(i + 1)*N_bins] = np.linspace(rgb_start[2], rgb_stop[2], num = N_bins + 1, dtype = np.float32)[:-1]

    rgb_cmap_array = np.dstack((r_cmap_array, g_cmap_array, b_cmap_array))

    rgb_cmap = LinearSegmentedColormap.from_list('rgb_cmap', rgb_cmap_array[0], N = N_bins)

    return rgb_cmap


def luv_colour_map(rgb_list, N_bins):

    N_segments = len(rgb_list) - 1

    L_cmap_array = np.zeros(N_segments*N_bins, dtype = np.float32)
    u_cmap_array = np.zeros(N_segments*N_bins, dtype = np.float32)
    v_cmap_array = np.zeros(N_segments*N_bins, dtype = np.float32)

    for i in range(N_segments):
        rgb_start = rgb_list[i]
        rgb_stop = rgb_list[i + 1]

        rgb_palette = np.array([rgb_start, rgb_stop], dtype = np.float32).reshape(1, 2, 3)
        luv_palette = cv.cvtColor(rgb_palette, cv.COLOR_RGB2Luv)

        a = luv_palette[0, 1, 2] - luv_palette[0, 0, 2]
        b = luv_palette[0, 0, 1] - luv_palette[0, 1, 1]
        c = -(a*luv_palette[0, 0, 1] + b*luv_palette[0, 1, 2])

        u_0 = -a*c/(a**2 + b**2)
        v_0 = -b*c/(a**2 + b**2)

        u_domain_1 = (luv_palette[0, 0, 1] < u_0 < luv_palette[0, 1, 1])
        u_domain_2 = (luv_palette[0, 0, 1] > u_0 > luv_palette[0, 1, 1])
        v_domain_1 = (luv_palette[0, 0, 2] < v_0 < luv_palette[0, 1, 2])
        v_domain_2 = (luv_palette[0, 0, 2] > v_0 > luv_palette[0, 1, 2])

        if np.any([u_domain_1, u_domain_2]) and np.any([v_domain_1, v_domain_2]):
            u_cmap_array[i*N_bins:i*N_bins + N_bins//2] = np.linspace(luv_palette[0, 0, 1], u_0, num = N_bins//2 + 1, dtype = np.float32)[:-1]
            u_cmap_array[i*N_bins + N_bins//2:(i + 1)*N_bins] = np.linspace(u_0, luv_palette[0, 1, 1], num = N_bins//2 + 1, dtype = np.float32)[:-1]
            v_cmap_array[i*N_bins:i*N_bins + N_bins//2] = np.linspace(luv_palette[0, 0, 2], v_0, num = N_bins//2 + 1, dtype = np.float32)[:-1]
            v_cmap_array[i*N_bins + N_bins//2:(i + 1)*N_bins] = np.linspace(v_0, luv_palette[0, 1, 2], num = N_bins//2 + 1, dtype = np.float32)[:-1]
        else:
            u_cmap_array[i*N_bins:(i + 1)*N_bins] = np.linspace(luv_palette[0, 0, 1], luv_palette[0, 1, 1], num = N_bins + 1, dtype = np.float32)[:-1]
            v_cmap_array[i*N_bins:(i + 1)*N_bins] = np.linspace(luv_palette[0, 0, 2], luv_palette[0, 1, 2], num = N_bins + 1, dtype = np.float32)[:-1]

        L_cmap_array[i*N_bins:(i + 1)*N_bins] = np.linspace(luv_palette[0, 0, 0], luv_palette[0, 1, 0], num = N_bins + 1, dtype = np.float32)[:-1]

    luv_cmap_array = np.dstack((L_cmap_array, u_cmap_array, v_cmap_array))
    rgb_cmap_array = cv.cvtColor(luv_cmap_array, cv.COLOR_Luv2RGB)

    luv_cmap = LinearSegmentedColormap.from_list('luv_cmap', rgb_cmap_array[0], N = N_bins)

    return luv_cmap


def RGB_to_LCh(rgb_array):

    luv_array = cv.cvtColor(rgb_array, cv.COLOR_RGB2Luv)

    L = luv_array[0, :, 0]
    C = np.hypot(luv_array[0, :, 1], luv_array[0, :, 2])
    h = np.arctan2(luv_array[0, :, 2], luv_array[0, :, 1])

    lch_array = np.dstack((L, C, h))
    return lch_array


def lch_colour_map(rgb_list, N_bins):

    N_segments = len(rgb_list) - 1

    L_cmap_array = np.zeros(N_segments*N_bins, dtype = np.float32)
    u_cmap_array = np.zeros(N_segments*N_bins, dtype = np.float32)
    v_cmap_array = np.zeros(N_segments*N_bins, dtype = np.float32)

    for i in range(N_segments):
        rgb_start = rgb_list[i]
        rgb_stop = rgb_list[i + 1]


        rgb_palette = np.array([rgb_start, rgb_stop], dtype = np.float32).reshape(1, 2, 3)
        lch_palette = RGB_to_LCh(rgb_palette)

        L_cmap_array[i*N_bins:(i + 1)*N_bins] = np.linspace(lch_palette[0, 0, 0], lch_palette[0, 1, 0], num = N_bins + 1, dtype = np.float32)[:-1]
        chroma_cmap_array = np.linspace(lch_palette[0, 0, 1], lch_palette[0, 1, 1], num = N_bins + 1, dtype = np.float32)[:-1]

        lch_palette[0, :, 2] = lch_palette[0, :, 2]%(2*np.pi)
        delta_h = lch_palette[0, 1, 2] - lch_palette[0, 0, 2]

        if np.abs(delta_h) < np.pi:
            hue_cmap_array = np.linspace(lch_palette[0, 0, 2], lch_palette[0, 1, 2], num = N_bins + 1, dtype = np.float32)[:-1]
        elif delta_h > 0:
            hue_cmap_array = np.linspace(lch_palette[0, 0, 2] + 2*np.pi, lch_palette[0, 1, 2], num = N_bins + 1, dtype = np.float32)[:-1]
        else:
            hue_cmap_array = np.linspace(lch_palette[0, 0, 2] - 2*np.pi, lch_palette[0, 1, 2], num = N_bins + 1, dtype = np.float32)[:-1]
        
        u_cmap_array[i*N_bins:(i + 1)*N_bins] = chroma_cmap_array*np.cos(hue_cmap_array)
        v_cmap_array[i*N_bins:(i + 1)*N_bins] = chroma_cmap_array*np.sin(hue_cmap_array)
    luv_cmap_array = np.dstack((L_cmap_array, u_cmap_array, v_cmap_array))

    rgb_cmap_array = cv.cvtColor(luv_cmap_array, cv.COLOR_Luv2RGB)

    lch_cmap = LinearSegmentedColormap.from_list('lch_cmap', rgb_cmap_array[0], N = N_bins)

    return lch_cmap

def example():
    rgb_list = [[54, 3, 27], [241, 0, 123], [250, 237, 185]]
    cmap_rgb = rgb_colour_map(np.array(rgb_list, dtype = np.float32)/255, 100)
    cmap_luv = luv_colour_map(np.array(rgb_list, dtype = np.float32)/255, 100)
    cmap_lch = lch_colour_map(np.array(rgb_list, dtype = np.float32)/255, 100)

    plt.figure()
    plt.scatter(np.zeros(100) - 0.005, np.linspace(0, 1, num = 100), c = np.linspace(0, 1, num = 100), cmap = cmap_rgb, label = 'rgb')
    plt.scatter(np.zeros(100), np.linspace(0, 1, num = 100), c = np.linspace(0, 1, num = 100), cmap = cmap_luv, label = 'luv')
    plt.scatter(np.zeros(100) + 0.005, np.linspace(0, 1, num = 100), c = np.linspace(0, 1, num = 100), cmap = cmap_lch, label = 'lch')
    plt.xlim(-0.05, 0.05)
    plt.legend()
    plt.show()
# %%
