"""
Created on 20.11.2019
@author: Igor Zagorodnov
"""

import matplotlib.pyplot as plt
from matplotlib import transforms
import numpy as np
from scipy import ndimage
import pickle
import copy
from mpl_toolkits.mplot3d import Axes3D


class TDSImage:
    def __init__(self):
        self.filename = None
        self.raw_image = None
        self.bg_image = None
        self.noise_thresh = 0.1
        self.charge = 250  # charge in [pC]
        self.dx_px = 0.0438912  # ps / px, pixel size in horizontal direction
        self.dy_px = 0.003260   # MeV / px, pixel size in vertical direction
        self.unif_filter_size = 150
        self.current = None
        self.img = None
        self.x = None
        self.y = None

    def read_image_from_pickle(self):
        # read image
        if self.filename is not None:
            with open(self.filename, 'rb') as f:
                self.raw_image = pickle.load(f)
            return True
        else:
            print("filename is None")
            self.raw_image = None
            return False

    def process(self):
        """
            function processes TDS images

            :param filename: path to a pickled image file
            :param bg_image: background image
            :param dx_px: ps/px, pixel size in horizontal direction
            :param dy_px: MeV/px, pixel size in vertical direction
            :param charge: pC
            :param noise_thresh: 0.1, remove background with value < noise_thresh * max(image)
            :param plotting: False
            :return:
            """
        if self.raw_image is None:
            if not self.read_image_from_pickle():
                print("No image")
                return

        img0 = self.raw_image.astype(np.double) - self.bg_image
        img0 = np.flipud(img0)
        img1 = ndimage.uniform_filter(img0, size=self.unif_filter_size)
        (ny, nx) = img0.shape

        # clip image
        (j0, i0) = np.unravel_index(np.argmax(img1, axis=None), img1.shape)
        hw = 100
        j1 = max(j0 - hw * 5, 0)
        j2 = min(j0 + hw * 5, ny)
        i1 = max(i0 - hw * 3, 0)
        i2 = min(i0 + hw * 3, nx)
        img_raw = img0[j1:j2, i1:i2].copy()

        # masking noisy background
        maxv = np.max(np.max(img1))
        inds_mask = np.asarray(img1 < maxv * self.noise_thresh).nonzero()
        img0[inds_mask] = 0.0

        # remove very high isolated values
        img1 = ndimage.uniform_filter(img0, size=3)
        inds_hi = np.asarray(1.5 * img1 < img0).nonzero()
        img0[inds_hi] = img1[inds_hi]
        self.img = ndimage.gaussian_filter(img0[j1:j2, i1:i2], sigma=0)
        (j0f, i0f) = ndimage.center_of_mass(self.img)
        i0i = int(i0f)
        j0i = int(j0f)
        (ny, nx) = self.img.shape

        # energy axis
        self.y = (np.linspace(0, ny - 1, ny) - j0f) * self.dy_px
        # time axis
        self.x = (np.linspace(0, nx - 1, nx) - i0f) * self.dx_px

        ro = sum(self.img)
        self.current = self.charge / self.dx_px * ro / sum(ro)

        self.energy_proj = sum(self.img.T)
        self.energy_proj = self.charge / self.dy_px * self.energy_proj / sum(self.energy_proj)

        m = sum(self.x * self.current) / sum(self.current)
        m2 = sum((self.x - m) ** 2 * self.current) / sum(self.current)
        self.x_sigma = np.sqrt(m2)
        self.slice_energy = np.sum(self.img.T * self.y, axis=1) * self.dy_px

        S = np.sum(self.img, axis=0) * self.dy_px

        self.inds_roi = np.asarray(S > 0.1 * np.max(S)).nonzero()[0]
        self.slice_energy[self.inds_roi] = self.slice_energy[self.inds_roi] / S[self.inds_roi]
        self.slice_energy_spread = np.zeros(nx)


        for j in range(len(self.inds_roi)):
            i = self.inds_roi[j]
            y2 = (self.y - self.slice_energy[i]) ** 2
            y_sigma0 = np.sum(self.img[:, i] * y2) * self.dy_px
            self.slice_energy_spread[i] = np.sqrt(y_sigma0 / S[i])

    def plot(self):

        fig = plt.figure(10, figsize=(9, 7))
        ax_im = plt.subplot(221)
        (ny, nx) = self.img.shape
        my_rainbow = copy.deepcopy(plt.get_cmap('rainbow'))
        my_rainbow.set_under('w')
        vmin = np.min(self.img) + (np.max(self.img) - np.min(self.img)) * 0.001
        plt.imshow(self.img, extent=[self.x[0], self.x[nx - 1], self.y[0], self.y[ny - 1]],
                   aspect='auto', origin='lower')#, vmin=vmin, cmap=my_rainbow)

        plt.ylabel("dE [MeV]")
        plt.plot(self.x[self.inds_roi], self.slice_energy[self.inds_roi], 'r:')
        #plt.plot(self.x[self.inds_roi], self.slice_energy[self.inds_roi] + 3 * self.slice_energy_spread[self.inds_roi], 'w:',
        #         self.x[self.inds_roi], self.slice_energy[self.inds_roi] - 3 * self.slice_energy_spread[self.inds_roi], 'w:')
        plt.setp(ax_im.get_xticklabels(), visible=False)


        plt.title("processed data")

        ax_cu = plt.subplot(223, sharex=ax_im)
        plt.plot(self.x, self.current)
        plt.xlabel('t [ps]')
        plt.ylabel('I [A]')
        plt.title('current, rms=' + "{:.2f}".format(self.x_sigma) + ' fs')

        ax_ep = plt.subplot(222, sharey=ax_im)

        plt.plot( self.energy_proj, self.y)#, transform= rot + base)
        yl_max = np.max(self.slice_energy[self.inds_roi] + 6 * self.slice_energy_spread[self.inds_roi])
        yl_min = np.min(self.slice_energy[self.inds_roi] - 6 * self.slice_energy_spread[self.inds_roi])
        plt.ylim(yl_min, yl_max)
        plt.setp(ax_ep.get_yticklabels(), visible=False)
        plt.setp(ax_ep.get_xticklabels(), visible=False)
        #ax_ep.yaxis.tick_right()
        #ax_ep.yaxis.set_label_position("right")
        #ax_ep.xaxis.tick_top()
        ax_ep.xaxis.set_label_position("top")
        #plt.xlabel('Density [a.u.]')
        plt.title('Projection on the energy axis')

        ax_es = plt.subplot(224, sharex=ax_im)
        ax_es.yaxis.tick_right()
        ax_es.yaxis.set_label_position("right")
        plt.plot(self.x[self.inds_roi], self.slice_energy_spread[self.inds_roi] * 1e3)
        plt.xlabel('t [ps]')
        plt.ylabel('$\sigma_E$, [keV]')
        plt.title('slice energy spread')
        plt.show()



def process_tds_image(filename, bg_image, dx_px, dy_px, charge, noise_thresh=0.1, plotting=False):
    """
    function processes TDS images

    :param filename: path to a pickled image file
    :param bg_image: background image
    :param dx_px: ps/px, pixel size in horizontal direction
    :param dy_px: MeV/px, pixel size in vertical direction
    :param charge: pC
    :param noise_thresh: 0.1, remove background with value < noise_thresh * max(image)
    :param plotting: False
    :return:
    """
    # read image
    with open(filename, 'rb') as f:
        img0 = pickle.load(f)

    img0 = img0.astype(np.double) - bg_image
    img0 = np.flipud(img0)
    img1 = ndimage.uniform_filter(img0, size=150)
    (ny, nx) = img0.shape

    # clip image
    (j0,i0) = np.unravel_index(np.argmax(img1, axis=None), img1.shape)
    hw = 100
    j1 = max(j0 - hw*5, 0)
    j2 = min(j0 + hw*5, ny)
    i1 = max(i0 - hw*3, 0)
    i2 = min(i0 + hw*3, nx)
    img_raw = img0[j1:j2, i1:i2].copy()

    # masking noisy background
    maxv = np.max(np.max(img1))
    inds = np.asarray(img1 < maxv * noise_thresh).nonzero()
    img0[inds] = 0.0

    # remove very high isolated values
    img1 = ndimage.uniform_filter(img0, size=3)
    inds = np.asarray(1.5*img1 < img0).nonzero()
    img0[inds] = img1[inds]
    img = ndimage.gaussian_filter(img0[j1:j2, i1:i2], sigma=0)
    (j0f, i0f) = ndimage.center_of_mass(img)
    i0i = int(i0f)
    j0i = int(j0f)
    (ny, nx) = img.shape

    # energy axis
    y = (np.linspace(0, ny-1, ny) - j0f) * dy_px
    # time axis
    x = (np.linspace(0, nx-1, nx) - i0f) * dx_px

    ro = sum(img)
    I = charge / dx_px * ro / sum(ro)

    roE = sum(img.T)
    roE = charge / dy_px * roE / sum(roE)

    m = sum(x * I) / sum(I)
    m2 = sum((x - m) ** 2 * I) / sum(I)
    sigx = np.sqrt(m2)
    dE_m = np.sum(img.T * y, axis=1) * dy_px

    S = np.sum(img, axis=0) * dy_px

    inds = np.asarray(S > 0.1 * np.max(S)).nonzero()[0]
    dE_m[inds] = dE_m[inds] / S[inds]
    dE_sigma = np.zeros(nx)
    # plt.plot(dE_m)
    # plt.show()

    for j in range(len(inds)):
        i = inds[j]
        y2 = (y - dE_m[i])**2
        y_sigma0 = np.sum(img[:, i]*y2) * dy_px
        dE_sigma[i] = np.sqrt(y_sigma0 / S[i])



    if plotting:
        X, Y = np.meshgrid(x, y)
        fig = plt.figure(10)
        ax3 = fig.add_subplot(2, 1, 1, projection='3d')
        ax3.plot_wireframe(X, Y, img_raw)
        plt.title("raw data")
        plt.xlabel("t[fs]")
        plt.ylabel("dE[MeV ]")
        ax4 = fig.add_subplot(2, 1, 2)
        ax4.imshow(img_raw, extent=[x[0], x[nx - 1], y[0], y[ny - 1]], aspect='auto', origin='lower')
        ax4.set(xlabel="t[ps]", ylabel="dE[MeV]")
        plt.xlabel("t[fs]")
        plt.ylabel("dE[MeV ]")

        fig = plt.figure(20)
        ax1 = fig.add_subplot(1, 1, 1, projection='3d')
        ax1.plot_wireframe(X, Y, img)
        plt.title("processed data")
        plt.xlabel("t[ps]")
        plt.ylabel("dE[MeV ]")

        fig = plt.figure(21)
        plt.imshow(img, extent=[x[0], x[nx - 1], y[0], y[ny - 1]], aspect='auto', origin='lower')
        plt.plot(x[i0i], y[j0i], 'wo')
        print(x[i0i], y[j0i])
        plt.ylabel("dE[MeV]")
        plt.plot(x[inds], dE_m[inds], 'r:')
        plt.plot(x[inds], dE_m[inds] + 3 * dE_sigma[inds], 'w:', x[inds], dE_m[inds] - 3 * dE_sigma[inds], 'w:')
        plt.xlabel('t[ps]')
        plt.ylabel('dE[keV]')
        plt.title("processed data")

        fig = plt.figure(30)
        plt.plot(x, I)
        plt.xlabel('t[ps]')
        plt.ylabel('I[A]')
        plt.title('current, rms=' + "{:.2f}".format(sigx) + ' fs')

        fig = plt.figure(31)
        plt.plot(y, roE)
        plt.xlabel('dE[MeV]')
        plt.ylabel('charge density[a.u.]')

        fig=plt.figure(32)
        plt.plot(x[inds], dE_sigma[inds]*1e3)
        plt.xlabel('t[ps]')
        plt.ylabel('dE[keV]')
        plt.title('slice energy spread')
        plt.show()
    return img, x, y, sigx, I, roE, dE_m, dE_sigma, inds
