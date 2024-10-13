import numpy as np
from PIL import Image
from numpy.fft import fftshift, fft2, ifft2, ifftshift
import matplotlib.pyplot as plt
import os


def fresnel(input_image, lam=532, pix=0.098, z1=0.001, z2=0.001, z_interval=0.001):
    # 同轴平面波菲涅尔变换
    # image = Image.open(in_path)
    # width, height = image.size
    # grayscale_image = image.convert("L")
    # grayscale_array = np.asarray(grayscale_image)
    lam = lam * 1e-9
    pix = pix * 1e-6

    grayscale_array = input_image
    # print(grayscale_array.dtype)
    grayscale_array = grayscale_array.astype(np.float32)
    # print(grayscale_array.dtype)

    height, width = grayscale_array.shape[:2]

    k = 2*np.pi/(lam)
    x = np.linspace(-pix*width/2, pix*width/2, width)
    y = np.linspace(-pix*height/2, pix*height/2, height)
    x, y = np.meshgrid(x, y)
    # z = np.linspace(z1, z2, int((z2-z1)/z_interval)+1)

    # 只算聚焦像
    r = np.sqrt(x**2+y**2+z1**2)
    h = 1/(1j*lam*r)*np.exp(1j*k*r)
    H = fft2(fftshift(h))*pix**2
    U1 = fft2(fftshift(grayscale_array))
    U2 = U1*H
    U3 = ifftshift(ifft2(U2))

    return abs(U3)

        # plt.imsave('F:/Result/re_{:d}_{:}.jpg'.format(i+1, z[i]), U3, cmap="gray")
        # img_pth = out_path + '/' + 're_{:d}_{:}.jpg'.format(i+1, z[i])
        # plt.imsave(img_pth, abs(U3), cmap='gray')


def batch(input_pth, output_pth):
    file_names = os.listdir(input_pth)
    image_files = [f for f in file_names if any(ext in f.lower() for ext in ('.jpg', '.jpeg', '.png', '.bmp'))]
    for image_file in image_files:
        image_path = os.path.join(input_pth, image_file)

        new_path = os.path.join(output_pth, image_file)
        if not os.path.exists(new_path):
            os.makedirs(new_path)

        fresnel(image_path, new_path)


# if __name__ == '__main__':
#     # 波长
#     lam = 532e-9
#     # 像素大小
#     pix = 0.186e-6
#     # 重建距离
#     z1 = 0.0001
#     z2 = 0.006
#     z_interval = 0.0001
#
#     input_path = r'C:\Users\d1009\Desktop\temp'
#     output_path = r'C:\Users\d1009\Desktop\temp\result'
#     # bg_path = 'F:/Data/20240329/bg.bmp'
#     batch(input_path, output_path)

