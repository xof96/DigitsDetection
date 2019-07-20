import os
import cv2

# %%
abs_path = os.path.abspath('.')
digits_path = 'data/digits/'
digits = os.listdir(os.path.join(abs_path, digits_path))

for i, path in enumerate(digits):
    digits[i] = os.path.join(abs_path, digits_path, path)


# %%
def get_num_and_ext(n_img):
    l = len(n_img) - 1
    while l >= 0:
        if n_img[l] == '.':
            l -= 1
            break
        l -= 1
    return n_img[l:len(n_img)]


# %%

def make_gray_scaled_pics(digits_path, abs_path, h=32, w=32):
    for d_path in digits_path:
        os.chdir(d_path)
        nums = os.listdir('.')
        for i, n_img in enumerate(nums):
            im = cv2.imread(n_img)
            gray_image = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            im = cv2.resize(gray_image, (h, w))
            cv2.imwrite('{}_{}'.format(i, get_num_and_ext(n_img)), im)
    os.chdir(abs_path)
    return


# %%
make_gray_scaled_pics(digits, abs_path)
