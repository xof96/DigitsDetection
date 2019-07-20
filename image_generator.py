import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from random import shuffle, randint

# %%
ABS_PATH = os.path.abspath('.')
DATA_PATH = 'data/digits'
digits = os.listdir(os.path.join(ABS_PATH, DATA_PATH))
DIGITS_PATHS = []
for d in digits:
    images = os.listdir(os.path.join(ABS_PATH, DATA_PATH, d))
    for i in images:
        DIGITS_PATHS.append(os.path.join(ABS_PATH, DATA_PATH, d, i))


# %%
def get_label(first_label_path):
    return int(first_label_path[-5])


def create_image(n_digits):
    index = randint(0, len(DIGITS_PATHS) - 1)
    first_label_path = DIGITS_PATHS[index]
    first_label = get_label(first_label_path)
    first_digit = cv2.imread(first_label_path, cv2.IMREAD_GRAYSCALE)
    label = [first_label]
    for i in range(n_digits - 1):
        index = randint(0, len(DIGITS_PATHS) - 1)
        next_label_path = DIGITS_PATHS[index]
        next_label = get_label(next_label_path)
        curr_number_img = cv2.imread(next_label_path, cv2.IMREAD_GRAYSCALE)
        first_digit = np.append(first_digit, curr_number_img, axis=1)
        label.append(next_label)
    return first_digit, label

# %%
def create_train_test_dataset(n_train, n_test, n_digits):
    d_training = {}
    for i in range(n_train):
        n, l = create_image(n_digits)
        str_l = str(l)
        while str_l in d_training:
            n, l = create_image(n_digits)
        d_training[str_l] = [n, l]

    d_testing = {}
    for i in range(n_test):
        n, l = create_image(n_digits)
        str_l = str(l)
        while str_l in d_training or str_l in d_testing:
            n, l = create_image(n_digits)
        d_testing[str_l] = [n, l]

    return list(d_training.values()), list(d_testing.values())


# %%

def save_dataset(train_path, test_path, n_train, n_test, n_digits):
    train, test = create_train_test_dataset(n_train, n_test, n_digits)
    with open(os.path.join(ABS_PATH, 'data/train.txt'), 'w') as train_file:
        for t, l in train:
            path = str(l[0])
            for c in l[1:]:
                path = '-'.join([path, str(c)])
            cv2.imwrite(os.path.join(train_path, '{}.png'.format(path)), t)
            label = ''.join(path.split('-'))
            train_file.write(os.path.join(train_path, '{}.png\t{}\n'.format(path, label)))

    with open(os.path.join(ABS_PATH, 'data/test.txt'), 'w') as test_file:
        for t, l in test:
            path = str(l[0])
            for c in l[1:]:
                path = '-'.join([path, str(c)])
            cv2.imwrite(os.path.join(test_path, '{}.png'.format(path)), t)
            label = ''.join(path.split('-'))
            test_file.write(os.path.join(test_path, '{}.png\t{}\n'.format(path, label)))


# %%

# save_dataset(os.path.join(ABS_PATH, 'data/train'), os.path.join(ABS_PATH, 'data/test'), 5000, 1000)
