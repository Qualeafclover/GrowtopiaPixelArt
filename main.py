import cv2
import numpy as np
import pygame
import copy
from sklearn.metrics import mean_squared_log_error as msle
from sklearn.metrics import mean_absolute_error as mae

pygame.init()
# pixels = (200, 108)
pixels = (100, 54)

# Normal blocks
# Dark blocks
# Pastel blocks
# Normal wallpapers
# Pastel wallpapers
# Dark wallpapers
B_multiplier = 1
G_multiplier = 1
R_multiplier = 1
saturation_multiplier = 1
value_multiplier = 1

advantage = {

}
inventory = (
    'Aqua Block.png',
    'Black Block.png',
    'Blue Block.png',
    'Brown Block.png',
    'Green Block.png',
    'Grey Block.png',
    'Orange Block.png',
    'Purple Block.png',
    'Red Block.png',
    'White Block.png',
    'Yellow Block.png',

    'Dark Aqua Block.png',
    'Dark Blue Block.png',
    'Dark Brown Block.png',
    'Dark Green Block.png',
    'Dark Grey Block.png',
    'Dark Orange Block.png',
    'Dark Purple Block.png',
    'Dark Red Block.png',
    'Dark Yellow Block.png',

    'Pastel Pink Block.png',
    'Pastel Orange Block.png',
    'Pastel Yellow Block.png',
    'Pastel Green Block.png',
    'Pastel Aqua Block.png',
    'Pastel Blue Block.png',
    'Pastel Purple Block.png',

    'Black Wallpaper.png',
    'Brown Wallpaper.png',
    'Grey Wallpaper.png',
    'Red Wallpaper.png',
    'Aqua Wallpaper.png',
    'Green Wallpaper.png',
    'White Wallpaper.png',
    'Yellow Wallpaper.png',
    'Orange Wallpaper.png',
    'Blue Wallpaper.png',
    'Purple Wallpaper.png',

    'Pastel Pink Wallpaper.png',
    'Pastel Orange Wallpaper.png',
    'Pastel Yellow Wallpaper.png',
    'Pastel Green Wallpaper.png',
    'Pastel Aqua Wallpaper.png',
    'Pastel Blue Wallpaper.png',
    'Pastel Purple Wallpaper.png',

    'Dark Red Wallpaper.png',
    'Dark Orange Wallpaper.png',
    'Dark Yellow Wallpaper.png',
    'Dark Green Wallpaper.png',
    'Dark Aqua Wallpaper.png',
    'Dark Blue Wallpaper.png',
    'Dark Purple Wallpaper.png',
    'Dark Grey Wallpaper.png',
    'Dark Brown Wallpaper.png',

    # MISC
    'Red Glass Block.png',
    'Blue Glass Block.png',
    'Air Duct.png',
    'The Darkness.png',

    'Art Wall.png',
    'Hospital Wall.png',
    'Ice.png',
    'Bubble Wrap.png',
    'Basic Blue Block.png',
            )
pg_inv = {png: pygame.image.load(png) for png in inventory}

# img_name = input('Please input the image name: ')
# img_name = 'koishi.png'
img_name = 'suga.png'
# img_name = 'umbrellaThing.jpg'

original_img = cv2.imread(img_name)
original_size = (lambda img: tuple(img.shape[1::-1]))(original_img)

new_img = cv2.resize(original_img, pixels, interpolation=0)

enlarged_img = cv2.resize(new_img, original_size, interpolation=0)

palette = []
for block in inventory:
    new_block = cv2.resize(cv2.imread(block), (1, 1))
    palette.append(new_block[0][0])

new_img[..., 0] = new_img[..., 0]*B_multiplier
new_img[..., 1] = new_img[..., 1]*G_multiplier
new_img[..., 2] = new_img[..., 2]*R_multiplier
new_img = cv2.cvtColor(new_img, cv2.COLOR_BGR2HSV)
new_img[..., 1] = new_img[..., 1]*saturation_multiplier
new_img[..., 2] = new_img[..., 2]*value_multiplier
new_img = cv2.cvtColor(new_img, cv2.COLOR_HSV2BGR)

img_list = np.array(new_img.flatten().astype(float)).tolist()
new_group = []
for item in range(0, len(img_list[:]), 3):
    new_group.append(img_list[item:item+3])

standardized = []
standardized_names = []
counter = 0
for item in new_group:
    score = []
    counter2 = 0
    for block in palette:
        try:
            multi = 1/advantage[inventory[counter2]]
        except KeyError:
            multi = 1
        counter2 += 1
        temp_score = mae([int(col) for col in item], block.tolist())
        temp_score += msle([int(col) for col in item], block.tolist())
        temp_score *= multi
        score.append(temp_score)
    max_pos = [i for i, j in enumerate(score) if j == min(score)][0]
    standardized.append(palette[max_pos])
    standardized_names.append(inventory[max_pos])
    counter += 1
    print(counter, 'upon', pixels[0]*pixels[1], 'complete...')

block_list = copy.copy(standardized_names)

standardized = np.array(standardized)
standardized_names = np.array(standardized_names)
standardized = standardized.reshape((pixels[1], pixels[0], 3))
standardized_names = standardized_names.reshape((pixels[1], pixels[0]))
standardized = cv2.resize(standardized, original_size, interpolation=0)

cv2.imwrite('pix_'+img_name, standardized)

if input('y to convert to blocks: ') == 'y':
    full_size = (pixels[0]*64, pixels[1]*64)
    window = pygame.display.set_mode(full_size)
    window.fill((0, 0, 0))
    for column in range(len(standardized_names)):
        for item in range(len(standardized_names[column])):
            window.blit(pg_inv[standardized_names[column][item]], (item*64, column*64))
    pygame.image.save(window, 'gt_'+img_name)
    big_size = cv2.imread('gt_'+img_name)
    small_size = cv2.resize(big_size, (pixels[0]*32, pixels[1]*32), interpolation=0)
    cv2.imwrite('gt_s_'+img_name, small_size)
    print('Saved all!')
else:
    while True:
        cv2.imshow(img_name, enlarged_img)
        cv2.imshow('new', standardized)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
unique_block = set(block_list)
needed_dict = {}
for item in unique_block:
    needed_dict[item[:-4]] = block_list.count(item)
print_dict = {k: v for k, v in sorted(needed_dict.items(), key=lambda y: y[1])}
for item in print_dict:
    print(item + ' '*(35-len(item[:-4])), print_dict[item])
quit()
