from PIL import Image
import numpy as np
from colorama import init, Fore, Back, Style
import os

def grayscale_ansi(ints, bg=False):
    # Convert grayscale into 3 ansi values
    if   ints < 65:  return Back.BLACK if bg else Fore.BLACK
    elif ints < 170: return Back.LIGHTBLACK_EX if bg else Fore.LIGHTBLACK_EX
    else:            return Back.WHITE if bg else Fore.WHITE



init()
image = Image.open('title.png').convert('L')


width  = os.get_terminal_size().columns - 6 # 3 margin
height = os.get_terminal_size().lines
_width = width +6


# ratio w 0.5 zoom
aspect_ratio = image.width / image.height
width  = int(width * 2 * 0.5)
height = int(int(width / aspect_ratio))

# resize to fit terminal
image = image.resize((width, height), Image.Resampling.LANCZOS)
pixels = np.array(image)

start_x = 3
start_y = (os.get_terminal_size().lines - height//2) // 2 #vert center


output = []
output.append(f"{Fore.WHITE}{Back.LIGHTBLACK_EX} " * _width)
for y in range(0, pixels.shape[0], 2):
    line = f"{Back.LIGHTBLACK_EX} " * start_x
    for x in range(pixels.shape[1]):
        # get top and bottom pixels
        top_pixel = pixels[y, x]
        if y+1 < pixels.shape[0]: 
              bottom_pixel = pixels[y + 1, x]
        else: bottom_pixel = top_pixel


        bg_color = grayscale_ansi(top_pixel,  bg=True)
        fg_color = grayscale_ansi(bottom_pixel, bg=False)
        line += f"{bg_color}{fg_color}▄"
    
    output.append(line  + (f"{Back.LIGHTBLACK_EX} " * start_x) + Style.RESET_ALL)


subtitle = "T h e   T e r m i n a l   V e r s i o n !"
height_s = (_width - len(subtitle))//2 + 1
output.append(f"{Back.LIGHTBLACK_EX} " * height_s  + Style.RESET_ALL
 + subtitle + f"{Back.LIGHTBLACK_EX} " * height_s  + Style.RESET_ALL 
 )



print("\n".join(output))

