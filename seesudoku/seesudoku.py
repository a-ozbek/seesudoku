import numpy as np
from skimage import io, color, transform
from keras.models import model_from_json
from skimage import transform, util
from skimage.filters import threshold_otsu
import os
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
import solvesudoku


CURRENT_PATH = os.path.abspath(os.path.dirname(__file__))
MODEL_ARCH_RELATIVE_PATH = './model/model_arch.txt'
MODEL_WEIGHTS_RELATIVE_PATH = './model/model_weights.h5'
FONT_RELATIVE_PATH = './fontfiles/open-sans/OpenSans-Regular.ttf'

# * Load the model
# Load the architecture
json_file = file(os.path.join(CURRENT_PATH, MODEL_ARCH_RELATIVE_PATH), 'r')
json_str = json_file.read()
model_l = model_from_json(json_str)
# Load the weights
model_l.load_weights(os.path.join(CURRENT_PATH, MODEL_WEIGHTS_RELATIVE_PATH))
print 'Model is successfully loaded.'

# --- Interface ---
def get_array(im_filename, return_im=False):
    """
    Returns the sudoku array in a 9x9 numpy array
    -1 for the empty spaces
    """
    im = io.imread(im_filename)
    digits = [_get_digit(im_crop, model_l) for im_crop in _get_crops(im)]
    sudoku_digits = np.array(digits).reshape(9, 9)
    
    if return_im:
        return sudoku_digits, im
    else:
        return sudoku_digits


def get_solution_array(im_filename):
    """
    Returns the solution 9x9 numpy array
    """
    array = get_array(im_filename)
    # Solve
    array_sol = solvesudoku.solve_sudoku(array)
    
    return array_sol


def get_solution_im(im_filename):
    """
    Writes solution image to file
    """    

    # * Get array from the image
    array, im = get_array(im_filename, return_im=True)
    
    # * Solve the array
    array_sol = solvesudoku.solve_sudoku(array)
    
    # * Get the centers
    # Read and Convert to grayscale
    im = color.rgb2gray(im)
    # Reshape im (Rescale the input image, make it perfect square)
    target_size = int(np.mean(im.shape))
    im = transform.resize(im, (target_size,)*2)    
    # Size of the square
    sq_size = target_size / 9.0
    # Get the centers
    centers = np.linspace(0,target_size,9,endpoint=False) + sq_size / 2.0

    # * Insert text (solutions)
    im1 = Image.open(im_filename)  # Load the image    
    draw = ImageDraw.Draw(im1)  # Draw
    font_ = ImageFont.truetype(os.path.join(CURRENT_PATH, FONT_RELATIVE_PATH), size=int(round(sq_size * 0.9)))  # Font
    #
    for i in range(9):
        for j in range(9):
            if array[i, j] == -1:
                digit = str(array_sol[i, j])
                digit_x, digit_y = int(round(centers[j]-sq_size/2.9)), int(round(centers[i]-sq_size/1.7))
                draw.text((digit_x, digit_y), digit, (255,0,0), font=font_)
    # Save the image
    file_extension = os.path.basename(im_filename).split('.')[-1]
    im1.save(os.path.join('.', os.path.basename(im_filename).split('.')[-2] + '_sol.' + file_extension))

    return True


# --- Private Functions ---

def _is_blank(im):
    """
    Returns whether the there is a digit or a blank image.
    """
    
    # Take the r% center
    r = 0.2
    h1 = int(float(im.shape[0]) * r)
    h2 = im.shape[0] - h1
    w1 = int(float(im.shape[1]) * r)   
    w2 = im.shape[1] - w1
    #
    im_center = im[h1:h2, w1:w2]
    
    if np.mean(im_center) < 0.06:
        return True
    else:
        return False


def _fix_crop(im):
    """
    Centers the digit image
        
    Args:
        im: Cropped small digit image (It is assumed to have black bavk)
    """
    # Threshold the image and make it binary
    try:
        t = threshold_otsu(im)
    except ValueError:
        return im
    
    im_bw = im > t
    
    # Get the bounding box of the digit 
    true_i = np.where(im_bw == True)
    bbox = (np.min(true_i[0])-1, np.max(true_i[0])+1, np.min(true_i[1])-1, np.max(true_i[1])+1)
    
    # Fix
    im_crop = im[bbox[0]:bbox[1], bbox[2]:bbox[3]]
    im_fixed = np.zeros(im.shape)
    # h
    h1 = (im_fixed.shape[0]/2) - (im_crop.shape[0]/2)
    h2 = h1 + (im_crop.shape[0]) 
    # w
    w1 = (im_fixed.shape[1]/2) - (im_crop.shape[1]/2)
    w2 = w1 + (im_crop.shape[1])
    im_fixed[h1:h2,w1:w2] = im_crop
    
    return im_fixed


def _get_crops(im):
    """
    Returns cropped images as list
    im: Whole sudoku image (It is assumed to have white background and black digits)
    """
    # Convert to grayscale
    im = color.rgb2gray(im)

    # Reshape im (Rescale the input image, make it perfect square)
    target_size = int(np.mean(im.shape))
    im = transform.resize(im, (target_size,)*2)
    
    # Size of the square
    sq_size = target_size / 9.0
    # Get the centers
    centers = np.linspace(0,target_size,9,endpoint=False) + sq_size / 2.0
    
    # Crop
    r = 0.71
    crop_size = int(round((sq_size * r) * 0.5))

    cropped_ims = []
    for i in range(9):
        for j in range(9):
            v1, v2 = centers[i] - crop_size, centers[i] + crop_size
            h1, h2 = centers[j] - crop_size, centers[j] + crop_size
            v1, v2, h1, h2 = int(round(v1)), int(round(v2)), int(round(h1)), int(round(h2))        
            cropped = im[v1:v2, h1:h2]
            
            # Process cropped_im
            cropped = 1.0 - cropped  # At this point, digit is white, background is black
            
            # Fix the cropped image (Place the digit to the center)
            cropped = _fix_crop(cropped)
            
            # Append
            cropped_ims.append(cropped)
  
    return cropped_ims


def _get_digit(im, model, input_shape=(128,128)):
    """
    im: Input Digit Image (Sudoku image is assumed to be black digits on a white background)
    model: model to predict the digit
    """
    
    # First check if it is blank
    if _is_blank(im):
        return -1
    
    # Predict the digit using trained model
    im = transform.resize(im, input_shape)
    im = np.expand_dims(im, axis=0)
    im = np.expand_dims(im, axis=3)
    im = 1.0 - im  # Invert the grayscale so that ...
    return int(model.predict_classes(im, verbose=0)[0]) + 1  # +1 because it predicts 1 less due to order of classes


