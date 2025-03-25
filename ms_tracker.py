from ex2_utils import *


class MeanShiftTracker():
    def __init__(self, params):
        self.parameters = params
        self.hist_q = None
        self.kernel = None
        self.patch_w = None
        self.patch_h = None
        self.x = None
        self.y = None

    def initialize(self, image, region):
        if len(region) == 8:
            x_ = np.array(region[::2])
            y_ = np.array(region[1::2])
            region = [np.min(x_), np.min(y_), np.max(x_) - np.min(x_) + 1, np.max(y_) - np.min(y_) + 1]

        x = max(region[0], 0)
        y = max(region[1], 0)

        self.position = (region[0] + region[2] / 2, region[1] + region[3] / 2)

        right = min(region[0] + region[2], image.shape[1] - 1)
        bottom = min(region[1] + region[3], image.shape[0] - 1)
        patch_w = int(right - x)
        patch_h = int(bottom - y)
        patch_w = patch_w if patch_w %2 == 1 else patch_w + 1
        patch_h = patch_h if patch_h %2 == 1 else patch_h + 1
        patch = get_patch(image, (self.position), (int(patch_w), int(patch_h)))[0]
        self.kernel = create_epanechnik_kernel(int(patch_w), int(patch_h), 0.8)
        hist_q = extract_histogram(patch, 16, self.kernel)
        self.hist_q = hist_q / np.sum(hist_q)
        self.patch_w = int(patch_w)
        self.patch_h = int(patch_h)
        self.x = self.position[0]
        self.y = self.position[1]


    def track(self, image):
        i=0
        while (i < 20):
            patch2, mask_kernel = get_patch(image, (round(self.x), round(self.y)), (self.patch_w, self.patch_h))
            cur_ker = self.kernel * mask_kernel
            hist_p = extract_histogram(patch2, 16, self.kernel)
            hist_p_normalized = hist_p / np.sum(hist_p)
            v = np.sqrt(self.hist_q/(hist_p_normalized + 1e-5))
            back_p = backproject_histogram(patch2, v, 16)
            x_range = np.arange(-(self.patch_w // 2), (self.patch_w // 2) + 1)
            y_range = np.arange(-(self.patch_h // 2), (self.patch_h // 2) + 1)
            
            x_diffs, y_diffs = np.meshgrid(x_range, y_range)

            dx = np.sum(x_diffs * back_p) / (np.sum(back_p) + 1e-7)
            dy = np.sum(y_diffs * back_p) / (np.sum(back_p) + 1e-7)

            if (np.sqrt(dx**2+dy**2) <= 1):
                return(round(self.x - self.patch_w /2), round(self.y - self.patch_h/2), self.patch_w, self.patch_h)

            self.x = ((self.x + dx))
            self.y = ((self.y + dy))
            i += 1

        return(round(self.x - self.patch_w /2), round(self.y - self.patch_h/2), self.patch_w, self.patch_h)

def mean_shift3(fun, kernel, x_start=None, y_start=None, N_iterations = 1):
    height, width = kernel.shape
    
    i = 1
    x = x_start
    y = y_start
    x_new = None
    y_new = None
    while True:

        x_range = np.arange(-(width // 2), width // 2 + 1)  # Width range
        y_range = np.arange(-(height // 2), height // 2 + 1)  # Height range

        x_diffs, y_diffs = np.meshgrid(x_range, y_range)  # Create 2D grids

        x_diffs = (x_diffs / (width)) ** 2 # Normalize x distances
        y_diffs = (y_diffs / (height)) ** 2 # Normalize y distances

        #x_coords, y_coords = np.meshgrid(np.arange(width), np.arange(height))

        x_new = x + int(np.sum(x_diffs * fun) / np.sum(fun))
        y_new = y + int(np.sum(y_diffs * fun) / np.sum(fun))
        
        if ((x_new == x) and (y_new == y)) or N_iterations <= i:
            return x_new, y_new
        
        i += 1
            
        x = x_new
        y = y_new
    

def mean_shift(fun, h, x_start=None, y_start=None, N_iterations = 100):
    width, height = fun.shape
    
    img = np.pad(fun, h, "constant", constant_values=(0))
    patch = np.zeros((1))
    h_side = int((h-1)/2)

    kernel = gausssmooth

    past_coordinates = []
    
    if x_start == None or y_start == None:
        while np.sum(patch) == 0:
            x = np.random.randint(0, width) + h
            y = np.random.randint(0, height) + h

            x_min = x - h_side
            x_max = x + h_side
            y_min = y - h_side
            y_max = y + h_side

            patch = img[y_min:y_max+1, x_min:x_max+1]
    
    else:
        x = x_start + h
        y = y_start + h

    i = 1

    x_new = None
    y_new = None
    while True:
    
        x_min = x - h_side
        x_max = x + h_side
        y_min = y - h_side
        y_max = y + h_side
        patch = img[y_min:y_max+1, x_min:x_max+1]
        rows = []
        for _ in range(h):
            rows.append([x for x in range(-h_side, h_side+1)])

        x_diffs = np.array(rows)
        y_diffs = x_diffs.T

        x_diffs = (x_diffs/h)**2
        y_diffs = (y_diffs/h)**2

        #x_diffs = gausssmooth(x_diffs, 0.75)
        #y_diffs = gausssmooth(y_diffs, 0.75)

        #need kernel

        x_coords = np.tile(np.arange(x_min, x_max+1), (h, 1))
        y_coords = np.tile(np.arange(y_min, y_max+1), (h, 1)).T

        x_new = int(round(np.sum(x_coords  * patch) / np.sum(patch)))
        y_new = int(round(np.sum(y_coords  * patch) / np.sum(patch)))
        
        if (x_new == x) and (y_new == y) or N_iterations <= i:
            return x_new - h, y_new - h
        
        i += 1
        past_coordinates.append([x,y])
            
        x = x_new
        y = y_new
    
            
    