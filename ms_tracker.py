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
        patch_w = region[2] if region[2]%2 == 1 else region[2] + 1
        patch_h = region[3] if region[3]%2 == 1 else region[3] + 1
        x = region[0]
        y = region[1]
        patch = get_patch(image, (x, y), (int(patch_w), int(patch_h)))[0]
        self.kernel = create_epanechnik_kernel(patch_w, patch_h, 2)
        hist_q = extract_histogram(patch, 16, self.kernel)
        self.hist_q = hist_q / np.sum(hist_q)
        self.patch_w = patch_w
        self.patch_h = patch_h
        self.x = int(x)
        self.y = int(y)


    def track(self, image):
        dx = 10
        dy = 10
        i=1
        while (dx != 0 or dy != 0) and (i < 20):
            patch2 = get_patch(image, (self.x, self.y), (self.patch_w, self.patch_h))[0]
            hist_p = extract_histogram(patch2, 16, self.kernel)
            hist_p_normalized = hist_p / np.sum(hist_p)
            v = np.sqrt(self.hist_q/(hist_p_normalized + 1e-3))
            back_p = backproject_histogram(image, v, 16)
            new_x, new_y = mean_shift(back_p, 11, x_start=self.x, y_start=self.y, N_iterations=1)
            dx = new_x - self.x
            dy = new_y - self.y
            self.x = self.x + dx
            self.y = self.y + dy
            i += 1
        return(self.x, self.y, self.patch_w, self.patch_h)
        

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

        x_new = int(round(np.sum(x_coords * -x_diffs * patch) / np.sum(-x_diffs * patch)))
        y_new = int(round(np.sum(y_coords * -y_diffs * patch) / np.sum(-y_diffs * patch)))
        
        if (x_new == x) and (y_new == y) or N_iterations <= i:
            return x_new - h, y_new - h
        
        i += 1
        past_coordinates.append([x,y])
            
        x = x_new
        y = y_new
    
            
    