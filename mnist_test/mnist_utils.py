from torchvision.datasets import MNIST 
from PIL import Image 

import random 

random.seed(10) 


class MNISTDatasetHandler(): 
    
    mtrain = MNIST('', train=True, download=True) 
    mtest = MNIST('', train=False, download=True) 

    idxss = [] 
    for label in range(10): 
        idxss.append(( (mtrain.targets==label).nonzero() , (mtest.targets==label).nonzero() )) 
    
    def __init__(self, digit:int): 
        self.digit = digit # only return values of these digits 
        self.first_len = len(MNISTDatasetHandler.idxss[self.digit][0]) 
        self.length = len(MNISTDatasetHandler.idxss[self.digit][0]) + len(MNISTDatasetHandler.idxss[self.digit][1]) 
    
    def __len__(self): 
        return self.length 

    def __getitem__(self, idx): # returns a pillow image 
        if idx >= self.first_len: 
            idx -= self.first_len 
            #print("IDX:",MNISTDatasetHandler.idxss[self.digit][1][idx].item())
            return MNISTDatasetHandler.mtest[MNISTDatasetHandler.idxss[self.digit][1][idx].item()][0]
        #print("IDX:",MNISTDatasetHandler.idxss[self.digit][0][idx].item()) 
        return MNISTDatasetHandler.mtrain[MNISTDatasetHandler.idxss[self.digit][0][idx].item()][0] 


class MultipleMNISTGenerator(): 

    default_final_size = (260, 260) 
    default_size_range = ((28,28), (56,56)) # inclusive 

    min_gap = 4 

    digit_generators = [MNISTDatasetHandler(digit) for digit in range(10)] 

    def __init__(self, final_size=default_final_size, size_range=default_size_range): 
        self.final_size = final_size 
        self.size_range = size_range 
        self.maxw = (final_size[0])//(2*(size_range[1][0]+2*MultipleMNISTGenerator.min_gap)-1) 
        self.maxh = (final_size[1])//(2*(size_range[1][1]+2*MultipleMNISTGenerator.min_gap)-1) 
        self.max_digits = self.maxw * self.maxh 

    def generate(self, target_digit:int, other_digits:list[int]=[], borders=False): 
        # first digit is target digit, others are others. Borders is for visualization 

        # returns (out_img, bboxes) 
        # out_img is a pillow image 
        # bboxes is ltrb of every digit added in sequence, first is the target digit. 
        assert len(other_digits)<self.max_digits, "Cannot have more than "+str(self.max_digits)+" digits in total, due to size of image." 
        imgs = [random.choice(MultipleMNISTGenerator.digit_generators[target_digit])] 
        for digit in other_digits: 
            imgs.append(random.choice(MultipleMNISTGenerator.digit_generators[digit])) # pillow image 
        
        bboxes = [] 
        possible_poss = [] 
        for l in range(self.final_size[0]): 
            for t in range(self.final_size[1]): 
                possible_poss.append((l,t)) 

        out_img = Image.new('L', self.final_size, 0) 
        
        for img in imgs: 
            sizex = random.randint(self.size_range[0][0], self.size_range[1][0]) 
            sizey = random.randint(self.size_range[0][1], self.size_range[1][1]) 
            img = img.resize((sizex, sizey)) 

            l, t = random.randint(0, self.final_size[0]-sizex), random.randint(0, self.final_size[1]-sizey)
            while any( (l <= r2 + MultipleMNISTGenerator.min_gap and l+sizex >= l2 - MultipleMNISTGenerator.min_gap) and 
                      (t <= b2 + MultipleMNISTGenerator.min_gap and t+sizey >= t2 - MultipleMNISTGenerator.min_gap) for l2, t2, r2, b2 in bboxes): # ltrb format 
                l, t = random.randint(0, self.final_size[0]-sizex), random.randint(0, self.final_size[1]-sizey)

            if borders: 
                pixels = img.load() 
                for i in range(sizex): 
                    pixels[i,0] = 255 
                    pixels[i,sizey-1] = 255 
                for i in range(sizey): 
                    pixels[0,i] = 255 
                    pixels[sizex-1,i] = 255 

            bbox = (l, t, l+sizex, t+sizey) # ltrb format 
            out_img.paste(img, bbox)
            bboxes.append(bbox) # ltrb format 

        return out_img, bboxes 

    def get_empty_image(self): 
        return Image.new('L', self.final_size, 0) 


if __name__=='__main__': 
    #m1 = MNISTDatasetHandler(1) 
    #img = random.choice(m1)
    #img.show() 
    #print(img.size)
    random.seed(None) 
    mmg = MultipleMNISTGenerator() 
    mmg.generate(1, [3,4], borders=True)[0].show() 
    





