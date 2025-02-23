'''
    Dataset for shapenet part segmentaion.
'''
# -*- coding: utf-8 -*-
import os
import os.path
import json
import numpy as np
import sys

def pc_normalize(pc):
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

class PartDataset():
    def __init__(self, root, npoints = 2500, classification = False, class_choice = None, split='train', normalize=True):
        self.npoints = npoints
        self.root = root
        self.catfile = os.path.join(self.root, 'synsetoffset2category.txt')
        self.cat = {}
        print(self.catfile)
        self.classification = classification
        self.normalize = normalize
        
        with open(self.catfile, 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = ls[1]
        #print(self.cat)
        if not class_choice is  None:
            self.cat = {k:v for k,v in self.cat.items() if k in class_choice}
            
        self.meta = {}
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_train_file_list.json'), 'r') as f:
            train_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_val_file_list.json'), 'r') as f:
            val_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_test_file_list.json'), 'r') as f:
            test_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        for item in self.cat:
            #print('category', item)
            self.meta[item] = []
            #dir_point = os.path.join(self.root, self.cat[item], 'points')
            #dir_seg = os.path.join(self.root, self.cat[item], 'points_label')
            dir_point = os.path.join(self.root, self.cat[item])
            dir_seg = os.path.join(self.root, self.cat[item])
            #print(dir_point, dir_seg)
            fns = sorted(os.listdir(dir_point))
            #print(fns[0][0:-4])
            if split=='trainval':
                fns = [fn for fn in fns if ((fn[0:-4] in train_ids) or (fn[0:-4] in val_ids))]
            elif split=='train':
                fns = [fn for fn in fns if fn[0:-4] in train_ids]
            elif split=='val':
                fns = [fn for fn in fns if fn[0:-4] in val_ids]
            elif split=='test':
                fns = [fn for fn in fns if fn[0:-4] in test_ids]
            else:
                print('Unknown split: %s. Exiting..'%(split))
                exit(-1)
                
            #print(os.path.basename(fns))
            for fn in fns:
                token = (os.path.splitext(os.path.basename(fn))[0]) 
                self.meta[item].append((os.path.join(dir_point, token + '.pts'), os.path.join(dir_seg, token + '.seg')))
        
        self.datapath = []
        for item in self.cat:
            for fn in self.meta[item]:
                self.datapath.append((item, fn[0], fn[1]))
            
         
        self.classes = dict(zip(self.cat, range(len(self.cat))))  
        self.num_seg_classes = 0
        if not self.classification:
            for i in range(len(self.datapath)/50):
                l = len(np.unique(np.loadtxt(self.datapath[i][-1]).astype(np.uint8)))
                if l > self.num_seg_classes:
                    self.num_seg_classes = l
        #print(self.num_seg_classes)
        
        self.cache = {} # from index to (point_set, cls, seg) tuple
        self.cache_size = 10000
               
    def __getitem__(self, index):
        if index in self.cache:
            point_set, seg, cls = self.cache[index]
        else:
            fn = self.datapath[index]
            cls = self.classes[self.datapath[index][0]]
            cls = np.array([cls]).astype(np.int32)
            point_set = np.loadtxt(fn[1]).astype(np.float32)
            if self.normalize:
                point_set = pc_normalize(point_set)
            seg = np.loadtxt(fn[2]).astype(np.int64) - 1
            #print(point_set.shape, seg.shape)
            if len(self.cache) < self.cache_size:
                self.cache[index] = (point_set, seg, cls)
                
        
        choice = np.random.choice(len(seg), self.npoints, replace=True)
        #resample
        point_set = point_set[choice, :]
        seg = seg[choice]
        if self.classification:
            return point_set, cls
        else:
            return point_set, seg
        
    def __len__(self):
        return len(self.datapath)


if __name__ == '__main__':
    d = PartDataset(root = '../data/shapenetcore_partanno_segmentation_benchmark_v0', class_choice = ['Airplane'], split='test')
    print(len(d))
    import time
    tic = time.time()
    for i in range(100):
        ps, seg = d[i]
        print (np.max(seg), np.min(seg))
    print(time.time() - tic)
    print(ps.shape, type(ps), seg.shape,type(seg))
    
    d = PartDataset(root = '../data/shapenetcore_partanno_segmentation_benchmark_v0', classification = True)
    print(len(d))
    ps, cls = d[0]
    print(ps.shape, type(ps), cls.shape,type(cls))

