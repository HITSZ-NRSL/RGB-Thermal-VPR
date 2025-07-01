import torch
import torch.utils.data as data
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset
import torchvision.transforms as transforms
from scipy.io import loadmat
import numpy as np
from sklearn.neighbors import NearestNeighbors
import cv2
import os
import faiss
from tqdm import tqdm
import logging

base_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def collate_fn(batch):
    """Creates mini-batch tensors from the list of tuples (images,
        triplets_local_indexes, triplets_global_indexes).
        triplets_local_indexes are the indexes referring to each triplet within images.
        triplets_global_indexes are the global indexes of each image.
    Args:
        batch: list of tuple (images, triplets_local_indexes, triplets_global_indexes).
            considering each query to have 10 negatives (negs_num_per_query=10):
            - images: torch tensor of shape (12, 3, h, w).
            - triplets_local_indexes: torch tensor of shape (10, 3).
            - triplets_global_indexes: torch tensor of shape (12).
    Returns:
        images: torch tensor of shape (batch_size*12, 3, h, w).
        triplets_local_indexes: torch tensor of shape (batch_size*10, 3).
        triplets_global_indexes: torch tensor of shape (batch_size, 12).
    """
    images                  = torch.cat([e[0] for e in batch])
    triplets_local_indexes  = torch.cat([e[1][None] for e in batch])
    triplets_global_indexes = torch.cat([e[2][None] for e in batch])
    for i, (local_indexes, global_indexes) in enumerate(zip(triplets_local_indexes, triplets_global_indexes)):
        local_indexes += len(global_indexes) * i  
    return images, torch.cat(tuple(triplets_local_indexes)), triplets_global_indexes


class BaseSTheReODual(data.Dataset):
    def __init__(self, args, dataset_folder, split='test'):
        super().__init__()
        self.dataset_folder = dataset_folder
        self.img_time = args.img_time
        self.matStruct = [loadmat(os.path.join(self.dataset_folder, seq, f'sthereo_{split}.mat'))['dbStruct'] for seq in
                          args.sequences]
        for seq in args.sequences:
            print("load dataset:", seq)
        self.seq_num = len(self.matStruct)
        self.resize = args.resize
        self.test_method = args.test_method
        # 这里是拿到所有databse的位置
        self.database_utms = np.concatenate(
            [mat['db_pose'][0, 0] for mat in self.matStruct]
        )
        if self.img_time == 'allday':
            self.queries_utms = np.concatenate([
                np.concatenate((
                    mat['q_pose_morning'][0, 0],
                    mat['q_pose_afternoon'][0, 0],
                    mat['q_pose_evening'][0, 0]
                )) for mat in self.matStruct
            ])
        elif self.img_time == 'daytime':
            self.queries_utms = np.concatenate([
                np.concatenate((
                    mat['q_pose_morning'][0, 0],
                    mat['q_pose_afternoon'][0, 0]
                )) for mat in self.matStruct
            ])

        elif self.img_time == 'nighttime':
            self.queries_utms = np.concatenate([
                mat['q_pose_evening'][0, 0] for mat in self.matStruct
            ])

        knn = NearestNeighbors(n_jobs=-1)
        knn.fit(self.database_utms)
        self.soft_positives_per_query = knn.radius_neighbors(
            self.queries_utms, radius=args.soft_positives_dist_threshold, return_distance=False
        )

        self.rgb_database_paths = np.concatenate(
            [mat['db_rgb'][0, 0] for mat in self.matStruct]
        )
        if self.img_time == 'allday':
            self.rgb_queries_paths = np.concatenate([
                np.concatenate((
                    mat['q_rgb_morning'][0, 0],
                    mat['q_rgb_afternoon'][0, 0],
                    mat['q_rgb_evening'][0, 0]
                )) for mat in self.matStruct
            ])
        elif self.img_time == 'daytime':
            self.rgb_queries_paths = np.concatenate([
                np.concatenate((
                    mat['q_rgb_morning'][0, 0],
                    mat['q_rgb_afternoon'][0, 0]
                )) for mat in self.matStruct
            ])

        elif self.img_time == 'nighttime':
            self.rgb_queries_paths = np.concatenate([
                mat['q_rgb_evening'][0, 0] for mat in self.matStruct
            ])

        self.t_database_paths = np.concatenate(
            [mat['db_t'][0, 0] for mat in self.matStruct]
        )
        if self.img_time == 'allday':
            self.t_queries_paths = np.concatenate([
                np.concatenate((
                    mat['q_t_morning'][0, 0],
                    mat['q_t_afternoon'][0, 0],
                    mat['q_t_evening'][0, 0]
                )) for mat in self.matStruct
            ])
        elif self.img_time == 'daytime':
            self.t_queries_paths = np.concatenate([
                np.concatenate((
                    mat['q_t_morning'][0, 0],
                    mat['q_t_afternoon'][0, 0]
                )) for mat in self.matStruct
            ])

        elif self.img_time == 'nighttime':
            self.t_queries_paths = np.concatenate([
                mat['q_t_evening'][0, 0] for mat in self.matStruct
            ])


        assert (self.t_database_paths.shape) == (self.rgb_database_paths.shape) and (self.t_queries_paths.shape) == (self.rgb_queries_paths.shape)

        self.rgb_img_paths = list(self.rgb_database_paths) + list(self.rgb_queries_paths)
        self.t_img_paths = list(self.t_database_paths) + list(self.t_queries_paths)
        self.database_num = len(self.rgb_database_paths)
        self.queries_num = len(self.rgb_queries_paths)

    def get_rgb_img(self, path):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = cv2.cvtColor(img, cv2.COLOR_BAYER_BG2RGB)
        return img

    def get_thermal_img(self, path):
        img = cv2.imread(path, cv2.IMREAD_ANYDEPTH)
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        return img


    def __getitem__(self, index):
        rgb_img = self.get_rgb_img(self.rgb_img_paths[index])
        thermal_img = self.get_thermal_img(self.t_img_paths[index])
        rgb_img = base_transform(rgb_img)
        thermal_img = base_transform(thermal_img)

        if self.test_method == "hard_resize" or self.test_method == "single_query":
            rgb_img = transforms.functional.resize(rgb_img, self.resize)
            thermal_img = transforms.functional.resize(thermal_img, self.resize)
        else:
            rgb_img = self.__test_query_transform(rgb_img)
            thermal_img = self.__test_query_transform(thermal_img)

        img = torch.cat((rgb_img, thermal_img), dim=0) 
        return img, index

    def __len__(self):
        return len(self.rgb_img_paths)

    def __repr__(self):
        return "STheReO"

    def get_positives(self):
        return self.soft_positives_per_query

    def __test_query_transform(self, img):
        ### Transform query image according to self.test_method
        C, H, W = img.shape
        if self.test_method == "central_crop":
            # NOTE: Scale before cropping
            scale = max(self.resize[0]/H, self.resize[1]/W)
            processed_img = torch.nn.functional.interpolate(img.unsqueeze(0), scale_factor=scale).squeeze(0)
            processed_img = transforms.functional.center_crop(processed_img, self.resize)
            assert processed_img.shape[1:] == torch.Size(self.resize), f"{processed_img.shape[1:]} {self.resize}"
        elif self.test_method == "five_crops" or self.test_method == "nearest_crop" or self.test_method == "maj_voting":
            shorter_side = min(self.resize)
            processed_img = transforms.functional.resize(img, shorter_side)
            processed_img = torch.stack(transforms.functional.five_crop(processed_img, shorter_side))
            assert processed_img.shape == torch.Size([5, 3, shorter_side, shorter_side]), \
                f"{processed_img.shape} {torch.Size([5, 3, shorter_side, shorter_side])}"


class TripletsSTheReODual(BaseSTheReODual):
    """Dataset used for training, it is used to compute the triplets
    with TripletsDataset.compute_triplets() with various mining methods.
    If is_inference == True, uses methods of the parent class BaseDataset,
    this is used for example when computing the cache, because we compute features
    of each image, not triplets.
    """

    def __init__(self, args, datasets_folder):
        super().__init__(args, datasets_folder, split='train')

        self.mining = args.mining
        self.neg_samples_num = args.neg_samples_num
        self.negs_num_per_query = args.negs_num_per_query
        if self.mining == "full":
            self.neg_cache = [np.empty((0,), dtype=np.int32) for _ in range(len(self.queries_num))]
        self.is_inference = False

        # data augmentation
        identity_transform = transforms.Lambda(lambda x: x)
        self.resized_transform = transforms.Compose([
            base_transform,
            transforms.Resize(self.resize) if self.resize is not None else identity_transform,
            # base_transform
        ])
        self.query_transform = transforms.Compose([
            self.resized_transform,
            transforms.ColorJitter(brightness=args.brightness) if args.brightness != None else identity_transform,
            transforms.ColorJitter(contrast=args.contrast) if args.contrast != None else identity_transform,
            transforms.ColorJitter(saturation=args.saturation) if args.saturation != None else identity_transform,
            transforms.ColorJitter(hue=args.hue) if args.hue != None else identity_transform,
            transforms.RandomPerspective(
                args.rand_perspective) if args.rand_perspective != None else identity_transform,
            transforms.RandomResizedCrop(size=self.resize, scale=(1 - args.random_resized_crop, 1)) \
                if args.random_resized_crop != None else identity_transform,
            transforms.RandomRotation(
                degrees=args.random_rotation) if args.random_rotation != None else identity_transform,
            # self.resized_transform,
        ])

        knn = NearestNeighbors(n_jobs=-1)
        knn.fit(self.database_utms)
        self.hard_positives_per_query = list(knn.radius_neighbors(self.queries_utms,
                                                                  radius=args.hard_positives_dist_threshold,
                                                                  return_distance=False))

        queries_without_any_hard_positive = \
        np.where(np.array([len(p) for p in self.hard_positives_per_query], dtype=object) == 0)[0]
        logging.info(f"There are {len(queries_without_any_hard_positive)} queries without any positives")

        # self.hard_positives_per_query = np.delete(self.hard_positives_per_query, queries_without_any_hard_positive)
        self.hard_positives_per_query = [
            positives for i, positives in enumerate(self.hard_positives_per_query)
            if i not in queries_without_any_hard_positive
        ]
        self.rgb_queries_paths = np.delete(self.rgb_queries_paths, queries_without_any_hard_positive)
        self.t_queries_paths = np.delete(self.t_queries_paths, queries_without_any_hard_positive)

        self.queries_num = len(self.rgb_queries_paths)


    def __getitem__(self, index):
        if self.is_inference:
            # At inference time return the single image. This is used for caching or computing NetVLAD's clusters
            return super().__getitem__(index)

        query_index, best_positive_index, neg_indexes = torch.split(self.triplets_global_indexes[index],
                                                                    (1, 1, self.negs_num_per_query))
        rgb_query = self.query_transform(self.get_rgb_img(self.rgb_queries_paths[query_index]))
        t_query = self.query_transform(self.get_thermal_img(self.t_queries_paths[query_index]))
        query = torch.cat((rgb_query, t_query), dim=0)

        rgb_positive = self.resized_transform(self.get_rgb_img(self.rgb_database_paths[best_positive_index]))
        t_positive = self.resized_transform(self.get_thermal_img(self.t_database_paths[best_positive_index]))
        positive = torch.cat((rgb_positive, t_positive), dim=0)

        rgb_negatives = [self.resized_transform(self.get_rgb_img(self.rgb_database_paths[i])) for i in neg_indexes]
        t_negatives = [self.resized_transform(self.get_thermal_img(self.t_database_paths[i])) for i in neg_indexes]
        negatives = [torch.cat((rgb, t), dim=0) for rgb, t in zip(rgb_negatives, t_negatives)]

        images = torch.stack((query, positive, *negatives), 0)

        # triplets_local_indexes = torch.empty((0,3), dtype=torch.int)
        # for neg_num in range(len(neg_indexes)):
        #     triplets_local_indexes = torch.cat((triplets_local_indexes, torch.tensor([[0, 1, neg_num+2]])))
        triplets_local_indexes = torch.tensor([([0, 1, neg_num + 2]) for neg_num in range(len(neg_indexes))])
        return images, triplets_local_indexes, self.triplets_global_indexes[index]


    def __len__(self):
        if self.is_inference:
            # At inference time return the number of images. This is used for caching or computing NetVLAD's clusters
            return super().__len__()
        else:
            return len(self.triplets_global_indexes)

    def compute_triplets(self, args, model):
        self.is_inference = True
        if self.mining == "partial":
            self.compute_triplets_partial(args, model)

    @staticmethod
    def compute_cache(args, model, subset_ds, cache_shape):
        """Compute the cache containing features of images, which is used to
        find best positive and hardest negatives."""

        subset_dl = DataLoader(dataset=subset_ds, num_workers=args.num_workers,
                               batch_size=args.infer_batch_size, shuffle=False,
                               pin_memory=(args.device == "cuda"))

        model = model.eval()
        # RAMEfficient2DMatrix can be replaced by np.zeros, but using
        # RAMEfficient2DMatrix is RAM efficient for full database mining.
        cache = RAMEfficient2DMatrix(cache_shape, dtype=np.float32)

        # W, H, C = args.dense_feature_map_size
        # cache_local_shape = [cache_shape[0], W, H, C]
        # cache_local = RAMEfficient4DMatrix(cache_local_shape, dtype=np.float32)
        with torch.no_grad():
            # logging.debug(f"Caching {len(subset_ds)} features")
            for images, indexes in tqdm(subset_dl, ncols=100):
                images = images.to(args.device)
                # local_features, global_features = model(images)
                global_features = model(images)
                cache[indexes.numpy()] = global_features.cpu().numpy()
                # cache_local[indexes.numpy] = local_features.cpu().numpy()
        return cache

    def get_query_features(self, query_index, cache):
        query_features = cache[query_index + self.database_num]
        if query_features is None:
            raise RuntimeError(f"For query {self.queries_paths[query_index]} " +
                               f"with index {query_index} features have not been computed!\n" +
                               "There might be some bug with caching")
        return query_features

    def get_best_positive_index(self, args, query_index, cache, query_features):
        positives_features = cache[self.hard_positives_per_query[query_index]]
        faiss_index = faiss.IndexFlatL2(args.features_dim)
        faiss_index.add(positives_features)
        # Search the best positive (within 10 meters AND nearest in features space)
        _, best_positive_num = faiss_index.search(query_features.reshape(1, -1), 1)
        best_positive_index = self.hard_positives_per_query[query_index][best_positive_num[0]].item()
        return best_positive_index

    def get_hardest_negatives_indexes(self, args, cache, query_features, neg_samples):
        neg_features = cache[neg_samples]
        faiss_index = faiss.IndexFlatL2(args.features_dim)
        faiss_index.add(neg_features)
        # Search the 10 nearest negatives (further than 25 meters and nearest in features space)
        _, neg_nums = faiss_index.search(query_features.reshape(1, -1), self.negs_num_per_query)
        neg_nums = neg_nums.reshape(-1)
        neg_indexes = neg_samples[neg_nums].astype(np.int32)
        return neg_indexes

    def compute_triplets_partial(self, args, model):
        '''
        cache: sampled queries and their positives, sampled database images
        negtives: eliminate soft positives from sampled database images, then get the hardest negatives
        '''
        self.triplets_global_indexes = []
        # Take 1000 random queries
        sampled_queries_indexes = np.random.choice(self.queries_num, args.cache_refresh_rate, replace=False)

        # Sample 1000 random database images for the negatives
        sampled_database_indexes = np.random.choice(self.database_num, self.neg_samples_num, replace=False)

        positives_indexes = [self.hard_positives_per_query[i] for i in sampled_queries_indexes]
        positives_indexes = [p for pos in positives_indexes for p in pos]
        database_indexes = list(sampled_database_indexes) + positives_indexes
        database_indexes = list(np.unique(database_indexes))

        subset_ds = Subset(self, database_indexes + list(sampled_queries_indexes + self.database_num))

        cache = self.compute_cache(args, model, subset_ds, cache_shape=(len(self), args.features_dim))

        for query_index in tqdm(sampled_queries_indexes, ncols=100):
            query_features = self.get_query_features(query_index, cache)
            best_positive_index = self.get_best_positive_index(args, query_index, cache, query_features)

            # Choose the hardest negatives within sampled_database_indexes, ensuring that there are no positives
            soft_positives = self.soft_positives_per_query[query_index]
            neg_indexes = np.setdiff1d(sampled_database_indexes, soft_positives, assume_unique=True)

            # Take all database images that are negatives and are within the sampled database images (aka database_indexes)
            neg_indexes = self.get_hardest_negatives_indexes(args, cache, query_features, neg_indexes)
            self.triplets_global_indexes.append((query_index, best_positive_index, *neg_indexes))
        self.triplets_global_indexes = torch.tensor(self.triplets_global_indexes)

class RAMEfficient2DMatrix:
    """This class behaves similarly to a numpy.ndarray initialized
    with np.zeros(), but is implemented to save RAM when the rows
    within the 2D array are sparse. In this case it's needed because
    we don't always compute features for each image, just for few of
    them"""

    def __init__(self, shape, dtype=np.float32):
        self.shape = shape
        self.dtype = dtype
        self.matrix = [None] * shape[0]

    def __setitem__(self, indexes, vals):
        assert vals.shape[1] == self.shape[1], f"{vals.shape[1]} {self.shape[1]}"
        for i, val in zip(indexes, vals):
            self.matrix[i] = val.astype(self.dtype, copy=False)

    def __getitem__(self, index):
        if hasattr(index, "__len__"):
            return np.array([self.matrix[i] for i in index])
        else:
            return self.matrix[index]

class RAMEfficient4DMatrix:
    """This class behaves similarly to a numpy.ndarray initialized
    with np.zeros(), but is implemented to save RAM when the rows
    within the 3D array are sparse. In this case it's needed because
    we don't always compute features for each image, just for few of
    them"""

    def __init__(self, shape, dtype=np.float32):
        self.shape = shape
        self.dtype = dtype
        self.matrix = [None] * shape[0]

    def __setitem__(self, indexes, vals):
        assert vals.shape[1] == self.shape[1], f"{vals.shape[1]} {self.shape[1]}"
        assert vals.shape[2] == self.shape[2], f"{vals.shape[2]} {self.shape[2]}"
        assert vals.shape[3] == self.shape[3], f"{vals.shape[3]} {self.shape[3]}"
        for i, val in zip(indexes, vals):
            self.matrix[i] = val.astype(self.dtype, copy=False)

    def __getitem__(self, index):
        if hasattr(index, "__len__"):
            return np.array([self.matrix[i] for i in index])
        else:
            return self.matrix[index]