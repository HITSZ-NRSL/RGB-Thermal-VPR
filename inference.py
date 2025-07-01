import faiss
import torch
import logging
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset
# from datetime import datetime
import time

# TODO: can be less memory cost
# TODO: finish the uncompleted parts
def inference(args, eval_ds, model, pca=None, k=1, use_cuda=True, verbose=True):
    '''
    hard_resize: directly use the resized image
    single_query: use the resized image, and set query_infer_batchsize=1 (used when the query images have varying size)
    central_crop: Take the biggest central crop of size self.resize. Preserves ratio.
    five_crops: use five crops of the image, and take the average of the features
    nearest_crop: use five crops of the image, 
    maj_voting: calculate features of five crops of the image, then use the nearest features
    * hard_size method for all database images
    * selected test_method for all query images
    '''

    # TODO test_efficient_ram_usage

    test_method = args.test_method
    
    model = model.eval()
    with torch.no_grad():
        ### Extract database features
        start_time = time.time()

        database_subset_ds = Subset(eval_ds, list(range(eval_ds.database_num)))
        database_dataloader = DataLoader(dataset=database_subset_ds, num_workers=args.num_workers,
                                        batch_size=args.infer_batch_size, pin_memory=(args.device=="cuda"))
    
        database_features = np.empty((eval_ds.database_num, args.features_dim), dtype="float32")
        
        # NOTE:GOOD CODE
        for inputs, indices in tqdm(database_dataloader, ncols=100):
            features = model(inputs.to(args.device)).view(-1, args.features_dim)
            features = features.cpu().numpy()
            database_features[indices.numpy(), :] = features

        logging.info(f"Finished extracting {eval_ds.database_num} database features in {time.time() - start_time:.2f} s")

        ### Extract query features
        start_time = time.time()

        queries_infer_batch_size = args.infer_batch_size
        queries_subset_ds = Subset(eval_ds, list(range(eval_ds.database_num, len(eval_ds))))
        queries_dataloader = DataLoader(dataset=queries_subset_ds, num_workers=args.num_workers,
                                        batch_size=queries_infer_batch_size, pin_memory=(args.device=="cuda"))

        queries_features = np.empty((eval_ds.queries_num, args.features_dim), dtype="float32")

        for inputs, indices in tqdm(queries_dataloader, ncols=100):

            features = model(inputs.to(args.device)).view(-1, args.features_dim)
            features = features.cpu().numpy()
            queries_features[indices.numpy()-eval_ds.database_num, :] = features

        logging.info(f"Finished extracting {eval_ds.queries_num} query features in {time.time() - start_time:.2f} s")
    
    # queries_features = all_features[eval_ds.database_num:]
    # database_features = all_features[:eval_ds.database_num]

    faiss_index = faiss.IndexFlatL2(args.features_dim)
    faiss_index.add(database_features)
    # del database_features, all_features
    del database_features

    ### Calculating recalls
    start_time = time.time()
    
    distances, predictions = faiss_index.search(queries_features, max(args.recall_values))
    del queries_features

    # NOTE: default args.recall_values is 20

    #### For each query, check if the predictions are correct
    positives_per_query = eval_ds.get_positives()
    # args.recall_values by default is [1, 5, 10, 20]
    recalls = np.zeros(len(args.recall_values))
    pre_num = eval_ds.queries_num
    for query_index, pred in enumerate(predictions):
        for i, n in enumerate(args.recall_values):
            # 如果没有正样本
            # print("type(positives_per_query[query_index]): ", type(positives_per_query[query_index]))
            # if positives_per_query[query_index].size == 0:
            #     eval_ds.queries_num -= 1
            #     continue
            if np.any(np.in1d(pred[:n], positives_per_query[query_index])):
                recalls[i:] += 1
                print(f"pred[:{n}]: {pred[:n]}")
                print(f"positives_per_query[{query_index}]: {positives_per_query[query_index]}")
                print(f"recalls: {recalls}")
                break
            elif i == 3:
                print(f"failed query_index: {query_index}")
                print(f"pred[:{n}]: {pred[:n]}")
                print(f"positives_per_query[{query_index}]: {positives_per_query[query_index]}")
                # print(f"recalls: {recalls}")
    # Divide by the number of queries*100, so the recalls are in percentages
    recalls = recalls / eval_ds.queries_num * 100
    # 以列表的形式打印，要用逗号隔开
    logging.info(f"recalls: {','.join(map(str, recalls))}")
    recalls_str = ", ".join([f"R@{val}: {rec:.1f}" for val, rec in zip(args.recall_values, recalls)])
    print(pre_num, eval_ds.queries_num)
    # logging.info(f"Finished calculating recalls in {time.time() - start_time:.2f} s")
    
    return recalls, recalls_str

### TODO
def top_n_voting(topn, predictions, distances, maj_weight):
    if topn == 'top1':
        n = 1
        selected = 0
    elif topn == 'top5':
        n = 5
        selected = slice(0, 5)
    elif topn == 'top10':
        n = 10
        selected = slice(0, 10)
    # find predictions that repeat in the first, first five,
    # or fist ten columns for each crop
    vals, counts = np.unique(predictions[:, selected], return_counts=True)
    # for each prediction that repeats more than once,
    # subtract from its score
    for val, count in zip(vals[counts > 1], counts[counts > 1]):
        mask = (predictions[:, selected] == val)
        distances[:, selected][mask] -= maj_weight * count/n
        
def fuse_inference(args, eval_ds, models):
    model_rgb, model_t = models
    ds_rgb, ds_t = eval_ds
    model_rgb.eval()
    model_t.eval()
    
    with torch.no_grad():
        ### Extract database features
        start_time = time.time()

        # eval_ds.test_method = "hard_resize"
        # NOTE: set batch_size == 1
        database_subset_ds_rgb = Subset(ds_rgb, list(range(ds_rgb.database_num)))
        database_dataloader_rgb = DataLoader(dataset=database_subset_ds_rgb, num_workers=args.num_workers,
                                        batch_size=1, pin_memory=(args.device=="cuda"))
        
        database_subset_ds_t = Subset(ds_t, list(range(ds_t.database_num)))
        database_dataloader_t = DataLoader(dataset=database_subset_ds_t, num_workers=args.num_workers,
                                        batch_size=1, pin_memory=(args.device=="cuda"))

        assert ds_rgb.database_num == ds_t.database_num
        
        database_features_rgb = np.empty((ds_rgb.database_num, args.features_dim), dtype="float32")
        database_features_t = np.empty((ds_t.database_num, args.features_dim), dtype="float32")
        database_features_cat = np.empty((ds_rgb.database_num, 2*args.features_dim), dtype="float32")
        database_features_add = np.empty((ds_rgb.database_num, args.features_dim), dtype="float32")

        for inputs, indices in tqdm(database_dataloader_rgb, ncols=100):
            features = model_rgb(inputs.to(args.device)).view(-1, args.features_dim)
            features = features.cpu().numpy()
            # if pca:
            #     features = pca.transform(features)
            database_features_rgb[indices.numpy(), :] = features
            database_features_cat[indices.numpy(), :args.features_dim] = features
            database_features_add[indices.numpy(), :] = features
        
        for inputs, indices in tqdm(database_dataloader_t, ncols=100):
            features = model_t(inputs.to(args.device)).view(-1, args.features_dim)
            features = features.cpu().numpy()
            # if pca:
            #     features = pca.transform(features)
            database_features_t[indices.numpy(), :] = features
            database_features_cat[indices.numpy(), args.features_dim:] = features
            database_features_add[indices.numpy(), :] += features

        logging.info(f"Finished extracting {ds_rgb.database_num}*2 database features in {time.time() - start_time:.2f} s")

        ### Extract query features
        start_time = time.time()

        # single query
        queries_infer_batch_size = 1
        # eval_ds.test_method = test_method
        queries_subset_ds_rgb = Subset(ds_rgb, list(range(ds_rgb.database_num, len(ds_rgb))))
        queries_dataloader_rgb = DataLoader(dataset=queries_subset_ds_rgb, num_workers=args.num_workers,
                                        batch_size=queries_infer_batch_size, pin_memory=(args.device=="cuda"))
        queries_subset_ds_t = Subset(ds_t, list(range(ds_t.database_num, len(ds_t))))
        queries_dataloader_t = DataLoader(dataset=queries_subset_ds_t, num_workers=args.num_workers,
                                        batch_size=queries_infer_batch_size, pin_memory=(args.device=="cuda"))
        
        assert ds_rgb.queries_num == ds_t.queries_num
        
        queries_features_rgb = np.empty((ds_rgb.queries_num, args.features_dim), dtype="float32")
        queries_features_t = np.empty((ds_t.queries_num, args.features_dim), dtype="float32")
        queries_features_cat = np.empty((ds_rgb.queries_num, 2*args.features_dim), dtype="float32")
        queries_features_add = np.empty((ds_rgb.queries_num, args.features_dim), dtype="float32")
        
        for inputs, indices in tqdm(queries_dataloader_rgb, ncols=100):
            features = model_rgb(inputs.to(args.device)).view(-1, args.features_dim)
            features = features.cpu().numpy()
            queries_features_rgb[indices.numpy()-ds_rgb.database_num, :] = features
            queries_features_cat[indices.numpy()-ds_rgb.database_num, :args.features_dim] = features
            queries_features_add[indices.numpy()-ds_rgb.database_num, :] = features
        
        for inputs, indices in tqdm(queries_dataloader_t, ncols=100):
            features = model_t(inputs.to(args.device)).view(-1, args.features_dim)
            features = features.cpu().numpy()
            queries_features_t[indices.numpy()-ds_rgb.database_num, :] = features
            queries_features_cat[indices.numpy()-ds_rgb.database_num, args.features_dim:] = features
            queries_features_add[indices.numpy()-ds_rgb.database_num, :] += features
                 
        logging.info(f"Finished extracting {ds_rgb.queries_num}*2 query features in {time.time() - start_time:.2f} s")
    
    faiss_index_rgb = faiss.IndexFlatL2(args.features_dim)
    faiss_index_t = faiss.IndexFlatL2(args.features_dim)
    faiss_index_cat = faiss.IndexFlatL2(2*args.features_dim)
    faiss_index_add = faiss.IndexFlatL2(args.features_dim)
    
    faiss_index_rgb.add(database_features_rgb)
    del database_features_rgb
    faiss_index_t.add(database_features_t)
    del database_features_t
    faiss_index_cat.add(database_features_cat)
    del database_features_cat
    faiss_index_add.add(database_features_add)
    del database_features_add

    ### Calculating recalls
    start_time = time.time()
    
    predictions_all = []
    distances, predictions = faiss_index_rgb.search(queries_features_rgb, max(args.recall_values))
    predictions_all.append(predictions)
    del queries_features_rgb
    distances, predictions = faiss_index_t.search(queries_features_t, max(args.recall_values))
    predictions_all.append(predictions)
    del queries_features_t
    distances, predictions = faiss_index_cat.search(queries_features_cat, max(args.recall_values))
    predictions_all.append(predictions)
    del queries_features_cat
    distances, predictions = faiss_index_add.search(queries_features_add, max(args.recall_values))
    predictions_all.append(predictions)
    del queries_features_add

    split = {
    'morning':list(range(0,365))+list(range(1371,1770))+list(range(2754,2869)),
    'afternoon':list(range(365,892))+list(range(1770,2231))+list(range(2869,2994)),
    'evening':list(range(892,1371))+list(range(2231,2754))+list(range(2994,3130)),
    'allday':list(range(0,3130))
    }
    #### For each query, check if the predictions are correct
    positives_per_query = ds_rgb.get_positives()
    # args.recall_values by default is [1, 5, 10, 20]
    recalls = {'morning':np.zeros((4, len(args.recall_values))), 'afternoon':np.zeros((4, len(args.recall_values))),\
        'evening':np.zeros((4, len(args.recall_values))), 'allday':np.zeros((4, len(args.recall_values)))}
    recalls_str = {'morning':[""]*4, 'afternoon':[""]*4, 'evening':[""]*4, 'allday':[""]*4}
    for k, predictions in enumerate(predictions_all):   # rgb, t, cat, add
        for key, indices in split.items(): # morning, afternoon, evening, allday
            for query_index in indices:
                pred = predictions[query_index]
                for i, n in enumerate(args.recall_values):
                    if np.any(np.in1d(pred[:n], positives_per_query[query_index])):
                        recalls[key][k, i:] += 1
                        break
            # Divide by the number of queries*100, so the recalls are in percentages
            method = ['rgb', 't', 'cat', 'add']
            recalls[key][k] = recalls[key][k] / len(indices) * 100
            recalls_str[key][k] = ", ".join([f"{method[k]}/{key}  R@{val}: {rec:.1f}" for val, rec in zip(args.recall_values, recalls[key][k])])
    
    # logging.info(f"Finished calculating recalls in {time.time() - start_time:.2f} s")

    return recalls, recalls_str