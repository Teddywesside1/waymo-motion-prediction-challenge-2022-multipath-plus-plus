import multiprocessing
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
# import tensorflow as tf
from utils.prerender_utils import get_visualizers, create_dataset, parse_arguments, merge_and_save
from utils.utils import get_config
from utils.features_description import generate_features_description

def main():
    import tensorflow as tf
    args = parse_arguments()
    dataset = create_dataset(args.data_path, args.n_shards, args.shard_id)
    visualizers_config = get_config(args.config)
    visualizers = get_visualizers(visualizers_config)

    with ThreadPoolExecutor(max_workers=args.n_jobs) as tp:
        print("start parsing tf_example!")
        thread_handles = []
        for data in tqdm(dataset.as_numpy_iterator()):
            data = tf.io.parse_single_example(data, generate_features_description())
            # merge_and_save(visualizers, data, args.output_path)
            task = tp.submit(merge_and_save, visualizers, data, args.output_path)
            thread_handles.append(task)
        
        pbar = tqdm(len(thread_handles))
        print("checking if tasks all finished")
        for task in as_completed(thread_handles):
            pbar.update(1)
            task.result()


    # p = multiprocessing.Pool(args.n_jobs)
    # processes = []
    # k = 0
    # for data in tqdm(dataset.as_numpy_iterator()):
    #     k += 1
    #     data = tf.io.parse_single_example(data, generate_features_description())
    #     # merge_and_save(visualizers, data, args.output_path)

    #     processes.append(
    #         p.apply_async(
    #             merge_and_save,
    #             kwds=dict(
    #                 visualizers=visualizers,
    #                 data=data,
    #                 output_path=args.output_path,
    #             ),
    #         )
    #     )
    
    # pbar = tqdm(len(processes))
    # for r in processes:
    #     pbar.n += 1
    #     r.get()

if __name__ == "__main__":
    main()