import os
from data.adam3d_cpmf import adam3d_classes
from multiprocessing import Pool


if __name__ == "__main__":
    classes = adam3d_classes()

    color_options = 'UNIFORM'
    data_path = "D:\Data\cpmf_multiview"

    n_processes = len(data_path)
    pool = Pool(processes=1)

    # n_views = [1, 3, 6, 9, 12, 15, 18, 21, 24, 27]
    n_views = [27]

    backbone_names = ['resnet18']

    no_fpfh_list = [False, True]

    for backbone in backbone_names:
        for no_fpfh in no_fpfh_list:
            for n in n_views:
                for cls in classes:
                    exp_name = f'{color_options}_{backbone}_{n}'
                    if not no_fpfh:
                        exp_name = f'{exp_name}_fpfh'
                    sh = f'python main.py --category {cls} --n-views {n} --no-fpfh {no_fpfh} --data-path {data_path} ' \
                         f'--exp-name {exp_name} --use-rgb {True} ' \
                         f'--backbone {backbone} --draw {True}'

                    print(f'exec {sh}')
                    pool.apply_async(os.system, (sh,))

    pool.close()
    pool.join()


