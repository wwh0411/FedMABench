from utils.io import merge_json, load_json, dump_json
from template import *

def split_json(path):
    data_list = load_json(path)
    path1 = path[:-5] + '_1.json'
    path2 = path[:-5] + '_2.json'

    data_list_1 = data_list[:int(len(data_list) / 2)]
    data_list_2 = data_list[int(len(data_list) / 2):]

    dump_json(data_list_1, path1)
    dump_json(data_list_2, path2)


def sample_json(path, num_samples):
    new_path = path.replace("train_5000", f"train_{num_samples}")
    train_data_dir_hl = new_path.replace("data_format", 'high')
    train_data_dir_ll = new_path.replace("data_format", 'low')
    json_data = load_json(path)[:num_samples]
    dump_json(json_data, new_path)

    def get_train_format(resu_list, level='high'):
        data_list = []
        data_origin_list = []

        for index, entry in enumerate(resu_list):
            # print(entry)
            # if index == choose:
            #     break
            # print('ins：', instruction)
            acts = entry["acts"]
            imgs = entry["imgs"]
            for i in range(entry['num_step']):
                if level == 'high':
                    query = template_train_hl.format(ins=entry['ins_pre'])
                    query_gt = template_train_hl.format(ins=entry['ins_gt'])
                else:
                    if entry["des_acts"]:
                        query = template_train_ll.format(ins=entry['ins_pre'], sub_ins=entry["des_acts"][i])
                    else:
                        query = template_train_ll.format(ins=entry['ins_pre'], sub_ins='')
                    try:
                        query_gt = template_train_ll.format(ins=entry['ins_gt'], sub_ins=entry["sub_ins_gt"][i])
                    except:
                        query_gt = template_train_ll.format(ins=entry['ins_gt'], sub_ins='')
                data_list.append({'query': query,
                                  'response': acts[i],
                                  'images': imgs[i]})
                data_origin_list.append({'query': query_gt,
                                         'response': acts[i],
                                         'images': imgs[i]})
        return data_list, data_origin_list


    data_list, data_origin_list = get_train_format(json_data, 'high')
    dump_json(data_list, train_data_dir_hl)
    dump_json(data_origin_list, train_data_dir_hl.replace('alg', 'gt'))
    data_list, data_origin_list = get_train_format(json_data, 'low')
    dump_json(data_list, train_data_dir_ll)
    dump_json(data_origin_list, train_data_dir_ll.replace('alg', 'gt'))





# sample_json('/ailab/user/wangwenhao/ms-swift/output/internvl2-8b/data_format/alg3_train_5000_v1.json', 1000)
# sample_json('/ailab/user/wangwenhao/ms-swift/output_aitw/qwen2-7b/data_format/web_shopping_alg4_train_5000_v1.json', 1000)
# sample_json('/ailab/user/wangwenhao/ms-swift/output_aitw/qwen2-7b/data_format/single_alg4_train_5000_v1.json', 1000)

sample_json('/ailab/user/wangwenhao/ms-swift/output/internvl2-1b/data_format/alg4_train_5000_v1.json', 1000)

# sample_json('/ailab/user/wangwenhao/ms-swift/output/data_format/alg2_train_5000_v1.json', 1000)
# sample_json('/ailab/user/wangwenhao/ms-swift/output/data_format/alg4_train_5000_v1.json', 1000)
# # sample_json('/ailab/user/wangwenhao/ms-swift/output/data_format/gt_train_5000_v1.json', 1000)
# sample_json('/ailab/user/wangwenhao/ms-swift/output/data_format/alg3_train_5000_v1.json', 1000)
# split_json('/ailab/user/wangwenhao/ms-swift/output/internvl2-2b/high/alg4_train_1000_v1.json')
# split_json('/ailab/user/wangwenhao/ms-swift/output/internvl2-2b/high/gt_train_1000_v1.json')

# split_json('/ailab/user/wangwenhao/ms-swift/output/internvl2-2b/low_none/alg4_train_5000_v1.json')
# split_json('/ailab/user/wangwenhao/ms-swift/output/internvl2-2b/low_none/gt_train_5000_v1.json')
# merge_json('/ailab/user/wangwenhao/ms-swift/output/internvl2-2b/low_none/gt_train_5000_v1_1.json', '/ailab/user/wangwenhao/ms-swift/output/internvl2-2b/low/alg4_train_5000_v1_2.json',
#            '/ailab/user/wangwenhao/ms-swift/output/internvl2-2b/low/gt_high+alg4_train_5000_v1.json')

# split_json('/ailab/user/wangwenhao/ms-swift/output/internvl2-2b/low/alg4_train_5000_v1.json')
# split_json('/ailab/user/wangwenhao/ms-swift/output/internvl2-2b/low/gt_train_5000_v1.json')
# merge_json('/ailab/user/wangwenhao/ms-swift/output/internvl2-2b/low/gt_train_5000_v1_1.json', '/ailab/user/wangwenhao/ms-swift/output/internvl2-2b/low/alg4_train_5000_v1_2.json',
#            '/ailab/user/wangwenhao/ms-swift/output/internvl2-2b/low/gt+alg4_train_5000_v1.json')

# merge_json('/ailab/user/wangwenhao/ms-swift/output/low_none/gt_train_1000_v1_1.json', '/ailab/user/wangwenhao/ms-swift/output/low/alg4_train_1000_v1_2.json',
#            '/ailab/user/wangwenhao/ms-swift/output/low/gt_high+alg4_train_1000_v1.json')

