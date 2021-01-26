import json, argparse
from tqdm import tqdm



def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_dir",
        default=None,
        type=str,
        required=True,
        help="Data input directory. The one you downloaded.",
    )

    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="Data output directory. To save dataset in the desired format.",
    )
    args = parser.parse_args()

    data_dir = args.input_dir
    json_save = args.output_dir

    with open(data_dir, "r", encoding="utf-8") as reader:
        my_data = json.load(reader)

    new_data=list()
    for data in tqdm(my_data['data']):
        for paragraph in data['paragraphs']:
            for qas in paragraph['qas']:
                data_dict=dict()
                paragraph_dict=dict()
                paragraph_dict['context']=paragraph['context']
                paragraph_dict['qas']=[qas]
                data_dict['paragraphs']=[paragraph_dict]
                data_dict['title'] = data['title']

                new_data.append(data_dict)
                # print()

    new_data_dict=dict()
    print('QA samples=',len(new_data))
    new_data_dict['data']=new_data
    new_data_dict['version']='similar to longformer triviaQA'

    with open(json_save, 'w',encoding="utf-8") as outfile:
        json.dump(new_data_dict, outfile)

    return


if __name__ == "__main__":
    main()
