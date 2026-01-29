from pipeline import pipeline
import multiprocessing
from utils.utils import get_args, ensure_work_dirs
import yaml
import json

def split_nums_evenly(num_workers, nums):
    base = nums // num_workers
    arr = [base] * (num_workers - 1)
    arr.append(nums - base * (num_workers - 1))
    return arr

def load_data_from_config(config):
    paths = config['data_paths']

    def read_json(path):
        if path:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return []

    text = read_json(paths.get('text', ''))
    figure = read_json(paths.get('image', ''))
    table = read_json(paths.get('table', ''))
    formula = read_json(paths.get('formula', ''))
    title = read_json(paths.get('title', ''))
    form = read_json(paths.get('form', ''))
    stamp = read_json(paths.get('stamp', ''))
    logo = read_json(paths.get('logo', ''))
    signature = read_json(paths.get('signature', ''))
    header = read_json(paths.get('header', ''))
    footer = read_json(paths.get('footer', ''))

    return title, table, text, formula, figure, form, stamp, logo, signature, header, footer

def chunkify(lst, n):
    k, m = divmod(len(lst), n)
    return [lst[i*k + min(i, m):(i+1)*k + min(i+1, m)] for i in range(n)]


if __name__ == "__main__":
    
    args = get_args()
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    title, table, text, formula, figure, form, stamp, logo, signature, header, footer = load_data_from_config(config)
    ensure_work_dirs(config)
    
    
    num_workers = config['num_workers']
    nums = config['nums']
    nums_list = split_nums_evenly(num_workers, nums)

    title_chunks   = chunkify(title,   num_workers)
    table_chunks   = chunkify(table,   num_workers)
    text_chunks    = chunkify(text,    num_workers)
    formula_chunks = chunkify(formula, num_workers)
    figure_chunks  = chunkify(figure,  num_workers)
    form_chunks    = chunkify(form,    num_workers)
    stamp_chunks   = chunkify(stamp,   num_workers)
    logo_chunks    = chunkify(logo,    num_workers)
    signature_chunks = chunkify(signature, num_workers)
    header_chunks  = chunkify(header,  num_workers)
    footer_chunks  = chunkify(footer,  num_workers)

    processes = []
    for i in range(num_workers):
        p = multiprocessing.Process(
            target=pipeline,
            args=(
                title_chunks[i],
                text_chunks[i],
                table_chunks[i],
                formula_chunks[i],
                figure_chunks[i],
                form_chunks[i],
                stamp_chunks[i],
                logo_chunks[i],
                signature_chunks[i],
                header_chunks[i],
                footer_chunks[i],
                nums_list[i],
                i
            )
        )
        processes.append(p)
        p.start()

    for p in processes:
        p.join()
