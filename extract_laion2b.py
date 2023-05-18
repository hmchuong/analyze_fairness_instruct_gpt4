from datasets import load_dataset
from multiprocessing import Pool
import tqdm

occupations = []
dataset = None

def search_data(idx):
    global dataset, occupations
    try:
        row = dataset[idx]
        text = row['TEXT'].lower()
    except:
        return None, None
    for occ in occupations:
        if occ in text:
            return row['URL'], occ
    
    return None, None

if __name__ == "__main__":
    dataset = load_dataset("laion/laion2B-en")['train']
    with open('occupations.txt') as f:
        occupations = [line[:35].rstrip() for line in f]
        occupations = occupations[1:]
        occupations = sorted(occupations)
    
    ids = range(dataset.num_rows)
    
    extracted = 0
    with Pool(processes=80) as p:
        with tqdm.tqdm(total=len(ids)) as pbar:
            for url, occ in p.imap_unordered(search_data, ids):
                pbar.update()
                if url is None:
                    continue
                with open(f"laion/{occ.replace(' ', '_')}.txt", "a+") as f:
                    f.write(f"{url}\n")
                extracted += 1
                pbar.set_description("Found {}".format(extracted))
    