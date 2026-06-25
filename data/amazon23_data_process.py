"""
amazon23_data_process.py
========================

Process the **Amazon Reviews 2023** (McAuley-Lab/Amazon-Reviews-2023) dataset into
the *exact* same set of output files / format produced by `amazon18_data_process.py`
in this same directory.

The only intentional differences from the amazon18 pipeline are:
    1. Data source: raw reviews + metadata are pulled straight from the HuggingFace
       Hub (`load_dataset`) instead of local `*_5.json` / `meta_*.json` files. The
       2023 field names differ, so a thin adapter maps them onto the amazon18 schema
       and the rest of the pipeline is reused verbatim.
    2. K-core threshold defaults to **7** (instead of 5).
    3. History truncation defaults to **50** items (instead of 10).
    4. Train/valid/test split is **6:2:2** (instead of 8:1:1).
    5. The amazon18 <=20-word title filter and the "<3000 items -> expand time
       window" recursion are removed (no arbitrary filters / heuristics).

----------------------------------------------------------------------------------
PREPROCESSING DECISIONS INHERITED FROM amazon18_data_process.py
----------------------------------------------------------------------------------
(These are reproduced here exactly unless noted; see argparse for the knobs.)

 1. TITLE SELECTION (the ONLY content/metadata gate):
      - Drop items with no `title`, or a title containing the string '<span id'
        (broken HTML), or a title with <= 1 character.
      - Normalize: replace &quot; / &amp;, strip surrounding spaces and quotes.
      - NO title-length / word-count cap (the amazon18 <=20-word filter is removed
        per request — no arbitrary filters). description/brand/categories are never
        used to include or exclude an item; they are only stored.

 2. REMOVE ITEMS WITHOUT A VALID TITLE from the review stream (any review whose
    item didn't survive step 1 is discarded).

 3. ITERATIVE K-CORE FILTERING (default K=7):
      - Repeatedly drop users and items with fewer than K interactions until the
        graph is stable (no further removals). Counts are recomputed every pass.
      - This is the only OTHER inclusion gate besides the title (step 1).

 4. TIMESTAMP WINDOW FILTER: reviews outside [st_year/st_month, ed_year/ed_month]
    are skipped, and the check is applied *inside* the k-core loop (json2csv-style).
    Defaults (1996-01 .. 2023-10) span the whole dataset, so it is a no-op unless
    you deliberately narrow it.
    ## NOTE for 2023: timestamps are in MILLISECONDS and are converted to seconds.

 5. PER-USER CHRONOLOGICAL SORT: each user's interactions are sorted by timestamp.
    Duplicate (repeated) user-item interactions are KEPT (no dedup).

 6. SLIDING-WINDOW SEQUENCE GENERATION: for each user and each position i>=1,
    emit (history = items[max(i-H, 0):i], target = items[i]) where H is the history
    window (default 50). This yields one training example per "next item".

 7. GLOBAL CHRONOLOGICAL 6:2:2 SPLIT: all generated sequences are sorted by target
    timestamp, then sliced 60% / 20% / 20% into train / valid / test (a global
    leakage-aware split, NOT a per-user split).

 8. HISTORY CAP IN ATOMIC FILES: the written `item_id_list` is additionally capped
    to the last `--history_max` ids (default 50). With H==cap this is consistent.

 9. ID REMAPPING: user / item integer ids are assigned in order of first
    appearance while iterating users in (per-user timestamp-sorted) order.

10. ITEM FEATURES (`*.item.json`): {title, description, brand, categories}.
    For 2023, `description` (a list) is joined into one string and `brand` is taken
    from the 2023 `store` field; `categories` is the flat 2023 category list.

11. REVIEW DATA (`*.review.json`): keyed by str((uid, iid, unixReviewTime)) with
    {"review": cleaned text, "summary": cleaned title}.

----------------------------------------------------------------------------------
OUTPUT FILES (identical names/format to amazon18_data_process.py)
----------------------------------------------------------------------------------
  <output>/<dataset>/<dataset>.train.inter   (header: user_id:token \t item_id_list:token_seq \t item_id:token)
  <output>/<dataset>/<dataset>.valid.inter
  <output>/<dataset>/<dataset>.test.inter
  <output>/<dataset>/<dataset>.inter.json    (user_idx -> [item_idx, ...])
  <output>/<dataset>/<dataset>.item.json     (item_idx -> {title, description, brand, categories})
  <output>/<dataset>/<dataset>.review.json   ("(uid, iid, ts)" -> {review, summary})
  <output>/<dataset>/<dataset>.user2id       (original_user_id \t user_idx)
  <output>/<dataset>/<dataset>.item2id       (original_parent_asin \t item_idx)
"""

import argparse
import collections
import gzip
import html
import json
import os
import re
import datetime

from tqdm import tqdm


# ----------------------------------------------------------------------------------
# Shared helpers (identical to amazon18_data_process.py)
# ----------------------------------------------------------------------------------
def clean_text(text):
    """Clean text by removing HTML tags and excessive whitespace"""
    if not text:
        return ""
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', str(text))
    # Decode HTML entities
    text = html.unescape(text)
    # Replace quotes
    text = text.replace("&quot;", "\"").replace("&amp;", "&")
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def check_path(path):
    """Create directory if it doesn't exist"""
    os.makedirs(path, exist_ok=True)


def write_json_file(data, file_path):
    """Write data to JSON file"""
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)


def write_remap_index(index_map, file_path):
    """Write index mapping to file"""
    with open(file_path, 'w') as f:
        for original, mapped in index_map.items():
            f.write(f"{original}\t{mapped}\n")


def get_timestamp_start(year, month):
    """Get timestamp (epoch seconds) for the start of a given year and month"""
    return int(datetime.datetime(year=year, month=month, day=1, hour=0, minute=0,
                                 second=0, microsecond=0).timestamp())


# ----------------------------------------------------------------------------------
# Amazon Reviews 2023 specifics
# ----------------------------------------------------------------------------------
# The full set of category configs available on the Hub. `--dataset` must be one of
# these (a couple of short aliases are accepted for convenience).
AVAILABLE_CATEGORIES = [
    'All_Beauty', 'Amazon_Fashion', 'Appliances', 'Arts_Crafts_and_Sewing',
    'Automotive', 'Baby_Products', 'Beauty_and_Personal_Care', 'Books',
    'CDs_and_Vinyl', 'Cell_Phones_and_Accessories', 'Clothing_Shoes_and_Jewelry',
    'Digital_Music', 'Electronics', 'Gift_Cards', 'Grocery_and_Gourmet_Food',
    'Handmade_Products', 'Health_and_Household', 'Health_and_Personal_Care',
    'Home_and_Kitchen', 'Industrial_and_Scientific', 'Kindle_Store',
    'Magazine_Subscriptions', 'Movies_and_TV', 'Musical_Instruments',
    'Office_Products', 'Patio_Lawn_and_Garden', 'Pet_Supplies', 'Software',
    'Sports_and_Outdoors', 'Subscription_Boxes', 'Tools_and_Home_Improvement',
    'Toys_and_Games', 'Unknown', 'Video_Games',
]

# Convenience short names -> official 2023 category config.
CATEGORY_ALIASES = {
    'Arts': 'Arts_Crafts_and_Sewing',
    'Games': 'Video_Games',
    'Sports': 'Sports_and_Outdoors',
    'Instruments': 'Musical_Instruments',
    'Scientific': 'Industrial_and_Scientific',
    'Office': 'Office_Products',
}

HF_REPO = 'McAuley-Lab/Amazon-Reviews-2023'


def resolve_category(name):
    """Map a possibly-aliased dataset name to an official 2023 category config."""
    category = CATEGORY_ALIASES.get(name, name)
    assert category in AVAILABLE_CATEGORIES, (
        f'Category "{category}" not available for Amazon Reviews 2023. '
        f'Available categories: {AVAILABLE_CATEGORIES}'
    )
    return category


def _join_list_field(value):
    """Amazon-2023 description/features are lists of strings; join into one string."""
    if isinstance(value, list):
        return clean_text(' '.join(str(v) for v in value if v))
    return clean_text(value)


def _iter_jsonl(path):
    """Yield parsed JSON objects from a (optionally gzipped) JSON-lines file."""
    opener = gzip.open if path.endswith('.gz') else open
    with opener(path, 'rt', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _download_raw_jsonl(category, kind, cache_dir=None):
    """
    Download a raw Amazon-2023 JSONL from the Hub via `hf_hub_download` (no dataset
    loading script -> works on new `datasets`/`huggingface_hub`, no trust_remote_code).

    kind: 'review' -> raw/review_categories/<category>.jsonl
          'meta'   -> raw/meta_categories/meta_<category>.jsonl
    """
    from huggingface_hub import hf_hub_download

    if kind == 'review':
        filename = f'raw/review_categories/{category}.jsonl'
    elif kind == 'meta':
        filename = f'raw/meta_categories/meta_{category}.jsonl'
    else:
        raise ValueError(f'unknown kind: {kind}')

    return hf_hub_download(
        repo_id=HF_REPO, repo_type='dataset', filename=filename, cache_dir=cache_dir,
    )


def load_amazon23(category, cache_dir=None):
    """
    Download the raw 2023 reviews + metadata JSONL files from the HuggingFace Hub
    and adapt them onto the amazon18 internal schema so the rest of the pipeline is
    reused unchanged.

    Returns:
        reviews  : list of dicts with keys
                   {reviewerID, asin, overall, unixReviewTime, reviewText, summary}
        metadata : list of dicts with keys
                   {asin, title, description, brand, categories}
    """
    print(f'[DATASET] Downloading Amazon Reviews 2023 raw reviews for: {category}')
    review_path = _download_raw_jsonl(category, 'review', cache_dir=cache_dir)
    print(f'[DATASET] Downloading Amazon Reviews 2023 raw metadata for: {category}')
    meta_path = _download_raw_jsonl(category, 'meta', cache_dir=cache_dir)

    # ---- Reviews: map 2023 fields -> amazon18 keys -------------------------------
    # 2023 -> 2018:
    #   user_id       -> reviewerID
    #   parent_asin   -> asin           (2023 best practice: group variants by parent)
    #   rating        -> overall
    #   timestamp(ms) -> unixReviewTime (seconds)
    #   text          -> reviewText
    #   title         -> summary
    reviews = []
    for r in tqdm(_iter_jsonl(review_path), desc='Adapting reviews'):
        user_id = r.get('user_id')
        parent_asin = r.get('parent_asin')
        if user_id is None or parent_asin is None:
            continue
        rating = r.get('rating')
        ts = r.get('timestamp')
        reviews.append({
            'reviewerID': user_id,
            'asin': parent_asin,
            'overall': float(rating) if rating is not None else 0.0,
            # 2023 timestamps are in milliseconds -> convert to seconds.
            'unixReviewTime': int(ts) // 1000 if ts is not None else 0,
            'reviewText': r.get('text') or '',
            'summary': r.get('title') or '',
        })

    # ---- Metadata: map 2023 fields -> amazon18 keys ------------------------------
    # 2023 -> 2018:
    #   parent_asin -> asin
    #   title       -> title
    #   description -> description (list joined to str)
    #   store       -> brand
    #   categories  -> categories (flat list)
    metadata = []
    for m in tqdm(_iter_jsonl(meta_path), desc='Adapting metadata'):
        parent_asin = m.get('parent_asin')
        if parent_asin is None:
            continue
        metadata.append({
            'asin': parent_asin,
            'title': m.get('title') if m.get('title') is not None else '',
            'description': _join_list_field(m.get('description')),
            'brand': m.get('store') if m.get('store') is not None else '',
            'categories': m.get('categories') if m.get('categories') is not None else [],
        })

    print(f'[DATASET] Loaded {len(reviews)} reviews and {len(metadata)} metadata rows')
    return reviews, metadata


# ----------------------------------------------------------------------------------
# Pipeline (ported from amazon18_data_process.py; only the metadata/review loaders
# take in-memory lists, and K / history are parameterized)
# ----------------------------------------------------------------------------------
def build_id_title(metadata):
    """
    Decision #1 + #2: clean titles and select items that actually have a title.

    An item is kept iff it has a non-empty title that is not broken HTML
    ('<span id' junk). There is NO title-length / word-count cap (the amazon18
    <=20-word filter was intentionally dropped). description/brand/categories are
    never used to include or exclude an item.

    Returns:
        id_title     : {asin -> cleaned title}
        remove_items : set of asins to drop (missing / HTML-junk title)
    """
    id_title = {}
    remove_items = set()

    for meta in tqdm(metadata, desc="Processing metadata"):
        if ('title' not in meta) or (not meta['title']) or (str(meta['title']).find('<span id') > -1):
            remove_items.add(meta['asin'])
            continue

        # Clean title like json2csv / amazon18
        meta['title'] = str(meta["title"]).replace("&quot;", "\"").replace("&amp;", "&").strip(" ").strip("\"")

        # Keep any item with a real (non-trivial) title; no length cap.
        if len(meta['title']) > 1:
            id_title[meta['asin']] = meta['title']
        else:
            remove_items.add(meta['asin'])

    return id_title, remove_items


def k_core_filtering(reviews, id_title, remove_items_init, K=7,
                     start_timestamp=None, end_timestamp=None):
    """Decision #3 + #4: iterative K-core with in-loop timestamp window filter."""
    remove_users = set()
    remove_items = set(remove_items_init)

    # Remove reviews whose item has no usable title (Decision #2)
    for review in reviews:
        if review['asin'] not in id_title:
            remove_items.add(review['asin'])

    new_reviews = reviews
    while True:
        new_reviews = []
        flag = False
        total = 0
        user_counts = dict()
        item_counts = dict()

        for review in tqdm(reviews, desc="K-core filtering"):
            # Filter by timestamp INSIDE the loop like json2csv
            if start_timestamp and end_timestamp:
                if int(review["unixReviewTime"]) < start_timestamp or int(review["unixReviewTime"]) > end_timestamp:
                    continue

            if review['reviewerID'] in remove_users or review['asin'] in remove_items:
                continue

            user_counts[review['reviewerID']] = user_counts.get(review['reviewerID'], 0) + 1
            item_counts[review['asin']] = item_counts.get(review['asin'], 0) + 1

            total += 1
            new_reviews.append(review)

        for user in user_counts:
            if user_counts[user] < K:
                remove_users.add(user)
                flag = True

        for item in item_counts:
            if item_counts[item] < K:
                remove_items.add(item)
                flag = True

        density = total / (len(user_counts) * len(item_counts)) if user_counts and item_counts else 0
        print(f"Users: {len(user_counts)}, Items: {len(item_counts)}, Reviews: {total}, Density: {density}")

        if not flag:
            break

        reviews = new_reviews

    return new_reviews, user_counts, item_counts


def convert_inters2dict(reviews):
    """Decision #6 + #10: per-user chronological sort + first-appearance id remap."""
    user2items = collections.defaultdict(list)
    user2index, item2index = dict(), dict()

    user_reviews = collections.defaultdict(list)
    for review in reviews:
        user_reviews[review['reviewerID']].append(review)

    for user in user_reviews:
        user_reviews[user].sort(key=lambda x: int(x['unixReviewTime']))

    interactions = []
    for user in user_reviews:
        if user not in user2index:
            user2index[user] = len(user2index)

        user_items = []
        for review in user_reviews[user]:
            item = review['asin']
            if item not in item2index:
                item2index[item] = len(item2index)

            user_items.append(item)
            interactions.append((
                user, item,
                float(review['overall']),
                int(review['unixReviewTime'])
            ))

        user2items[user2index[user]] = [item2index[item] for item in user_items]

    return user2items, user2index, item2index, interactions


def generate_interaction_list(reviews, user2index, item2index, id_title, history_max=50):
    """Decision #7 + #8: sliding-window sequences, globally time-sorted."""
    interact = dict()
    item2id = {item: idx for item, idx in item2index.items()}

    for review in tqdm(reviews, desc="Building interaction list"):
        user = review['reviewerID']
        item = review['asin']
        if item not in item2id or item not in id_title:
            continue

        if user not in interact:
            interact[user] = {'items': [], 'ratings': [], 'timestamps': [], 'item_ids': [], 'titles': []}

        interact[user]['items'].append(item)
        interact[user]['ratings'].append(review['overall'])
        interact[user]['timestamps'].append(review['unixReviewTime'])
        interact[user]['item_ids'].append(item2id[item])
        interact[user]['titles'].append(id_title[item])

    interaction_list = []
    for user in tqdm(interact.keys(), desc="Creating interaction sequences"):
        items = interact[user]['items']
        ratings = interact[user]['ratings']
        timestamps = interact[user]['timestamps']
        item_ids = interact[user]['item_ids']
        titles = interact[user]['titles']

        all_data = list(zip(items, ratings, timestamps, item_ids, titles))
        all_data.sort(key=lambda x: int(x[2]))
        items, ratings, timestamps, item_ids, titles = zip(*all_data)
        items, ratings, timestamps, item_ids, titles = list(items), list(ratings), list(timestamps), list(item_ids), list(titles)

        # Sliding window with max history of `history_max`
        for i in range(1, len(items)):
            st = max(i - history_max, 0)
            interaction_list.append([
                user,                # user_id
                items[st:i],         # item_asins (history)
                items[i],            # item_asin (target)
                item_ids[st:i],      # history_item_id
                item_ids[i],         # item_id (target)
                titles[st:i],        # history_item_title
                titles[i],           # item_title (target)
                ratings[st:i],       # history_rating
                ratings[i],          # rating (target)
                timestamps[st:i],    # history_timestamp
                timestamps[i]        # timestamp (target)
            ])

    interaction_list.sort(key=lambda x: int(x[-1]))
    return interaction_list


def convert_to_atomic_files(args, interaction_list, user2index):
    """Decision #8 + #9: 8:1:1 chronological split, history capped to history_max."""
    print('Convert dataset: ')
    print(' Dataset: ', args.dataset)

    check_path(os.path.join(args.output_path, args.dataset))

    # Decision #8: global chronological 6:2:2 split (train / valid / test).
    total_len = len(interaction_list)
    train_end = int(total_len * 0.6)
    valid_end = int(total_len * 0.8)

    splits = {
        'train': interaction_list[:train_end],
        'valid': interaction_list[train_end:valid_end],
        'test': interaction_list[valid_end:],
    }

    print(f"Train interactions: {len(splits['train'])}")
    print(f"Valid interactions: {len(splits['valid'])}")
    print(f"Test interactions: {len(splits['test'])}")

    for split, data in splits.items():
        out_file = os.path.join(args.output_path, args.dataset, f'{args.dataset}.{split}.inter')
        with open(out_file, 'w') as file:
            file.write('user_id:token\titem_id_list:token_seq\titem_id:token\n')
            for interaction in data:
                user_id_original = interaction[0]
                user_id = user2index[user_id_original]
                history_item_ids = [str(x) for x in interaction[3]]
                target_item_id = str(interaction[4])

                # Limit history to last `history_max` items
                history_seq = history_item_ids[-args.history_max:]
                file.write(f'{user_id}\t{" ".join(history_seq)}\t{target_item_id}\n')

    return splits['train'], splits['valid'], splits['test']


def load_review_data(reviews, user2index, item2index):
    """Decision #12: review_data keyed by str((uid, iid, unixReviewTime))."""
    review_data = {}

    for review in tqdm(reviews, desc='Load reviews'):
        try:
            user = review['reviewerID']
            item = review['asin']

            if user in user2index and item in item2index:
                uid = user2index[user]
                iid = item2index[item]
                timestamp = review['unixReviewTime']
                unique_key = str((uid, iid, timestamp))
            else:
                continue

            review_text = clean_text(review['reviewText']) if 'reviewText' in review else ''
            summary = clean_text(review['summary']) if 'summary' in review else ''

            review_data[unique_key] = {"review": review_text, "summary": summary}
        except (ValueError, KeyError):
            continue

    return review_data


def create_item_features(metadata, item2index, id_title):
    """Decision #11: item2feature -> {title, description, brand, categories}."""
    item2feature = collections.defaultdict(dict)

    asin_to_meta = {meta['asin']: meta for meta in metadata}

    for item_asin, item_id in item2index.items():
        if item_asin in asin_to_meta:
            meta = asin_to_meta[item_asin]

            title = id_title.get(item_asin, clean_text(meta.get("title", "")))

            descriptions = meta.get("description", "")
            descriptions = clean_text(descriptions) if descriptions else ""

            brand = str(meta.get("brand", "") or "").replace("by\n", "").strip()

            categories = meta.get("categories", [])
            if categories and len(categories) > 0:
                # Handle both list and string formats
                if isinstance(categories[0], list):
                    flat_categories = []
                    for cat_group in categories:
                        flat_categories.extend(cat_group)
                    categories = flat_categories

                new_categories = []
                for category in categories:
                    if "</span>" not in str(category):
                        new_categories.append(str(category).strip())
                categories = ",".join(new_categories).strip()
            else:
                categories = ""

            item2feature[item_id] = {
                "title": title,
                "description": descriptions,
                "brand": brand,
                "categories": categories,
            }

    return item2feature


def process_dataset(args, metadata, reviews, start_timestamp, end_timestamp):
    """Decisions #1-#4: title selection, k-core filtering (within the time window)."""
    id_title, remove_items = build_id_title(metadata)

    if not id_title:
        print(f"Error: No items with usable titles for dataset {args.dataset}")
        return None

    print(f"Loaded {len(metadata)} metadata items, {len(id_title)} with valid titles")

    print("Performing k-core filtering...")
    filtered_reviews, user_counts, item_counts = k_core_filtering(
        reviews, id_title, remove_items, args.user_k, start_timestamp, end_timestamp
    )

    print(f"After filtering: {len(user_counts)} users, {len(item_counts)} items, {len(filtered_reviews)} reviews")

    return filtered_reviews, user_counts, item_counts, metadata, id_title


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Industrial_and_Scientific',
                        help='Amazon-2023 category config (or short alias, e.g. Scientific / Office)')
    parser.add_argument('--user_k', type=int, default=7, help='k-core filtering threshold (users and items)')
    parser.add_argument('--item_k', type=int, default=7, help='(kept for CLI parity; k-core uses --user_k)')
    parser.add_argument('--history_max', type=int, default=50, help='max user history length (was 10 in amazon18)')
    parser.add_argument('--st_year', type=int, default=1996, help='start year')
    parser.add_argument('--st_month', type=int, default=1, help='start month')
    parser.add_argument('--ed_year', type=int, default=2023, help='end year (2023 data ends ~Sep 2023)')
    parser.add_argument('--ed_month', type=int, default=10, help='end month')
    parser.add_argument('--cache_dir', type=str, default=None, help='HuggingFace hub cache dir')
    parser.add_argument('--output_path', type=str, default='./Amazon23', help='output directory')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    # `item_k` mirrors the amazon18 CLI but the iterative k-core uses a single K
    # (user_k) for both users and items, exactly like amazon18_data_process.py.
    args.dataset = resolve_category(args.dataset)

    print(f'Processing dataset: {args.dataset}')
    print(f'Initial time range: {args.st_year}-{args.st_month} to {args.ed_year}-{args.ed_month}')
    print(f'K-core threshold: {args.user_k}, history max: {args.history_max}')

    start_timestamp = get_timestamp_start(args.st_year, args.st_month)
    end_timestamp = get_timestamp_start(args.ed_year, args.ed_month)

    # Load reviews + metadata from the HuggingFace Hub (adapted to amazon18 schema)
    print("Loading Amazon Reviews 2023 data...")
    reviews, metadata = load_amazon23(args.dataset, cache_dir=args.cache_dir)

    if not reviews:
        print(f"Error: No reviews found for dataset {args.dataset}")
        exit(1)
    if not metadata:
        print(f"Error: No metadata found for dataset {args.dataset}")
        exit(1)

    print(f"Loaded {len(reviews)} total reviews")

    result = process_dataset(args, metadata, reviews, start_timestamp, end_timestamp)

    if result is None:
        print("Failed to process dataset")
        exit(1)

    filtered_reviews, user_counts, item_counts, metadata, id_title = result

    print("Final filtering results:")
    density = len(filtered_reviews) / (len(user_counts) * len(item_counts)) if user_counts and item_counts else 0
    print(f"Users: {len(user_counts)}, Items: {len(item_counts)}, Reviews: {len(filtered_reviews)}")
    print(f"Density: {density}")

    print("Converting to amazon18 format...")
    user2items, user2index, item2index, interactions = convert_inters2dict(filtered_reviews)

    print("After conversion:")
    print(f"  User2index: {len(user2index)} users")
    print(f"  Item2index: {len(item2index)} items")
    print(f"  Interactions: {len(interactions)} interactions")

    print("Generating interaction list for 8:1:1 split...")
    interaction_list = generate_interaction_list(
        filtered_reviews, user2index, item2index, id_title, history_max=args.history_max
    )

    print(f"Generated {len(interaction_list)} interaction sequences")

    train_interactions, valid_interactions, test_interactions = convert_to_atomic_files(
        args, interaction_list, user2index
    )

    user2items_final = collections.defaultdict(list)
    for user_idx, item_list in user2items.items():
        user2items_final[user_idx] = item_list

    write_json_file(user2items_final, os.path.join(args.output_path, args.dataset, f'{args.dataset}.inter.json'))

    print("Creating item features...")
    item2feature = create_item_features(metadata, item2index, id_title)

    print("Loading review data...")
    review_data = load_review_data(filtered_reviews, user2index, item2index)

    print("Final statistics:")
    print(f"Users: {len(user2index)}")
    print(f"Items: {len(item2index)}")
    print(f"Reviews: {len(review_data)}")
    print(f"Total interaction sequences: {len(interaction_list)}")
    print(f"Train sequences: {len(train_interactions)}")
    print(f"Valid sequences: {len(valid_interactions)}")
    print(f"Test sequences: {len(test_interactions)}")

    write_json_file(item2feature, os.path.join(args.output_path, args.dataset, f'{args.dataset}.item.json'))
    write_json_file(review_data, os.path.join(args.output_path, args.dataset, f'{args.dataset}.review.json'))

    write_remap_index(user2index, os.path.join(args.output_path, args.dataset, f'{args.dataset}.user2id'))
    write_remap_index(item2index, os.path.join(args.output_path, args.dataset, f'{args.dataset}.item2id'))

    print("Processing completed!")
