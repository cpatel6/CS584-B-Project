"""
Synthetic sentiment dataset generator.

Produces realistic-enough Amazon-review-style and IMDb-style texts
with binary sentiment labels (0 = negative, 1 = positive) so the
pipeline can run without a network connection.
"""

import random
import pandas as pd
import os

# ---------------------------------------------------------------------------
# Template vocabulary
# ---------------------------------------------------------------------------

_POSITIVE_PRODUCT = [
    "This product is absolutely fantastic and exceeded all my expectations.",
    "I am very happy with this purchase. Works perfectly right out of the box.",
    "Excellent quality for the price. I would definitely recommend to a friend.",
    "Outstanding build quality. It has been working flawlessly for months.",
    "Great value! Does exactly what the description says and more.",
    "I love this item. It arrived quickly and the packaging was secure.",
    "Five stars all the way. Best purchase I have made this year.",
    "Super easy to set up and use. Very satisfied with the results.",
    "Highly recommend this product. It solved my problem immediately.",
    "Amazing product. The quality is top-notch and delivery was fast.",
    "Works exactly as advertised. I could not be happier with my order.",
    "Very durable and well made. It looks even better in person.",
    "Perfect for what I needed. Great customer service too.",
    "Solid construction and comfortable to use. Worth every penny.",
    "This exceeded my expectations in every way. Will buy again.",
    "Really impressed with how well this works. Excellent design.",
    "Good quality product and fast shipping. Very pleased overall.",
    "Best purchase in a long time. Highly recommend to everyone.",
    "Very happy with this. It does everything I hoped it would.",
    "Wonderful product. Easy to use and very effective.",
]

_NEGATIVE_PRODUCT = [
    "This product broke after just two days. Very disappointing.",
    "Terrible quality. Nothing like the pictures shown on the listing.",
    "Complete waste of money. I would not recommend this to anyone.",
    "Stopped working after one week. The build quality is very poor.",
    "Very cheap feeling materials. Arrived damaged and unusable.",
    "I am extremely dissatisfied. This does not match the description at all.",
    "Returned immediately. The product is defective right out of the box.",
    "Zero stars if I could. Worst purchase I have ever made.",
    "Very poor quality control. The item fell apart on the first use.",
    "Do not buy this. It is a complete scam and waste of your money.",
    "Broke immediately. Packaging was terrible and product was already cracked.",
    "Really bad quality. Instructions were unclear and it does not function.",
    "Awful product. Customer service was unhelpful when I tried to return it.",
    "Total junk. Nothing works as advertised. Extremely frustrated.",
    "Disappointed with the purchase. Quality is far below expectations.",
    "Not worth the money at all. I regret buying this product.",
    "Very unhappy with this item. It stopped working after a single use.",
    "Poor design and weak materials. Do not waste your money on this.",
    "This item is defective. I had to throw it away after one day.",
    "Misleading product listing. The real item is nothing like described.",
]

_POSITIVE_MOVIE = [
    "This film is a masterpiece. The acting and direction are superb.",
    "One of the best movies I have seen in years. Highly recommended.",
    "An incredible cinematic experience. The story was deeply moving.",
    "Brilliant performances from the entire cast. A must-see film.",
    "A beautiful and powerful movie. Left me speechless at the end.",
    "Exceptional storytelling with breathtaking visuals. Loved every moment.",
    "This movie touched my heart. A genuinely emotional experience.",
    "Great writing and outstanding performances. I thoroughly enjoyed it.",
    "A true work of art. The cinematography and soundtrack are perfect.",
    "I watched this three times and loved it more each time. Wonderful.",
    "Riveting from start to finish. The best film of the year.",
    "Incredibly well-made movie with a compelling and original story.",
    "A fantastic watch. The characters are rich and the plot is gripping.",
    "Excellent movie. Great pacing and the ending was very satisfying.",
    "An absolute joy to watch. I recommend this to every film lover.",
    "Moving and thought-provoking. This movie will stay with me forever.",
    "Superb direction and a powerful script. This movie deserves all awards.",
    "One of the greatest films ever made. Completely blown away.",
    "Loved the characters and the narrative. An amazing viewing experience.",
    "This film exceeded every expectation. A genuine five-star movie.",
]

_NEGATIVE_MOVIE = [
    "Terrible movie. The plot made no sense and the acting was wooden.",
    "One of the worst films I have ever seen. A total waste of time.",
    "The story was boring and predictable. I fell asleep halfway through.",
    "Poor direction and awful dialogue. I cannot recommend this to anyone.",
    "Dull, uninspired, and painfully slow. I wanted to leave the theater.",
    "The special effects were cheap and the story was completely incoherent.",
    "A very disappointing film. None of the characters were likable at all.",
    "Terrible writing. The plot had too many holes to enjoy the movie.",
    "Extremely boring and poorly acted. I regret watching this film.",
    "Absolute disaster of a movie. The ending was particularly frustrating.",
    "Not worth your time. The script is lazy and the pacing is awful.",
    "Possibly the worst film released this decade. Completely unwatchable.",
    "The actors looked bored and the director had no vision whatsoever.",
    "Badly made from start to finish. The editing was jarring and confusing.",
    "I cannot understand the positive reviews. This film is genuinely terrible.",
    "Overcomplicated and dull. The CGI looked outdated and the acting was weak.",
    "Wasted potential. A great concept ruined by poor execution.",
    "One of the most boring films I have ever suffered through.",
    "Painfully unfunny and unengaging. Complete disappointment throughout.",
    "Do not waste your money on this. A forgettable and mediocre film.",
]

# Additional sentence fragments to augment variety
_POS_FRAGMENTS = [
    "I would buy this again.", "Shipping was very fast.", "The quality surprised me.",
    "I am impressed overall.", "Works great from day one.", "Really good value.",
    "Easy to use and reliable.", "Very happy with the result.", "No complaints at all.",
    "Exactly as described.", "Really exceeded my expectations.", "Highly satisfied.",
    "Works as advertised.", "Great product, great service.", "I love it.",
]

_NEG_FRAGMENTS = [
    "Very disappointing overall.", "Would not recommend.", "Save your money.",
    "Not as described.", "Poor customer support.", "Broke too quickly.",
    "Felt cheaply made.", "Instructions were useless.", "Nothing worked correctly.",
    "Extremely frustrating experience.", "Complete waste of money.", "Very poor value.",
    "Never buying from this brand again.", "Defective right away.", "Terrible experience.",
]


def _generate_texts(base_templates, fragments, n: int, seed: int):
    rng = random.Random(seed)
    texts = []
    for i in range(n):
        base = rng.choice(base_templates)
        # optionally append a fragment for variety
        if rng.random() < 0.6:
            base = base + " " + rng.choice(fragments)
        texts.append(base)
    return texts


def generate_amazon_polarity(n: int = 2000, seed: int = 42) -> pd.DataFrame:
    """Generate n synthetic Amazon-polarity-style samples (balanced)."""
    half = n // 2
    remainder = n - half

    pos = _generate_texts(_POSITIVE_PRODUCT, _POS_FRAGMENTS, half, seed)
    neg = _generate_texts(_NEGATIVE_PRODUCT, _NEG_FRAGMENTS, remainder, seed + 1)

    texts = pos + neg
    labels = [1] * half + [0] * remainder

    df = pd.DataFrame({"text": texts, "label": labels})
    rng = random.Random(seed)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    return df


def generate_imdb(n: int = 1000, seed: int = 99) -> pd.DataFrame:
    """Generate n synthetic IMDb-style samples (balanced)."""
    half = n // 2
    remainder = n - half

    pos = _generate_texts(_POSITIVE_MOVIE, _POS_FRAGMENTS, half, seed)
    neg = _generate_texts(_NEGATIVE_MOVIE, _NEG_FRAGMENTS, remainder, seed + 1)

    texts = pos + neg
    labels = [1] * half + [0] * remainder

    df = pd.DataFrame({"text": texts, "label": labels})
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    return df


def save_synthetic_datasets(data_dir: str = "data", n_amazon: int = 2000,
                             n_imdb: int = 1000, seed: int = 42):
    os.makedirs(data_dir, exist_ok=True)
    amazon_path = os.path.join(data_dir, "amazon_polarity.csv")
    imdb_path = os.path.join(data_dir, "imdb.csv")

    if not os.path.exists(amazon_path):
        df = generate_amazon_polarity(n_amazon, seed)
        df.to_csv(amazon_path, index=False)
        print(f"Saved {len(df)} Amazon rows → {amazon_path}")

    if not os.path.exists(imdb_path):
        df = generate_imdb(n_imdb, seed + 100)
        df.to_csv(imdb_path, index=False)
        print(f"Saved {len(df)} IMDb rows → {imdb_path}")
