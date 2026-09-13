"""Synthetic comparative/ordering reasoning dataset for Hindi (Phase 3, section 3.1).

"Synthetic" here means every example is generated programmatically from a template plus
randomly-sampled entity names and numeric attribute values -- never scraped or downloaded --
so the correct answer is known exactly by construction (the script itself picked the numbers).
This is required specifically because grading finetuned-vs-pretrained accuracy needs a
guaranteed-correct label for every example, which real scraped text cannot provide.

Three task types, per the assignment's own examples:
  - pairwise:     given two entities' values for one attribute, which is greater/smaller?
  - transitive3:  given three entities' values, which is smallest/largest?
  - chain:        given A>B and B>C (stated only as two pairwise facts), what is the relation
                   between A and C? (answered as which of A/C is greater, for a consistent,
                   exact-match-scorable answer format across all three task types)

Attributes: height (cm), age (years), price (rupees), quantity (items) -- generic enough to
phrase naturally about people (height/age) or objects (price/quantity).

Train/test leakage control: the entity name pool is split into TRAIN_NAMES and TEST_NAMES
before any examples are generated, and test examples draw names *only* from TEST_NAMES -- so
test never reuses a name seen during training, not just a different random combination of the
same names (the assignment's explicit example of avoiding train-test leakage).
"""

import json
import os
import random

SEED = 1337

# Common Hindi given names (Devanagari) -- used as comparison-task entities. Split into a
# train pool and a disjoint test pool below, never mixed.
NAMES = [
    "राम", "श्याम", "मोहन", "सोहन", "गीता", "सीता", "रवि", "सुनीता", "अजय", "विजय",
    "प्रिया", "पूजा", "अनिल", "सुनील", "कविता", "रेखा", "दीपक", "अशोक", "नेहा", "रीना",
    "संजय", "मनोज", "सुरेश", "राजेश", "किरण", "सुमन", "अरुण", "वरुण", "मीना", "टीना",
    "गौरव", "सौरभ", "अंजलि", "काजल", "विकास", "निखिल", "पायल", "शालिनी", "रोहन", "करण",
    "दिव्या", "स्वाति", "हर्ष", "यश", "नीता", "ममता", "प्रकाश", "उमेश", "ललिता", "शोभा",
]

OBJECT_NAMES = [
    "किताब", "कलम", "मेज", "कुर्सी", "बैग", "साइकिल", "मोबाइल", "घड़ी", "जूता", "छाता",
    "बोतल", "थैला", "पंखा", "टोकरी", "आम", "सेब", "केला", "अनार", "तरबूज", "नारियल",
]

ATTRIBUTES = {
    "height": {"unit": "सेमी", "min": 140, "max": 200, "subjects": "people"},
    "age": {"unit": "साल", "min": 5, "max": 90, "subjects": "people"},
    "price": {"unit": "रुपये", "min": 10, "max": 10000, "subjects": "objects"},
    "quantity": {"unit": "", "min": 1, "max": 100, "subjects": "objects"},
}

PAIRWISE_TEMPLATES = {
    "height": [
        "{a} की लंबाई {va} {unit} है और {b} की लंबाई {vb} {unit} है। कौन लंबा है?",
        "{a} की ऊंचाई {va} {unit} है, जबकि {b} की ऊंचाई {vb} {unit} है। इनमें से कौन ज्यादा लंबा है?",
    ],
    "age": [
        "{a} की उम्र {va} साल है और {b} की उम्र {vb} साल है। कौन बड़ा है?",
        "{a} की आयु {va} साल है, जबकि {b} की आयु {vb} साल है। इनमें से कौन छोटा है?",
    ],
    "price": [
        "{a} की कीमत {va} रुपये है और {b} की कीमत {vb} रुपये है। कौन महंगा है?",
        "{a} की क़ीमत {va} रुपये है, जबकि {b} की क़ीमत {vb} रुपये है। इनमें से कौन सस्ता है?",
    ],
    "quantity": [
        "{a} की संख्या {va} है और {b} की संख्या {vb} है। किसकी संख्या ज्यादा है?",
        "{a} {va} की संख्या में हैं, जबकि {b} {vb} की संख्या में हैं। किसकी संख्या कम है?",
    ],
}

TRANSITIVE_TEMPLATES = {
    "height": ["{a} की लंबाई {va} {unit}, {b} की लंबाई {vb} {unit}, और {c} की लंबाई {vc} {unit} है। इनमें सबसे लंबा कौन है?"],
    "age": ["{a} की उम्र {va} साल, {b} की उम्र {vb} साल, और {c} की उम्र {vc} साल है। इनमें सबसे बड़ा कौन है?"],
    "price": ["{a} की कीमत {va} रुपये, {b} की कीमत {vb} रुपये, और {c} की कीमत {vc} रुपये है। इनमें सबसे महंगा कौन है?"],
    "quantity": ["{a} की संख्या {va}, {b} की संख्या {vb}, और {c} की संख्या {vc} है। इनमें सबसे ज्यादा संख्या किसकी है?"],
}

# Two independent phrasing families so the correct answer is not always the first-named
# entity (which would let the model shortcut to "always output the first name" without
# actually reasoning): "ascending" states A>B>C and asks A vs C (answer named first);
# "descending" states C<B<A the other way round and asks C vs A (answer named last).
CHAIN_TEMPLATES_ASCENDING = [
    "{a} {b} से बड़ा है। {b} {c} से बड़ा है। {a} और {c} में से कौन बड़ा है?",
    "{a}, {b} से अधिक है। {b}, {c} से अधिक है। {a} और {c} में कौन अधिक है?",
]
CHAIN_TEMPLATES_DESCENDING = [
    "{c} {b} से छोटा है। {b} {a} से छोटा है। {c} और {a} में से कौन बड़ा है?",
    "{c}, {b} से कम है। {b}, {a} से कम है। {c} और {a} में कौन अधिक है?",
]


def gen_pairwise(names_pool, rng):
    attr = rng.choice(list(ATTRIBUTES.keys()))
    spec = ATTRIBUTES[attr]
    pool = OBJECT_NAMES if spec["subjects"] == "objects" else names_pool
    a, b = rng.sample(pool, 2)
    va, vb = rng.sample(range(spec["min"], spec["max"] + 1), 2)
    template = rng.choice(PAIRWISE_TEMPLATES[attr])
    ask_smaller = "छोटा" in template or "सस्ता" in template or "कम" in template
    prompt = template.format(a=a, b=b, va=va, vb=vb, unit=spec["unit"])
    if ask_smaller:
        answer = a if va < vb else b
    else:
        answer = a if va > vb else b
    return {"task_type": "pairwise", "attribute": attr, "prompt": prompt, "answer": answer,
            "entities": [a, b], "values": [va, vb]}


def gen_transitive3(names_pool, rng):
    attr = rng.choice(list(ATTRIBUTES.keys()))
    spec = ATTRIBUTES[attr]
    pool = OBJECT_NAMES if spec["subjects"] == "objects" else names_pool
    a, b, c = rng.sample(pool, 3)
    va, vb, vc = rng.sample(range(spec["min"], spec["max"] + 1), 3)
    template = TRANSITIVE_TEMPLATES[attr][0]
    prompt = template.format(a=a, b=b, c=c, va=va, vb=vb, vc=vc, unit=spec["unit"])
    values = {a: va, b: vb, c: vc}
    answer = max(values, key=values.get)
    return {"task_type": "transitive3", "attribute": attr, "prompt": prompt, "answer": answer,
            "entities": [a, b, c], "values": [va, vb, vc]}


def gen_chain(names_pool, rng):
    a, b, c = rng.sample(names_pool, 3)
    # a > b > c is stated (in one of two independently-phrased ways), so a > c always.
    if rng.random() < 0.5:
        template = rng.choice(CHAIN_TEMPLATES_ASCENDING)
    else:
        template = rng.choice(CHAIN_TEMPLATES_DESCENDING)
    prompt = template.format(a=a, b=b, c=c)
    answer = a
    return {"task_type": "chain", "attribute": None, "prompt": prompt, "answer": answer,
            "entities": [a, b, c], "values": None}


GENERATORS = [gen_pairwise, gen_transitive3, gen_chain]


def generate_split(names_pool, n, rng):
    examples = []
    seen_prompts = set()
    attempts = 0
    while len(examples) < n and attempts < n * 20:
        attempts += 1
        gen_fn = rng.choice(GENERATORS)
        ex = gen_fn(names_pool, rng)
        if ex["prompt"] in seen_prompts:
            continue
        seen_prompts.add(ex["prompt"])
        examples.append(ex)
    return examples


def main():
    rng = random.Random(SEED)
    names = NAMES.copy()
    rng.shuffle(names)
    n_test_names = max(8, int(len(names) * 0.2))
    test_names = names[:n_test_names]
    train_names = names[n_test_names:]

    train = generate_split(train_names, 4000, rng)
    val = generate_split(train_names, 500, rng)
    test = generate_split(test_names, 500, rng)

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    os.makedirs(out_dir, exist_ok=True)
    for split_name, data in [("train", train), ("val", val), ("test", test)]:
        with open(os.path.join(out_dir, f"{split_name}.jsonl"), "w", encoding="utf-8") as f:
            for ex in data:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    task_counts = {}
    for ex in train + val + test:
        task_counts[ex["task_type"]] = task_counts.get(ex["task_type"], 0) + 1

    stats = {
        "language": "hindi",
        "seed": SEED,
        "n_train": len(train), "n_val": len(val), "n_test": len(test),
        "n_train_names": len(train_names), "n_test_names": len(test_names),
        "train_test_name_overlap": len(set(train_names) & set(test_names)),
        "task_type_counts": task_counts,
        "attributes": list(ATTRIBUTES.keys()),
    }
    with open(os.path.join(out_dir, "stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
