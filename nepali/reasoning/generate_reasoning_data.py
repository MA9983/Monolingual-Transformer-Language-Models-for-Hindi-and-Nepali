"""Synthetic comparative/ordering reasoning dataset for Nepali (Phase 3, section 3.1).

Independent of hindi/reasoning/generate_reasoning_data.py -- same task-type design (pairwise,
transitive3, chain), but its own Nepali name pool and its own natural Nepali phrasing (not a
word-for-word translation of the Hindi templates), per the assignment's requirement that text
be in the target language's own natural phrasing. See the Hindi script's module docstring for
the full rationale on why this data is synthetic and how train/test leakage is avoided.
"""

import json
import os
import random

SEED = 1337

# Common Nepali given names (Devanagari) -- distinct from the Hindi name pool.
NAMES = [
    "राम", "हरि", "गोपाल", "विष्णु", "सीता", "गीता", "कमला", "सुनिता", "विनोद", "प्रकाश",
    "सरिता", "पार्वती", "दिपक", "सन्तोष", "बिमला", "राधा", "किशोर", "नारायण", "लक्ष्मी", "सरस्वती",
    "विकास", "अनिल", "सुशीला", "कृष्ण", "गणेश", "माया", "शोभा", "रमेश", "सुरेश", "दिलिप",
    "अनिता", "सुजाता", "मिलन", "समीर", "पूर्णिमा", "इन्दिरा", "बसन्त", "हिमाल", "सागर", "आकाश",
    "स्मृति", "प्रतिमा", "बिष्णु", "टिका", "चन्द्र", "जयराम", "फूलमाया", "देवी", "मंगल", "धन",
]

OBJECT_NAMES = [
    "किताब", "कलम", "टेबुल", "कुर्सी", "झोला", "साइकल", "मोबाइल", "घडी", "जुत्ता", "छाता",
    "बोतल", "टोकरी", "पंखा", "थैलो", "आँप", "स्याउ", "केरा", "अनार", "तरबुजा", "नरिवल",
]

ATTRIBUTES = {
    "height": {"unit": "सेमी", "min": 140, "max": 200, "subjects": "people"},
    "age": {"unit": "वर्ष", "min": 5, "max": 90, "subjects": "people"},
    "price": {"unit": "रुपैयाँ", "min": 10, "max": 10000, "subjects": "objects"},
    "quantity": {"unit": "", "min": 1, "max": 100, "subjects": "objects"},
}

PAIRWISE_TEMPLATES = {
    "height": [
        "{a}को उचाइ {va} {unit} छ र {b}को उचाइ {vb} {unit} छ। कोसँग बढी उचाइ छ?",
        "{a} {va} {unit} अग्लो छ, जबकि {b} {vb} {unit} अग्लो छ। यी मध्ये कोबढी अग्लो छ?",
    ],
    "age": [
        "{a}को उमेर {va} वर्ष छ र {b}को उमेर {vb} वर्ष छ। कोबढी उमेर पुगेको छ?",
        "{a} {va} वर्षको छ, जबकि {b} {vb} वर्षको छ। यी मध्ये कोबढी सानो छ?",
    ],
    "price": [
        "{a}को मूल्य {va} रुपैयाँ छ र {b}को मूल्य {vb} रुपैयाँ छ। कोबढी महँगो छ?",
        "{a}को दाम {va} रुपैयाँ छ, जबकि {b}को दाम {vb} रुपैयाँ छ। यी मध्ये कोबढी सस्तो छ?",
    ],
    "quantity": [
        "{a}को संख्या {va} छ र {b}को संख्या {vb} छ। कसको संख्या बढी छ?",
        "{a} {va} वटा छन्, जबकि {b} {vb} वटा छन्। कसको संख्या कम छ?",
    ],
}

TRANSITIVE_TEMPLATES = {
    "height": ["{a}को उचाइ {va} {unit}, {b}को उचाइ {vb} {unit}, र {c}को उचाइ {vc} {unit} छ। यी मध्ये सबैभन्दा अग्लो को हो?"],
    "age": ["{a}को उमेर {va} वर्ष, {b}को उमेर {vb} वर्ष, र {c}को उमेर {vc} वर्ष छ। यी मध्ये सबैभन्दा जेठो को हो?"],
    "price": ["{a}को मूल्य {va} रुपैयाँ, {b}को मूल्य {vb} रुपैयाँ, र {c}को मूल्य {vc} रुपैयाँ छ। यी मध्ये सबैभन्दा महँगो कुन हो?"],
    "quantity": ["{a}को संख्या {va}, {b}को संख्या {vb}, र {c}को संख्या {vc} छ। यी मध्ये सबैभन्दा बढी संख्या कसको हो?"],
}

CHAIN_TEMPLATES_ASCENDING = [
    "{a}, {b} भन्दा ठूलो छ। {b}, {c} भन्दा ठूलो छ। {a} र {c} मध्ये को ठूलो हो?",
    "{a}, {b} भन्दा बढी छ। {b}, {c} भन्दा बढी छ। {a} र {c} मध्ये कोसँग बढी छ?",
]
CHAIN_TEMPLATES_DESCENDING = [
    "{c}, {b} भन्दा सानो छ। {b}, {a} भन्दा सानो छ। {c} र {a} मध्ये को ठूलो हो?",
    "{c}, {b} भन्दा कम छ। {b}, {a} भन्दा कम छ। {c} र {a} मध्ये कोसँग बढी छ?",
]


def gen_pairwise(names_pool, rng):
    attr = rng.choice(list(ATTRIBUTES.keys()))
    spec = ATTRIBUTES[attr]
    pool = OBJECT_NAMES if spec["subjects"] == "objects" else names_pool
    a, b = rng.sample(pool, 2)
    va, vb = rng.sample(range(spec["min"], spec["max"] + 1), 2)
    template = rng.choice(PAIRWISE_TEMPLATES[attr])
    ask_smaller = "सानो" in template or "सस्तो" in template or "कम" in template
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
        "language": "nepali",
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
