import os
import re
import json
from json import JSONDecodeError
from glob import glob
from tqdm import tqdm
from typing import List, Union, Dict, Any
from datasets import load_dataset
from datasets.arrow_dataset import Batch


def clean(s):
    s = s.replace("\n", "")
    s = s.replace("\t", "")
    s = re.sub(", +}", ",}", s)
    s = re.sub(", +]", ",]", s)
    s = s.replace(",}", "}")
    s = s.replace(",]", "]")
    s = s.replace("'", '"')
    s = s.replace(".,", ",")
    return s


def prepare_dialect_dataset(filenames: List[str]):
    results = []
    for filename in tqdm(filenames):

        with open(filename, encoding="utf-8-sig") as f:
            s = f.read()
            try:
                data = json.loads(s)
            except JSONDecodeError:
                data = json.loads(clean(s))

        utterance = data["utterance"]
        for u in utterance:
            if u["standard_form"] != u["dialect_form"]:
                dialect_idx = []
                for e in u["eojeolList"]:
                    if e["isDialect"]:
                        dialect_idx.append(e["id"])
                sample = {
                    "id": u["id"],
                    "do": filename.split("/")[1],
                    "standard": u["standard_form"],
                    "dialect": u["dialect_form"],
                    "dialect_idx": dialect_idx,
                }
                results.append(sample)

    return results


NAMES = [
    "NAME", "NAEM", "anem", "anme", "mane", "naem", "nam", "nmae",
    "이름", "고자영", "최미영",
]  # fmt: skip
PAT_LIST = [
    "&NAEM4&",
    "&NAME&",
    "&NAME18&",
    "&adderess2&",
    "&adderss11&",
    "&address&",
    "&address1&",
    "&address10&",
    "&address11&",
    "&address12&",
    "&address13&",
    "&address14&",
    "&address15&",
    "&address16&",
    "&address17&",
    "&address18&",
    "&address19&",
    "&address2&",
    "&address20&",
    "&address21&",
    "&address22&",
    "&address23&",
    "&address3&",
    "&address4&",
    "&address5&",
    "&address6&",
    "&address7&",
    "&address8&",
    "&address9&",
    "&addressa&",
    "&adress&",
    "&anem6&",
    "&anme1&",
    "&anme5&",
    "&anme6&",
    "&mane1&",
    "&mane4&",
    "&mane5&",
    "&naem1&",
    "&naem16&",
    "&naem2&",
    "&naem6&",
    "&naem7&",
    "&naem9&",
    "&nam13&",
    "&nam16e&",
    "&nam1e&",
    "&nam3&",
    "&nam4&",
    "&nam51&",
    "&nam7&",
    "&namE5&",
    "&name&",
    "&name0&",
    "&name1&",
    "&name10&",
    "&name11&",
    "&name12&",
    "&name13&",
    "&name14&",
    "&name145&",
    "&name15&",
    "&name16&",
    "&name17&",
    "&name18&",
    "&name19&",
    "&name2&",
    "&name20&",
    "&name21&",
    "&name22&",
    "&name23&",
    "&name24&",
    "&name25&",
    "&name26&",
    "&name27&",
    "&name28&",
    "&name29&",
    "&name3&",
    "&name30&",
    "&name31&",
    "&name32&",
    "&name33&",
    "&name34&",
    "&name35&",
    "&name36&",
    "&name37&",
    "&name38&",
    "&name39&",
    "&name4&",
    "&name40&",
    "&name41&",
    "&name42&",
    "&name43&",
    "&name44&",
    "&name45&",
    "&name46&",
    "&name47&",
    "&name48&",
    "&name49&",
    "&name5&",
    "&name50&",
    "&name51&",
    "&name52&",
    "&name54&",
    "&name55&",
    "&name56&",
    "&name57&",
    "&name59&",
    "&name6&",
    "&name60&",
    "&name61&",
    "&name62&",
    "&name63&",
    "&name64&",
    "&name65&",
    "&name67&",
    "&name68&",
    "&name7&",
    "&name8&",
    "&name9&",
    "&names5&",
    "&nmae2&",
    "&nmae3&",
    "&company2&",
    "&company3&",
    "&company_name1&",
    "&company_name2&",
    "&가자&",
    "&고자영2&",
    "&상호명1&",
    "&상호명2&",
    "&서연림1&",
    "&선옥언니&",
    "&월령&",
    "&유튜브&",
    "&이름1&",
    "&이름2&",
    "&이름4&",
    "&이름5&",
    "&인가&",
    "&좌미영2&",
    "&한림농협&",
]

PAT_MAP = {}
for p in PAT_LIST:
    PAT_MAP[p] = "[OTHER]"
    if "add" in p:
        PAT_MAP[p] = "[ADDRESS]"
    for n in NAMES:
        if n in p:
            PAT_MAP[p] = "[NAME]"


PATTERN1 = re.compile("(\(\(\)\))|(\{\w+\})|[#-]")
PATTERN2 = re.compile("(\(\(\w+\)\))")
PATTERN3 = re.compile("(&\w+&)")
PATTERN4 = re.compile("\((\w+)\)/\((\w+)\)")


def preprocess(examples: Batch) -> Union[Dict, Any]:
    ids = examples["id"]
    dos = examples["do"]
    standard_texts = examples["standard"]
    dialect_texts = examples["dialect"]
    dialect_idxs = examples["dialect_idx"]

    new_examples = {
        "id": [],
        "do": [],
        "standard": [],
        "dialect": [],
        "dialect_idx": [],
    }

    iterator = zip(ids, dos, standard_texts, dialect_texts, dialect_idxs)
    for _id, do, standard_text, dialect_text, dialect_idx in iterator:
        # remove (()), {\w+} patterns
        standard_text = re.sub(PATTERN1, "", standard_text).replace("  ", " ")
        dialect_text = re.sub(PATTERN1, "", dialect_text).replace("  ", " ")
        # remove \n, \t patterns
        standard_text = re.sub("[\t\n]", " ", standard_text).replace("  ", " ")
        dialect_text = re.sub("[\t\n]", " ", dialect_text).replace("  ", " ")
        # remove sample which has ((\w+)) patterns
        if PATTERN2.findall(standard_text) + PATTERN2.findall(dialect_text):
            continue

        # $\w+$ pattern mapping
        def _replace_pat(m):
            return PAT_MAP.get(m.group(0), "[OTHER]")

        standard_text = PATTERN3.sub(_replace_pat, standard_text)
        dialect_text = PATTERN3.sub(_replace_pat, dialect_text)

        # (\w+)/(\w+)
        standard_text = re.sub(PATTERN4, r"\2", standard_text)
        dialect_text = re.sub(PATTERN4, r"\1", dialect_text)

        new_examples["id"].append(_id)
        new_examples["do"].append(do)
        new_examples["standard"].append(standard_text)
        new_examples["dialect"].append(dialect_text)
        new_examples["dialect_idx"].append(dialect_idx)

    return new_examples


def prepare_for_style_classification(examples: Batch) -> Union[Dict, Any]:
    ids = examples["id"]
    dos = examples["do"]
    standard_texts = examples["standard"]
    dialect_texts = examples["dialect"]

    new_examples = {"id": [], "do": [], "text": [], "label": []}

    iterator = zip(ids, dos, standard_texts, dialect_texts)
    for _id, do, standard_text, dialect_text in iterator:
        new_examples["id"].extend([_id, _id])
        new_examples["do"].extend([do, do])
        new_examples["text"].extend([standard_text, dialect_text])
        new_examples["label"].extend([0, 1])

    return new_examples


def prepare_for_style_transfer(examples: Batch) -> Union[Dict, Any]:
    ids = examples["id"]
    dos = examples["do"]
    standard_texts = examples["standard"]
    dialect_texts = examples["dialect"]

    new_examples = {
        "id": [],
        "source": [],
        "target": [],
        "src_lang": [],
        "tgt_lang": [],
    }

    iterator = zip(ids, dos, standard_texts, dialect_texts)
    for _id, do, standard_text, dialect_text in iterator:
        new_examples["id"].extend([_id, _id])
        new_examples["source"].extend([standard_text, dialect_text])
        new_examples["target"].extend([dialect_text, standard_text])
        new_examples["src_lang"].extend(["standard", do])
        new_examples["tgt_lang"].extend([do, "standard"])

    return new_examples


if __name__ == "__main__":
    if not os.path.isfile("data/train_dialect.json"):
        train_files = glob("data/*/train/*.json")
        train_samples = prepare_dialect_dataset(train_files)
        json.dump({"data": train_samples}, open("data/train_dialect.json", "w"))

    if not os.path.isfile("data/valid_dialect.json"):
        valid_files = glob("data/*/valid/*.json")
        valid_samples = prepare_dialect_dataset(valid_files)
        json.dump({"data": valid_samples}, open("data/valid_dialect.json", "w"))

    data_files = {
        "train": "data/train_dialect.json",
        "valid": "data/valid_dialect.json",
    }
    dialect = load_dataset("json", data_files=data_files, field="data")

    # Data preprocessing
    dialect_dataset = dialect.map(function=preprocess, batched=True, batch_size=1000)
    dialect_dataset_for_sc = dialect_dataset.map(
        function=prepare_for_style_classification,
        batched=True,
        batch_size=1000,
        remove_columns=dialect_dataset.column_names["train"],
    )
    dialect_dataset_for_st = dialect_dataset.map(
        function=prepare_for_style_transfer,
        batched=True,
        batch_size=1000,
        remove_columns=dialect_dataset.column_names["train"],
    )

    dialect_dataset_for_sc.save_to_disk("data/style_classification")
    dialect_dataset_for_st.save_to_disk("data/style_transfer")
