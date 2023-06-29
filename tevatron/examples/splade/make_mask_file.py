from transformers import AutoTokenizer
import json
import random
import argparse



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tokenizer_name', required=True)
    parser.add_argument('--save_path', required=True)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, cache_dir='cache')
    # tokenizer = AutoTokenizer.from_pretrained("distilbert-base-multilingual-cased")
    vocab_dict = tokenizer.get_vocab()

    vocab_dict = {v: k for k, v in vocab_dict.items()}

    rand_token_ids = random.sample(list(range(len(vocab_dict))), 768)

    # write lines to file
    with open(args.save_path, "w") as f:
       for id in rand_token_ids:
              f.write(vocab_dict[id] + "\n")


if __name__ == "__main__":
    main()