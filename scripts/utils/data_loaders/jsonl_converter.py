""" Convert raw data files to .jsonl objects - this script will likely be used for WikiLingua """
import os
import json


class JSONLConverter:
    def __init__(self, source_language, target_language, src_lang, tgt_lang, data_split, input_path, output_path):
        self.source_language = source_language
        self.target_language = target_language
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.data_split = data_split
        self.input_path = input_path
        self.output_path = output_path

    def write_to_file(self, output_dict):
        output_filename = self.output_path + f"/{self.source_language}-{self.target_language}_{self.data_split}.jsonl"
        if not os.path.exists(output_filename):
            with open(output_filename, "w") as f:
                json.dump(output_dict, f)
                f.write('\n')
        else:  # append
            with open(output_filename, "a") as f:
                json.dump(output_dict, f)
                f.write('\n')

    def process_data(self):
        source_file = f'{self.input_path}/{self.target_language}/{self.data_split}.src.{self.src_lang}'
        target_file = f'{self.input_path}/{self.target_language}/{self.data_split}.tgt.{self.tgt_lang}'
        source = open(source_file)
        target = open(target_file)
        for num, (line1, line2) in enumerate(zip(source, target)):
            if num % 1000 == 0:
                print(num)
            output_dict = {
                'id': num,
                'text': line1,
                'summary': line2
            }
            self.write_to_file(output_dict)


if __name__ == '__main__':
    target_mapping = {"arabic": "ar", "chinese": "zh", "czech": "cs", "dutch": "nl", "english": "en",
                      "french": "fr", "german": "de", "hindi": "hi", "indonesian": "id", "italian": "it",
                      "japanese": "ja", "korean": "ko", "portuguese": "pt", "russian": "ru", "spanish": "es",
                      "thai": "th", "turkish": "tr", "vietnamese": "vi"}
    source_language = "english"
    # target_language = "english"
    target_languages = ["arabic", "chinese", "czech", "dutch", "english", "french", "german", "hindi",
                        "indonesian", "italian", "japanese", "korean", "portuguese", "russian", "spanish",
                        "thai", "turkish", "vietnamese"]
    for target_language in target_languages:
        src_lang = "en"
        tgt_lang = target_mapping[target_language]
        data_splits = ["train", "test", "val"]
        input_path = "data/raw_data/WikiLingua_data_splits"
        output_path = "data/processed_data/wikilingua"
        for data_split in data_splits:
            print(f"Running: {data_split}")
            jsonl_converter = JSONLConverter(source_language, target_language, src_lang, tgt_lang, data_split, input_path, output_path)

            # Run
            jsonl_converter.process_data()
