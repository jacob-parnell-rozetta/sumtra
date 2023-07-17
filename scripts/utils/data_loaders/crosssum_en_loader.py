import os
import json
import datasets

_CITATION = ""
_DESCRIPTION = "Loads the CrossSum dataset in a many-to-one scenario (i.e. many languages to english)"

_DOCUMENT = "text"
_SUMMARY = "summary"


class CrossSumConfig(datasets.BuilderConfig):
    """BuilderConfig for SuperGLUE."""

    def __init__(self, features, supervised_keys, citation, description, data_path, **kwargs):
        super(CrossSumConfig, self).__init__(version=datasets.Version("3.0.0"), **kwargs)
        self.features = features
        self.supervised_keys = supervised_keys
        self.citation = citation
        self.description = description
        self.data_path = data_path


class CrossSum(datasets.GeneratorBasedBuilder):
    # VERSION = datasets.Version("3.0.0")
    O2M_PATH = "data/raw_data/CrossSum_en/one_to_many/"
    # Note: when specifying a source language, use the syntax of the LHS language described below
    BUILDER_CONFIGS = [
        CrossSumConfig(name='english-spanish', features=[_DOCUMENT, _SUMMARY], supervised_keys=(_DOCUMENT, _SUMMARY), citation=_CITATION, description="English to Spanish split", data_path=O2M_PATH),
        CrossSumConfig(name='english-french', features=[_DOCUMENT, _SUMMARY], supervised_keys=(_DOCUMENT, _SUMMARY), citation=_CITATION, description="English to French split", data_path=O2M_PATH),
        CrossSumConfig(name='english-arabic', features=[_DOCUMENT, _SUMMARY], supervised_keys=(_DOCUMENT, _SUMMARY), citation=_CITATION, description="English to Arabic split", data_path=O2M_PATH),
        CrossSumConfig(name='english-ukrainian', features=[_DOCUMENT, _SUMMARY], supervised_keys=(_DOCUMENT, _SUMMARY), citation=_CITATION, description="English to Ukrainian split", data_path=O2M_PATH),
        CrossSumConfig(name='english-azerbaijani', features=[_DOCUMENT, _SUMMARY], supervised_keys=(_DOCUMENT, _SUMMARY), citation=_CITATION, description="English to Azerbaijani split", data_path=O2M_PATH),
        CrossSumConfig(name='english-bengali', features=[_DOCUMENT, _SUMMARY], supervised_keys=(_DOCUMENT, _SUMMARY), citation=_CITATION, description="English to Bengali split", data_path=O2M_PATH),
        CrossSumConfig(name='english-english', features=[_DOCUMENT, _SUMMARY], supervised_keys=(_DOCUMENT, _SUMMARY), citation=_CITATION, description="English to English split", data_path=O2M_PATH),
        CrossSumConfig(name='english-hindi', features=[_DOCUMENT, _SUMMARY], supervised_keys=(_DOCUMENT, _SUMMARY), citation=_CITATION, description="English to Hindi split", data_path=O2M_PATH),
        CrossSumConfig(name='english-thai', features=[_DOCUMENT, _SUMMARY], supervised_keys=(_DOCUMENT, _SUMMARY), citation=_CITATION, description="English to Thai split", data_path=O2M_PATH),
    ]
    DEFAULT_CONFIG_NAME = "english-english"

    def _info(self):
        # All features are the same no matter the language
        features = {feature: datasets.Value("string") for feature in self.config.features}
        return datasets.DatasetInfo(
            description=_DESCRIPTION + ' - ' + self.config.description,
            features=datasets.Features(features),
            supervised_keys=self.config.supervised_keys,
        )

    def _split_generators(self, dl_manager):
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"path": os.path.join(self.config.data_path, self.config.name) + "_train.jsonl"}
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={"path": os.path.join(self.config.data_path, self.config.name) + "_val.jsonl"}
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"path": os.path.join(self.config.data_path, self.config.name) + "_test.jsonl"}
            ),
        ]

    def _generate_examples(self, path=None):
        """Yields examples."""
        with open(path, encoding="utf-8") as f:
            for id_, line in enumerate(f):
                d = json.loads(line)
                input_article = d['text']
                summary = d['summary']
                yield id_, {
                    "text": input_article,
                    "summary": summary,
                }
