import os
import json
import datasets

_CITATION = ""
_DESCRIPTION = "Loads the WikiLingua dataset in a many-to-one scenario (i.e. many languages to english)"

_DOCUMENT = "text"
_SUMMARY = "summary"


class WikiLinguaConfig(datasets.BuilderConfig):
    """BuilderConfig for SuperGLUE."""

    def __init__(self, features, supervised_keys, citation, description, data_path, **kwargs):
        super(WikiLinguaConfig, self).__init__(version=datasets.Version("3.0.0"), **kwargs)
        self.features = features
        self.supervised_keys = supervised_keys
        self.citation = citation
        self.description = description
        self.data_path = data_path


class WikiLingua(datasets.GeneratorBasedBuilder):
    # VERSION = datasets.Version("3.0.0")
    PATH_TO_FILES = "data/processed_data/wikilingua/"
    # Note: when specifying a source language, use the syntax of the LHS language described below
    BUILDER_CONFIGS = [
        WikiLinguaConfig(name='english-english', features=[_DOCUMENT, _SUMMARY], supervised_keys=(_DOCUMENT, _SUMMARY), citation=_CITATION, description="English to English split", data_path=PATH_TO_FILES),
        WikiLinguaConfig(name='english-russian', features=[_DOCUMENT, _SUMMARY], supervised_keys=(_DOCUMENT, _SUMMARY), citation=_CITATION, description="English to Russian split", data_path=PATH_TO_FILES),
        WikiLinguaConfig(name='english-chinese', features=[_DOCUMENT, _SUMMARY], supervised_keys=(_DOCUMENT, _SUMMARY), citation=_CITATION, description="English to Chinese split", data_path=PATH_TO_FILES),
        WikiLinguaConfig(name='english-arabic', features=[_DOCUMENT, _SUMMARY], supervised_keys=(_DOCUMENT, _SUMMARY), citation=_CITATION, description="English to Arabic split", data_path=PATH_TO_FILES),
        WikiLinguaConfig(name='english-turkish', features=[_DOCUMENT, _SUMMARY], supervised_keys=(_DOCUMENT, _SUMMARY), citation=_CITATION, description="English to Turkish split", data_path=PATH_TO_FILES),
        WikiLinguaConfig(name='english-thai', features=[_DOCUMENT, _SUMMARY], supervised_keys=(_DOCUMENT, _SUMMARY), citation=_CITATION, description="English to Thai split", data_path=PATH_TO_FILES),
        WikiLinguaConfig(name='english-indonesian', features=[_DOCUMENT, _SUMMARY], supervised_keys=(_DOCUMENT, _SUMMARY), citation=_CITATION, description="English to Indonesian split", data_path=PATH_TO_FILES),
        WikiLinguaConfig(name='english-spanish', features=[_DOCUMENT, _SUMMARY], supervised_keys=(_DOCUMENT, _SUMMARY), citation=_CITATION, description="English to Spanish split", data_path=PATH_TO_FILES),
        WikiLinguaConfig(name='english-hindi', features=[_DOCUMENT, _SUMMARY], supervised_keys=(_DOCUMENT, _SUMMARY), citation=_CITATION, description="English to Hindi split", data_path=PATH_TO_FILES),
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
