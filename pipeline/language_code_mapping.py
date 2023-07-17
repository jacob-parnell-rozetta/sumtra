""" https://github.com/csebuetnlp/xl-sum/tree/master/multilingual_rouge_scoring for mROUGE support"""
LANGUAGE_CODE_MAPPING = {
    'ar': {'mbart_code': 'ar_AR', 'name': 'arabic', 'mrouge': 1},
    'cs': {'mbart_code': 'cs_CZ', 'name': 'czech', 'mrouge': 0},
    'de': {'mbart_code': 'de_DE', 'name': 'german', 'mrouge': 1},
    'en': {'mbart_code': 'en_XX', 'name': 'english', 'mrouge': 1},
    'es': {'mbart_code': 'es_XX', 'name': 'spanish', 'mrouge': 1},
    'et': {'mbart_code': 'et_EE', 'name': 'estonian', 'mrouge': 0},
    'fi': {'mbart_code': 'fi_FI', 'name': 'finnish', 'mrouge': 0},
    'fr': {'mbart_code': 'fr_XX', 'name': 'french', 'mrouge': 1},
    'gu': {'mbart_code': 'gu_IN', 'name': 'gujarati', 'mrouge': 0},
    'hi': {'mbart_code': 'hi_IN', 'name': 'hindi', 'mrouge': 1},
    'it': {'mbart_code': 'it_IT', 'name': 'italian', 'mrouge': 1},
    'ja': {'mbart_code': 'ja_XX', 'name': 'japanese', 'mrouge': 1},
    'kk': {'mbart_code': 'kk_KZ', 'name': 'kazakh', 'mrouge': 0},
    'ko': {'mbart_code': 'ko_KR', 'name': 'korean', 'mrouge': 0},
    'lt': {'mbart_code': 'lt_LT', 'name': 'lithuanian', 'mrouge': 0},
    'lv': {'mbart_code': 'lv_LV', 'name': 'latvian', 'mrouge': 0},
    'my': {'mbart_code': 'my_MM',  'name': 'burmese', 'mrouge': 1},
    'ne': {'mbart_code': 'ne_NP', 'name': 'nepali', 'mrouge': 0},
    'nl': {'mbart_code': 'nl_XX', 'name': 'dutch', 'mrouge': 1},
    'ro': {'mbart_code': 'ro_RO', 'name': 'romanian', 'mrouge': 1},
    'ru': {'mbart_code': 'ru_RU', 'name': 'russian', 'mrouge': 1},
    'si': {'mbart_code': 'si_LK', 'name': 'sinhala', 'mrouge': 0},
    'tr': {'mbart_code': 'tr_TR', 'name': 'turkish', 'mrouge': 1},
    'vi': {'mbart_code': 'vi_VN', 'name': 'vietnamese', 'mrouge': 0},
    'zh': {'mbart_code': 'zh_CN', 'name': 'chinese', 'mrouge': 1},
    'zh_s': {'mbart_code': 'zh_CN', 'name': 'chinese_simplified', 'mrouge': 1},
    'zh_t': {'mbart_code': None, 'alt_code': 'zh_CN', 'name': 'chinese_traditional', 'mrouge': 1},
    'af': {'mbart_code': 'af_ZA', 'name': 'afrikaans', 'mrouge': 0},
    'az': {'mbart_code': 'az_AZ', 'name': 'azerbaijani', 'mrouge': 0},
    'bn': {'mbart_code': 'bn_IN', 'name': 'bengali', 'mrouge': 1},
    'fa': {'mbart_code': 'fa_IR', 'name': 'persian', 'mrouge': 0},
    'he': {'mbart_code': 'he_IL', 'name': 'hebrew', 'mrouge': 0},
    'hr': {'mbart_code': 'hr_HR', 'name': 'croatian', 'mrouge': 0},
    'id': {'mbart_code': 'id_ID', 'name': 'indonesian', 'mrouge': 0},
    'ka': {'mbart_code': 'ka_GE', 'name': 'georgian', 'mrouge': 0},
    'km': {'mbart_code': 'km_KH', 'name': 'cambodian', 'mrouge': 0},
    'mk': {'mbart_code': 'mk_MK', 'name': 'macedonian', 'mrouge': 0},
    'ml': {'mbart_code': 'ml_IN', 'name': 'malayalam', 'mrouge': 0},
    'mn': {'mbart_code': 'mn_MN', 'name': 'mongolian', 'mrouge': 0},
    'mr': {'mbart_code': 'mr_IN', 'name': 'marathi', 'mrouge': 0},
    'pl': {'mbart_code': 'pl_PL', 'name': 'polish', 'mrouge': 0},
    'ps': {'mbart_code': 'ps_AF', 'name': 'pushto', 'mrouge': 0},
    'pt': {'mbart_code': 'pt_XX', 'name': 'portuguese', 'mrouge': 1},
    'sv': {'mbart_code': 'sv_SE', 'name': 'swedish', 'mrouge': 0},
    'sw': {'mbart_code': 'sw_KE', 'name': 'swahili', 'mrouge': 0},
    'ta': {'mbart_code': 'ta_IN', 'name': 'tamil', 'mrouge': 0},
    'te': {'mbart_code': 'te_IN', 'name': 'telugu', 'mrouge': 0},
    'th': {'mbart_code': 'th_TH', 'name': 'thai', 'mrouge': 1},
    'tl': {'mbart_code': 'tl_XX', 'name': 'tagalog', 'mrouge': 0},
    'uk': {'mbart_code': 'uk_UA', 'name': 'ukrainian', 'mrouge': 0},
    'ur': {'mbart_code': 'ur_PK', 'name': 'urdu', 'mrouge': 0},
    'xh': {'mbart_code': 'xh_ZA', 'name': 'xhosa', 'mrouge': 0},
    'gl': {'mbart_code': 'gl_ES', 'name': 'galician', 'mrouge': 0},
    'sl': {'mbart_code': 'sl_SI', 'name': 'slovenian', 'mrouge': 0},
    'bg': {'mbart_code': None, 'alt_code': 'ru_RU', 'name': 'bulgarian', 'mrouge': 1},  # bulgarian -> russian
    'el': {'mbart_code': None, 'alt_code': 'mk_MK', 'name': 'greek', 'mrouge': 1},  # greek -> macedonian
    'am': {'mbart_code': None, 'alt_code': 'ar_AR', 'name': 'amharic', 'mrouge': 1},  # amharic -> arabic (arabic has mrouge)
    'yo': {'mbart_code': None, 'alt_code': 'sw_KE', 'name': 'yoruba', 'mrouge': 0},  # yoruba -> swahili
    'ha': {'mbart_code': None, 'alt_code': 'ar_AR', 'name': 'hausa', 'mrouge': 1},  # hausa -> arabic (arabic has mrouge)
    'ig': {'mbart_code': None, 'alt_code': 'sw_KE', 'name': 'igbo', 'mrouge': 0},  # igbo -> swahili
    'rn': {'mbart_code': None, 'alt_code': 'sw_KE', 'name': 'kirundi', 'mrouge': 0},  # kirundi -> swahili
    'ky': {'mbart_code': None, 'alt_code': 'tr_TR', 'name': 'kyrgyz', 'mrouge': 1},  # kyrgyz -> turkish (turkish has mrouge)
    'om': {'mbart_code': None, 'alt_code': 'ar_AR', 'name': 'oromo', 'mrouge': 1},  # oromo -> arabic (arabic has mrouge)
    'pa': {'mbart_code': None, 'alt_code': 'hi_IN', 'name': 'punjabi', 'mrouge': 1},  # punjabi -> hindi (hindi has mrouge)
    'gd': {'mbart_code': None, 'alt_code': 'en_XX', 'name': 'scottish_gaelic', 'mrouge': 0},  # scottish_gaelic -> english
    'sr_c': {'mbart_code': None, 'alt_code': 'ru_RU', 'name': 'serbian_cyrillic', 'mrouge': 1},  # serbian_cyrillic -> russian (russian has mrouge)
    'sr_l': {'mbart_code': None, 'alt_code': 'sl_SI', 'name': 'serbian_latin', 'mrouge': 0},  # serbian_latin -> slovenian
    'so': {'mbart_code': None, 'alt_code': 'ar_AR', 'name': 'somali', 'mrouge': 1},  # somali -> arabic (arabic has mrouge)
    'ti': {'mbart_code': None, 'alt_code': 'ar_AR', 'name': 'tigrinya', 'mrouge': 1},  # tigrinya -> arabic (arabic has mrouge)
    'uz': {'mbart_code': None, 'alt_code': 'tr_TR', 'name': 'uzbek', 'mrouge': 1},  # uzbek -> turkish (turkish has mrouge)
    'cy': {'mbart_code': None, 'alt_code': 'en_XX', 'name': 'welsh', 'mrouge': 0},  # welsh -> english
}