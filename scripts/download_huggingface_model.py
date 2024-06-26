import os
from transformers import AutoConfig, AutoTokenizer, AutoModel

# bert_model_name = "facebook/mbart-large-50-many-to-one-mmt"
# bert_model_name = "facebook/mbart-large-50-one-to-many-mmt"
bert_model_name = "facebook/mbart-large-50-many-to-many-mmt"
# bert_model_name = "csebuetnlp/mT5_m2m_crossSum"
if os.path.isdir('pretrained_lm/'+bert_model_name):
    os.mkdir('pretrained_lm/'+bert_model_name)

tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
model_config = AutoConfig.from_pretrained(bert_model_name)
model = AutoModel.from_pretrained(bert_model_name)

bert_model_name = bert_model_name.replace('/', '-')
bert_model_name = bert_model_name.replace('_', '-')

tokenizer.save_pretrained('pretrained_lm/'+bert_model_name)
model.save_pretrained('pretrained_lm/'+bert_model_name)
model_config.save_pretrained('pretrained_lm/'+bert_model_name)
