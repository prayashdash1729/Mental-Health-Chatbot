# Mental Health Chatbot

[![Open In Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/prayashdash/falcon7b-finetune-mental-health)
[![Finetuned Model](https://img.shields.io/badge/Hugging%20Face-Models-orange?logo=huggingface)](https://huggingface.co/greenmantis/falcon-7b-sharded-bf16-finetuned-mental-health-conv)

## Finetuning of Falcon-7B LLM with QLoRA and bitsandbytes on Mental Health.

### Introduction:
Mental health issues are often misunderstood, leading to fear, discomfort, and negative perceptions, exacerbated by media stereotypes. Overcoming this stigma requires education, awareness, empathy, and challenging misconceptions while ensuring accessible care. Mental health is vital for overall well-being, quality of life, and effective daily functioning, and untreated issues can worsen physical health, linking mental and physical health closely.


### Dataset:
The dataset contrains 172 question-answer conversation between a patient and a healthcare provider. The dataset can be found here:

[![Hugging Face Datasets](https://img.shields.io/badge/Hugging%20Face-Datasets-orange?logo=huggingface)](https://huggingface.co/datasets/heliosbrahma/mental_health_chatbot_dataset)


### Model Finetuning:

The notebook used for finetuning can be accesed at:

[![Open In Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/prayashdash/falcon7b-finetune-mental-health)


I have used sharded Falcon-7B pre-trained model and finetuned it using bitsandbytes and QLoRA utilising quantisation and PEFT techniques. Using a shared model with Accelerate allows efficient fine-tuning of large models by dynamically managing memory across CPU and GPU, enabling the process in smaller memory environments.

The original model can be found here:

[![Hugging Face Models](https://img.shields.io/badge/Hugging%20Face-Models-orange?logo=huggingface)](https://huggingface.co/ybelkada/falcon-7b-sharded-bf16)


I chose to perform the fine-tuning on Kaggle because it does not have single-session limits, unlike Colab, which provides about three hours of GPU access per session. Kaggle gives you 30 hours of GPU compute per week and provides around 15GB of memory with two T4 GPUs. So you can train models for multiple hours even with a free account. The entire fine-tuning process took just under five hours.


### Finetunning procedure

The following `bitsandbytes` quantization config was used during training:
```python
- quant_method: bitsandbytes
- _load_in_8bit: False
- _load_in_4bit: True
- llm_int8_threshold: 6.0
- llm_int8_skip_modules: None
- llm_int8_enable_fp32_cpu_offload: False
- llm_int8_has_fp16_weight: False
- bnb_4bit_quant_type: nf4
- bnb_4bit_use_double_quant: True
- bnb_4bit_compute_dtype: bfloat16
- bnb_4bit_quant_storage: uint8
- load_in_4bit: True
- load_in_8bit: False
```

The following `qlora` quantization config was used during training:
```python
- quant_method: qlora
- lora_alpha: 32
- lora_dropout: 0.05
- lora_rank: 32
- bias: "none"
- task_type: "CAUSAL_LM"
- target_modules: [
        "query_key_value",
        "dense",
        "dense_h_to_4h",
        "dense_4h_to_h",
    ]
```
### Hyperparameters

The following hyperparameters were used during training:
```python
- learning_rate: 0.0002
- train_batch_size: 8
- eval_batch_size: 8
- training_steps: 100
- seed: 42
- gradient_accumulation_steps: 8
- total_train_batch_size: 64
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- max_seq_length: 256
```

---

## Model Inference:
PEFT fine-tuned model has been updated here:

[![Finetuned Model](https://img.shields.io/badge/Hugging%20Face-Models-orange?logo=huggingface)](https://huggingface.co/greenmantis/falcon-7b-sharded-bf16-finetuned-mental-health-conv)
<br>

The model takes less than 3 minutes to generate the response. I have compared the PEFT model response with the original model response in `falcon7b-finetune-mental-health.ipynb` notebook. It is clear that the Original model seems to halucinate and generate vague responses, whereas the PEFT model generates more coherent and relevant responses to the questions.
