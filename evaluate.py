from transformers import BertTokenizer, BertForQuestionAnswering
from transformers.data.metrics.squad_metrics import (
    compute_predictions_logits,
    squad_evaluate,
)
import json
import torch
from tqdm import tqdm 
# Load the pre-trained BERT model and tokenizer
model_name = 'bert-base-uncased'
model = BertForQuestionAnswering.from_pretrained(model_name)
tokenizer = BertTokenizer.from_pretrained(model_name)

# Load the SQuAD 2.0 dev set
with open('squad/dev-v2.0.json', 'r') as f:
    squad_data = json.load(f)

# Prepare the data for evaluation
eval_examples = squad_data['data']

# Initialize lists to store the predictions and references
all_predictions = []
all_references = []

# Evaluate each example in the dev set
for example in tqdm(eval_examples):
    for paragraph in example['paragraphs']:
        context = paragraph['context']
        for qa in paragraph['qas']:
            question = qa['question']
            qas_id = qa['id']
            
            # Tokenize the input
            inputs = tokenizer.encode_plus(question, context, add_special_tokens=True, return_tensors='pt')
            input_ids = inputs['input_ids'].tolist()[0]
            token_type_ids = inputs['token_type_ids'].tolist()[0]
            
            # Generate the predictions
            outputs = model(**inputs)
            start_logits, end_logits = outputs.start_logits, outputs.end_logits
            start_index = torch.argmax(start_logits)
            end_index = torch.argmax(end_logits)
            answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[start_index:end_index+1]))
            
            # Store the predicted answer and reference
            all_predictions.append({'id': qas_id, 'prediction_text': answer})
            all_references.append({'id': qas_id, 'answers': qa['answers']})

# Calculate the F1 score and EM score
predictions = compute_predictions_logits(
    eval_examples,
    all_predictions,
    no_answer_probability=None,
    na_prob_thresh=1.0
)
results = squad_evaluate(all_references, predictions)

# Print the F1 score and EM score
print(f"F1 score: {results['f1']}")
print(f"EM score: {results['exact_match']}")
