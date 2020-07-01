import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

CLASS = {0: 'toxic', 1: 'severe_toxic', 2: 'obscene', 3: 'threat', 4: 'insult', 5: 'identity_hate'}

tokenizer, model = None, None


def load_distilroberta():
    """
    Load DistilRoBERTa tokenizer and model.
    :return: None.
    """
    global tokenizer, model

    tokenizer = AutoTokenizer.from_pretrained('jpcorb20/toxic-detector-distilroberta')
    model = AutoModelForSequenceClassification.from_pretrained('jpcorb20/toxic-detector-distilroberta')
    model.eval()


def process_input(text):
    """
    Process text into inputs for model.
    :param text: comment (str).
    :return: dict and tensor for huggingface model.
    """
    global tokenizer

    inputs = tokenizer(text, return_tensors="pt")
    labels = torch.tensor([1]).unsqueeze(0)

    return inputs, labels


def map_logit(logits, threshold=0.5):
    """
    Map list of logits to class label and {0,1}.
    :param logits: logits returned by model (List[int]).
    :param threshold: threshold for 0 to 1 (float).
    :return: dict with labels as keys and {0,1} as predicted (dict).
    """
    global CLASS
    return {j: (1 if i >= threshold else 0) for j, i in zip(CLASS.values(), logits)}


def compute_labels(inputs, labels, threshold=0.71):
    """
    From huggingface model, compute logits and return associated class labels.
    :param inputs: tokenized text as dict with input_ids tensor and attention_mask tensor (dict).
    :param labels: indicate the model what to ouput (torch.Tensor).
    :param threshold: threshold between 0 and 1 (float).
    :return: dict with labels as keys and {0,1} as predicted (dict).
    """
    global model

    outputs = model(**inputs, labels=labels)
    logits = outputs[:2][1]

    return map_logit(logits.detach().numpy()[0], threshold=threshold)


def infer_labels(text):
    """
    Load model and tokenizer in module and infer labels on text.
    :param text: comment (str).
    :return: dict with labels as keys and {0,1} as predicted (dict).
    """

    if model is None or tokenizer is None:
        load_distilroberta()

    inputs, labels = process_input(text)

    return compute_labels(inputs, labels)


if __name__ == "__main__":

    test = "You are dumb, but it is ok."  # I don't support any test text here. It is solely for test purpose.

    result = infer_labels(test)

    print(result)
