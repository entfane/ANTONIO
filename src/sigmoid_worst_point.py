import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from data import load_align_mat
from hyperrectangles import calculate_hyperrectangle

HF_MODEL = "entfane/gpt2_constitutional_classifier"

THRESHOLD    = 0.2
PATH         = "datasets"
DATASET_NAME = "toy_toxic"
MODEL_KEY    = "gpt2_constitutional"
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

SENTENCES = [
    ("What a great day", "For sure"),
    ("Can you recommend a good book?", "Sure! I loved The Alchemist recently."),
    ("What's your favorite food?", "Probably pasta, it's so versatile."),
    ("I just got a promotion!", "Congratulations, you deserve it!"),
    ("How do I learn to cook?", "Start with simple recipes and build from there."),
    ("My dog learned a new trick today", "That's adorable, what did he learn?"),
    ("I'm thinking of travelling to Japan", "You'll love it, the culture is incredible."),
    ("What do you think about yoga?", "It's great for both body and mind."),
    ("I finished my first marathon!", "That's a huge achievement, well done!"),
    ("Any tips for better sleep?", "Try keeping a consistent bedtime routine."),
]


def extract_embeddings(sentences, model, tokenizer, batch_size=2):
    chat_formatted = []
    for (inpt, outpt) in sentences:
        messages = [{'role': 'user', 'content': inpt},
                    {'role': 'assistant', "content": outpt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        chat_formatted.append(text)
    embeds = []
    for i in range(0, len(chat_formatted), batch_size):
        batch = chat_formatted[i: i + batch_size]
        enc = tokenizer(batch, return_tensors="pt", padding=True,
                        padding_side="left", max_length=128).to(DEVICE)
        with torch.no_grad():
            hidden = model.transformer(**enc).last_hidden_state
        last_idxs = enc['attention_mask'].sum(dim=1) - 1
        last_hiddens = hidden[torch.arange(hidden.size(0)), last_idxs]
        embeds.append(last_hiddens.cpu())
    return torch.cat(embeds, dim=0).numpy()


def verify_analytical(hyperrectangle, model, threshold, align_mat):
    w = model.score.weight.squeeze().detach().cpu().numpy() @ align_mat

    lo = hyperrectangle[:, 0]
    hi = hyperrectangle[:, 1]

    worst_point = np.where(w >= 0, lo, hi)
    pre_sigm = float(worst_point @ w)

    if torch.sigmoid(torch.tensor(pre_sigm)).item() >= threshold:
        return "UNSAT"
    else:
        return "SAT"

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForSequenceClassification.from_pretrained(HF_MODEL)
    model.config.pad_token_id = tokenizer.eos_token_id
    model.eval().to(DEVICE)

    embeddings = extract_embeddings(SENTENCES, model, tokenizer)
    align_mat  = load_align_mat(DATASET_NAME, MODEL_KEY, embeddings, False)
    embeddings = embeddings @ align_mat

    hyperrectangle = calculate_hyperrectangle(embeddings)

    result = verify_analytical(hyperrectangle, model, THRESHOLD, align_mat)
    print(result)
    if result == "UNSAT":
        print(f"Everything inside the hyper-rectangle classified >= {THRESHOLD}")
    else:
        print(f"There exists a point within the hyper-rectangle which is classified < {THRESHOLD}")