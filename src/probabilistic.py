import math
import numpy as np
import torch
from scipy.stats import norm
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from transformers import AutoTokenizer, AutoModelForSequenceClassification

HF_MODEL     = "entfane/gpt2_constitutional_classifier"
THRESHOLD    = 0.2
N_COMPONENTS = 2     
PCA_DIM      = 8     
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


def verify_gmm(embeddings, model, threshold, n_components, pca_dim):
    w_full = model.score.weight.squeeze().detach().cpu().numpy()
    b      = model.score.bias.detach().cpu().numpy().item() if model.score.bias is not None else 0.0

    pca = PCA(n_components=pca_dim)
    emb_reduced = pca.fit_transform(embeddings) 
    w_reduced   = pca.components_ @ w_full              

    gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=0)
    gmm.fit(emb_reduced)

    threshold_logit = math.log(threshold / (1 - threshold))

    total_prob = 0.0
    for k in range(n_components):
        pi_k    = gmm.weights_[k]
        mu_k    = gmm.means_[k]
        sigma_k = gmm.covariances_[k]

        mean_y = w_reduced @ mu_k + b
        var_y  = w_reduced.T @ sigma_k @ w_reduced
        std_y  = math.sqrt(max(var_y, 1e-12))
        z      = (threshold_logit - mean_y) / std_y
        prob_k = norm.sf(z)

        total_prob += pi_k * prob_k

    return total_prob


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForSequenceClassification.from_pretrained(HF_MODEL)
    model.config.pad_token_id = tokenizer.eos_token_id
    model.eval().to(DEVICE)

    embeddings = extract_embeddings(SENTENCES, model, tokenizer)

    prob = verify_gmm(embeddings, model, THRESHOLD, N_COMPONENTS, PCA_DIM)

    print(f"\n  P(score > {THRESHOLD}) = {prob:.4f}")
    if prob >= 0.95:
        print(f"  ✓ With {prob*100:.1f}% probability, the model scores > {THRESHOLD} on this distribution.")
    else:
        print(f"  ✗ Only {prob*100:.1f}% probability above threshold — not sufficient.")