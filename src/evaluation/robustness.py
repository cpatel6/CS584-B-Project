import random
import copy

class PerturbationTesting:
    """
    Phase 6: Robustness testing via perturbation
    """
    def __init__(self, seed=42):
        self.seed = seed
        
    def shuffle_word_order(self, text: str) -> str:
        random.seed(self.seed + hash(text) % 10000) # Deterministic perturbation
        words = text.split()
        random.shuffle(words)
        return " ".join(words)
        
    def inject_noise(self, text: str, noise_prob=0.1) -> str:
        random.seed(self.seed + hash(text) % 10000)
        words = list(text)
        for i in range(len(words)):
            if words[i].isalpha() and random.random() < noise_prob:
                # noise injection (typos)
                words[i] = random.choice('abcdefghijklmnopqrstuvwxyz')
        return "".join(words)
        
    def create_perturbed_dataset(self, df, perturbation_type="noise", param=0.1):
        df_perturbed = df.copy()
        if perturbation_type == "noise":
            df_perturbed['text'] = df_perturbed['text'].apply(lambda x: self.inject_noise(x, param))
        elif perturbation_type == "shuffle":
            df_perturbed['text'] = df_perturbed['text'].apply(self.shuffle_word_order)
        return df_perturbed
