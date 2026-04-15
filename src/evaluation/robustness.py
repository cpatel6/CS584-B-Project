import random
import copy
import hashlib

class PerturbationTesting:
    """
    Phase 6: Robustness testing via perturbation
    """
    def __init__(self, seed=42):
        self.seed = seed

    def _stable_text_seed(self, text: str) -> int:
        digest = hashlib.sha256(str(text).encode("utf-8")).hexdigest()
        return self.seed + int(digest[:16], 16)
        
    def shuffle_word_order(self, text: str) -> str:
        random.seed(self._stable_text_seed(text))  # Deterministic perturbation
        words = text.split()
        random.shuffle(words)
        return " ".join(words)
        
    def inject_noise(self, text: str, noise_prob=0.1) -> str:
        random.seed(self._stable_text_seed(text))
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

    def scale_data_regime(self, df, fraction=1.0):
        fraction = max(0.0, min(float(fraction), 1.0))
        if fraction == 1.0:
            return df.copy()
        sample_size = max(1, int(len(df) * fraction))
        return df.sample(n=sample_size, random_state=self.seed).reset_index(drop=True)
