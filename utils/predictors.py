import numpy as np
import utils.utils as utils
from sentence_transformers import SentenceTransformer
import pandas as pd

class Predictor:
    """
    The predictor class is soley responsible for handling different prediction strategies. 
    It does not format the prompt for each task and so 
    1) task-specific serialization of each sample 
    2) the task-specific instructions
    must be provided.
    """
    def __init__(self, dataset, strategy="ZeroShot", hippa=False, target=None, training_set=None):
        """
        :param model: The LLM to use for predictions.
        :param strategy: Prediction strategy ("FewShot", "CoT", "ZeroShot", "SelfConsistency").
        :param debug: Whether to enable debug mode.
        :param training_set: The training dataset for few-shot prompting.
        """
        self.dataset = dataset
        self.strategy = strategy
        self.target = target
        self.hippa = hippa
        self.training_set = training_set
        if training_set is not None:
            model_name = 'pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb'
            self.symptom_encoder = SentenceTransformer(model_name)
            self.embeddings_cache = np.load('./data/mimic-iv-private/symptom_embeddings.npy',allow_pickle=True)
    
    def predict(self, *args, **kwargs):
        """
        Route prediction based on strategy.
        """
        if self.strategy == "FewShot" or self.strategy=="FewShotCoT":
            return self.few_shot_prediction(*args, **kwargs)
        if self.strategy == "KATE":
            return self.kate_prediction(*args, **kwargs)
        elif self.strategy == "ZeroShot" or self.strategy == "CoT":
            return self.zero_shot_prediction(*args, **kwargs)
        elif self.strategy == "SelfConsistency":
            return self.self_consistency_prediction(*args, **kwargs)
        else:
            raise ValueError(f"Unsupported strategy: {self.strategy}")

    def zero_shot_prediction(self, prompt, **kwargs):
        """Zero-shot prediction implementation."""
        if self.hippa:
            return utils.query_gpt_safe(prompt, **kwargs)
        return utils.query_llm(prompt, **kwargs)

    def _format_prompt_few_shots(self, test_prompt, serialize, serialization_strategy, instruction_prompt, k_shots):

        formatted_examples = "\n\n".join(
            f"Input: {serialize(ex, serialization_strategy)}\nOutput: {ex['acuity']}"
            for i, ex in k_shots.iterrows()
        )
        return f"{instruction_prompt}\n\n{formatted_examples}\n\nInput: {test_prompt}\nOutput:"
    
    def few_shot_prediction(self, test_prompt, serialization_func = None, instruction_prompt='', k_shots=5, **kwargs):
        #print(self.training_set)
        if self.training_set is None:
            raise ValueError("Few-shot strategy requires a training set of examples.")
        _, k_shots = utils.stratified_df(self.training_set, target_col=self.target,test_size=k_shots)
        prompt = self._format_prompt_few_shots(test_prompt, serialization_func, instruction_prompt, k_shots)
        if self.hippa:
            return utils.query_gpt_safe(prompt, **kwargs)
        else:
            return utils.query_llm(prompt, **kwargs)

    
    def kate_prediction(self, test_prompt, row ,serialization_func = None, serialization_strategy = 'spaces', instruction_prompt='', k_shots=5, **kwargs):
        #print(self.training_set)
        if self.training_set is None:
            raise ValueError("Few-shot strategy requires a training set of examples.")
        k_shots = self._retrieve_top_k_examples(row, k_shots)
        prompt = self._format_prompt_few_shots(test_prompt, serialization_func, serialization_strategy, instruction_prompt, k_shots)
        if self.hippa:
            return utils.query_gpt_safe(prompt, **kwargs)
        else:
            return utils.query_llm(prompt, **kwargs)

    def get_top_k_similar(self, embedding, embeddings, k):
        """
        Find the top-k most similar samples to a given embedding.
        """
        similarities = np.dot(embeddings, embedding)
        top_k_indices = np.argsort(similarities)[-k:][::-1]
        return top_k_indices, similarities[top_k_indices]

    def _retrieve_top_k_examples(self, row, k):
        """
        Retrieve the top K most similar examples to the prompt based on Ada embeddings.

        :param row: The input row containing chief complaint and vital signs.
        :param k: Number of examples to retrieve.
        :return: DataFrame of top K examples.
        """
        symptom = row['chiefcomplaint']
        symptom_embedding = self.symptom_encoder.encode(symptom, batch_size=1, convert_to_numpy=True)
        
        # First narrow down to k*3 samples by symptom
        top_k_indices, _ = self.get_top_k_similar(symptom_embedding, self.embeddings_cache, k*3)
        
        # Then, narrow down to k by vitals
        vital_sign_columns = ['temperature', 'heartrate', 'resprate', 'o2sat', 'sbp', 'dbp', 'pain']
        numeric_vital_signs = [col for col in vital_sign_columns if pd.api.types.is_numeric_dtype(row[col])]

        query_vital_signs = row[numeric_vital_signs].values.astype(float)
        top_25_vital_signs = self.training_set.loc[top_k_indices, numeric_vital_signs]
        top_25_vital_signs = top_25_vital_signs.apply(pd.to_numeric, errors='coerce')  # Coerce invalid values to NaN
        top_25_vital_signs = top_25_vital_signs.dropna()  # Drop rows with NaN values
        
        # Ensure the resulting values are numpy floats
        top_25_vital_signs = top_25_vital_signs.values.astype(float)
        query_vital_signs_normalized = query_vital_signs / np.linalg.norm(query_vital_signs)
        top_25_vital_signs_normalized = top_25_vital_signs / np.linalg.norm(top_25_vital_signs, axis=1, keepdims=True)
        
        vital_signs_similarity = np.dot(top_25_vital_signs_normalized, query_vital_signs_normalized)
        top_k_vital_indices = np.argsort(vital_signs_similarity)[-k:][::-1]
        
        # Retrieve top-k samples
        top_k_vital_samples = self.training_set.loc[top_k_indices[top_k_vital_indices]]
        
        # Add the 'pain' column back to the result (unchanged)
        top_k_vital_samples['pain'] = self.training_set.loc[top_k_indices[top_k_vital_indices], 'pain']
        
        return top_k_vital_samples

    def self_consistency_prediction(self, prompt):
        """Self-consistency prediction implementation."""
        # Add Self-Consistency-specific logic here
        pass
    
# if __name__ == "__main__":
#     # Example usage of the Predictor class
#     model = "gpt-3.5-turbo"
#     strategy = "ZeroShot"
#     target = "output"
#     training_set = [
#         {"input": "What is the capital of France?", "output": "Paris"},
#         {"input": "What is the capital of Germany?", "output": "Berlin"},
#         {"input": "What is the capital of Italy?", "output": "Rome"},
#     ]
#     prompt = "What is the capital of Spain?"
    
#     predictor = Predictor(strategy, target=target, training_set=training_set)
#     response = predictor.predict(prompt, model=model)
#     print(response) 