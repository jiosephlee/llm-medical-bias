import numpy as np
import utils.utils as utils
import utils.prompts as prompts
from sentence_transformers import SentenceTransformer
import pandas as pd
import json

class Predictor:
    """
    The predictor class is responsible for handling different prediction strategies, including:
    0) appropriate prompt for the strategy
    1) task-specific serialization of each sample 
    2) the task-specific instructions
    3) strategy logic for each prediction
    """
    def __init__(self, dataset, strategy="Vanilla", hippa=False, target=None, training_set=None):
        """
        :param model: The LLM to use for predictions.
        :param strategy: Prediction strategy ("FewShot", "CoT", "Vanilla", "SelfConsistency").
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
            embeddings_path = utils._DATASETS[dataset]['training_embeddings_filepath']
            self.embeddings_cache = np.load(embeddings_path,allow_pickle=True)
    
    def predict(self, *, row, model, k_shots, return_json, serialization_strategy, vitals_off, bias, debug=False):
        """
        Route prediction based on strategy.
        """
        if self.strategy in ["FewShot", "FewShotCoT"]:
            return self.few_shot_prediction(row=row, model=model, k_shots=k_shots, return_json=return_json, serialization_strategy=serialization_strategy, vitals_off=vitals_off, bias=bias, debug=debug)
        elif self.strategy in ["KATE", "KATECoT"]:
            return self.kate_prediction(row=row, model=model, k_shots=k_shots, return_json=return_json, serialization_strategy=serialization_strategy, vitals_off=vitals_off, bias=bias, debug=debug)
        elif self.strategy in ["Vanilla", "Vanillav0", "AutoCoT", "CoT"]:
            return self.zero_shot_prediction(row=row, model=model, return_json=return_json, serialization=serialization_strategy, vitals_off=vitals_off, bias=bias, debug=debug)
        elif self.strategy in ["SelfConsistency"]:
            return self.self_consistency_prediction(row=row, model=model, return_json=return_json, serialization=serialization_strategy, vitals_off=vitals_off, bias=bias, debug=debug)
        else:
            raise ValueError(f"Unsupported strategy: {self.strategy}")

    def zero_shot_prediction(self, *, row, model, return_json=False, serialization='natural', vitals_off=False, bias=False, debug=False, **kwargs):
        """Zero-shot prediction implementation."""

        prompt = prompts.format_instruction_prompt_for_blackbox(
            row=row,
            strategy=self.strategy,
            dataset=self.dataset.lower(),
            return_json=return_json,
            serialization=serialization
        )
        
        if self.hippa:
            response = utils.query_gpt_safe(prompt, model=model, return_json=return_json, debug=debug, is_prompt_full=True)
        else:
            response = utils.query_llm_full(prompt, model=model, return_json=return_json, debug=debug)
        
        return prompt, response

    def few_shot_prediction(self, *, row, model, k_shots=5, return_json=False, serialization='natural', vitals_off=False, bias=False, debug=False, **kwargs):
        """Few-shot prediction implementation."""
        if self.training_set is None:
            raise ValueError("Few-shot strategy requires a training set of examples.")
        
        _, examples = utils.stratified_df(self.training_set, target_col=self.target, test_size=k_shots)
        
        formatted_examples = "\n\n".join([
            prompts.format_row(example, dataset=self.dataset.lower(), serialization=serialization) + 
            f"\nAcuity Level: {example[self.target]}"
            for _, example in examples.iterrows()
        ])
        
        prompt = prompts.format_instruction_prompt_for_blackbox(
            row=row,
            strategy=self.strategy,
            dataset=self.dataset.lower(),
            return_json=return_json,
            serialization=serialization,
            examples=formatted_examples
        )
        
        if self.hippa:
            response = utils.query_gpt_safe(prompt, model=model, return_json=return_json, debug=debug, is_prompt_full=True)
        else:
            response = utils.query_llm_full(prompt, model=model, return_json=return_json, debug=debug)
        
        return prompt, response

    def self_consistency_prediction(self, *, row, model, return_json=False, serialization='natural', vitals_off=False, bias=False, debug=False, num_trials=5, **kwargs):
        """Self-consistency prediction implementation."""
        responses = []
        
        for _ in range(num_trials):
            prompt = prompts.format_instruction_prompt_for_blackbox(
                row=row,
                strategy='Vanilla',
                dataset=self.dataset.lower(),
                return_json=True,
                serialization=serialization
            )
            
            if self.hippa:
                response = utils.query_gpt_safe(prompt, model=model, return_json=True, debug=debug, is_prompt_full=True)
            else:
                response = utils.query_llm_full(prompt, model=model, return_json=True, debug=debug)
            
            try:
                response_data = json.loads(response)
                responses.append(response_data['Acuity'])
            except (json.JSONDecodeError, KeyError) as e:
                if debug:
                    print(f"Error parsing response: {e}")
                    print(f"Raw response: {response}")
                continue
        
        if not responses:
            raise ValueError("No valid responses received from any trial")
        
        from statistics import mode
        final_acuity = mode(responses)
        
        if return_json:
            final_response = json.dumps({
                "Acuity": final_acuity,
                "Reasoning": f"Based on {len(responses)} trials, the most common prediction was {final_acuity}. All predictions: {responses}"
            })
        else:
            final_response = f"Based on {len(responses)} trials, the most common prediction was {final_acuity}. All predictions: {responses}"
        
        return prompt, final_response

    def kate_prediction(self, *, row, model, k_shots=5, return_json=False, serialization='natural', vitals_off=False, bias=False, debug=False, **kwargs):
        """KATE prediction implementation."""
        if self.training_set is None:
            raise ValueError("KATE strategy requires a training set of examples.")
        
        examples = self._retrieve_top_k_examples(row, k_shots)
        
        formatted_examples = "\n\n".join([
            prompts.format_row(example, dataset=self.dataset.lower(), serialization=serialization) + 
            f"\nAcuity Level: {example[self.target]}"
            for _, example in examples.iterrows()
        ])
        
        prompt = prompts.format_instruction_prompt_for_blackbox(
            row=row,
            strategy=self.strategy,
            dataset=self.dataset.lower(),
            return_json=return_json,
            serialization=serialization,
            examples=formatted_examples
        )
        
        if self.hippa:
            response = utils.query_gpt_safe(prompt, model=model, return_json=return_json, debug=debug, is_prompt_full=True)
        else:
            response = utils.query_llm_full(prompt, model=model, return_json=return_json, debug=debug)
        
        return prompt, response
    
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
    
# if __name__ == "__main__":
#     # Example usage of the Predictor class
#     model = "gpt-3.5-turbo"
#     strategy = "Vanilla"
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