import numpy as np
import utils.utils as utils
import inspect

class Predictor:
    """
    The predictor class is soley responsible for handling different prediction strategies. 
    It does not format the prompt for each task and so 
    1) task-specific serialization of each sample 
    2) the task-specific instructions
    must be provided.
    """
    def __init__(self, dataset, strategy="ZeroShot", hippa=False, target=None, training_set=None, precomputed_embeddings=None):
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
        self.embeddings_cache = precomputed_embeddings
    
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

    def _format_prompt_few_shots(self, test_prompt, serialize, instruction_prompt, k_shots):

        formatted_examples = "\n\n".join(
            f"Input: {serialize(self.dataset,ex)}\nOutput: {ex['acuity']}"
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

    
    def kate_prediction(self, row, test_prompt, serialization_func = None, instruction_prompt='', k_shots=5, **kwargs):
        #print(self.training_set)
        if self.training_set is None:
            raise ValueError("Few-shot strategy requires a training set of examples.")
        _, k_shots = self._retrieve_top_k_examples(row, k_shots)
        prompt = self._format_prompt_few_shots(test_prompt, serialization_func, instruction_prompt, k_shots)
        if self.hippa:
            return utils.query_gpt_safe(prompt, **kwargs)
        else:
            return utils.query_llm(prompt, **kwargs)

    def _retrieve_top_k_examples(self, row, k):
        """
        Retrieve the top K most similar examples to the prompt based on Ada embeddings.

        :param prompt: The input prompt.
        :param k: Number of examples to retrieve.
        :return: List of top K examples.
        """

        return None

    def _format_prompt_with_kate(self, prompt, examples):
        """
        Format the prompt by incorporating examples following the KATE approach.

        :param prompt: The input prompt.
        :param examples: List of examples retrieved for few-shot prompting.
        :return: Formatted prompt string.
        """
        formatted_examples = "\n\n".join(
            f"Example {i + 1}:\nInput: {ex['input']}\nOutput: {ex['output']}"
            for i, ex in enumerate(examples)
        )
        return f"{formatted_examples}\n\nTask:\n{prompt}"

    def _predict_kate(self, formatted_prompt):
        """Internal method to call the model API with the formatted KATE prompt."""
        if self.debug:
            print("Executing KATE prediction with the following prompt:")
            print(formatted_prompt)
        return utils.query_llm(formatted_prompt, model=self.model, debug=self.debug)

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