import numpy as np
import utils.utils as utils

class Predictor:
    """Prediction strategy with multiple options including CoT, FewShot, ZeroShot, and SelfConsistency."""
    def __init__(self, model, strategy="ZeroShot", hippa=True, target=None, training_set=None, precomputed_embeddings=None):
        """
        Initialize the Predictor.

        :param model: The language model to use for predictions.
        :param strategy: Prediction strategy ("FewShot", "CoT", "ZeroShot", "SelfConsistency").
        :param debug: Whether to enable debug mode.
        :param training_set: The training dataset for few-shot prompting (list of dictionaries with 'input' and 'output').
        """
        self.model = model
        self.strategy = strategy
        self.target = target
        self.hippa = hippa
        self.training_set = training_set or []  # List of {'input': str, 'output': str}
        self.embeddings_cache = precomputed_embeddings

    def predict(self, *args, **kwargs):
        """
        Route prediction based on strategy.

        :param prompt: The input prompt for prediction.
        :return: Model response.
        """
        if self.strategy == "FewShot":
            return self.few_shot_prediction(*args, **kwargs)
        if self.strategy == "KATE":
            return self.kate_prediction(*args, **kwargs)
        elif self.strategy == "CoT":
            return self.chain_of_thought_prediction(*args, **kwargs)
        elif self.strategy == "ZeroShot":
            return self.zero_shot_prediction(*args, **kwargs)
        elif self.strategy == "SelfConsistency":
            return self.self_consistency_prediction(*args, **kwargs)
        else:
            raise ValueError(f"Unsupported strategy: {self.strategy}")

    def few_shot_prediction(self, test_sample, instruction_prompt, format, k=5):
        if not self.training_set:
            raise ValueError("Few-shot strategy requires a training set of examples.")
        k_shots = utils.stratified_sample_df(self.training_set, target_col=self.target,sample_size=k)
        prompt = format(instruction_prompt, test_sample, k_shots)
        response = utils.query_gpt_safe(prompt)
        
        
    def kate_prediction(self, prompt, k=5):
        """
        Few-shot prediction using KATE.

        :param prompt: The input prompt.
        :param k: Number of examples to retrieve for few-shot context.
        :return: Model response.
        """
        if not self.training_set:
            raise ValueError("Few-shot strategy requires a training set of examples.")
        
        # Retrieve top K similar examples
        top_k_examples = self._retrieve_top_k_examples(prompt, k)
        
        # Format the prompt with the retrieved examples
        formatted_prompt = self._format_prompt_with_kate(prompt, top_k_examples)
        
        # Call the KATE-specific prediction function
        return self._predict_kate(formatted_prompt)

    def _precompute_embeddings(self):
        """Precompute embeddings for the training set using Ada embeddings."""
        if self.debug:
            print("Precomputing embeddings for the training set...")
        embeddings_cache = []
        for example in self.training_set:
            input_text = example['input']
            embedding = get_embedding(input_text, model="text-embedding-ada-002")
            embeddings_cache.append(embedding)
        return embeddings_cache

    def _retrieve_top_k_examples(self, prompt, k):
        """
        Retrieve the top K most similar examples to the prompt based on Ada embeddings.

        :param prompt: The input prompt.
        :param k: Number of examples to retrieve.
        :return: List of top K examples.
        """
        # Compute the embedding for the prompt
        prompt_embedding = get_embedding(prompt, model="text-embedding-ada-002")
        
        # Compute similarity scores
        similarities = [
            cosine_similarity(prompt_embedding, example_embedding)
            for example_embedding in self.embeddings_cache
        ]
        
        # Get indices of top K most similar examples
        top_k_indices = np.argsort(similarities)[-k:][::-1]
        
        # Retrieve the corresponding examples
        top_k_examples = [self.training_set[i] for i in top_k_indices]
        if self.debug:
            print(f"Retrieved top {k} examples based on similarity scores.")
        return top_k_examples

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
        return utils.query_gpt(formatted_prompt, model=self.model, debug=self.debug)

    def chain_of_thought_prediction(self, prompt):
        """Chain-of-thought prediction implementation."""
        # Add CoT-specific logic here
        pass

    def zero_shot_prediction(self, prompt, **kwargs):
        """Zero-shot prediction implementation."""
        if self.hippa:
            return utils.query_gpt_safe(prompt, model=self.model, **kwargs)
        return utils.query_gpt(prompt, model=self.model, **kwargs)

    def self_consistency_prediction(self, prompt):
        """Self-consistency prediction implementation."""
        # Add Self-Consistency-specific logic here
        pass
    
if __name__ == "__main__":
    # Example usage of the Predictor class
    model = "gpt-3.5-turbo"
    strategy = "FewShot"
    target = "output"
    training_set = [
        {"input": "What is the capital of France?", "output": "Paris"},
        {"input": "What is the capital of Germany?", "output": "Berlin"},
        {"input": "What is the capital of Italy?", "output": "Rome"},
    ]
    prompt = "What is the capital of Spain?"
    
    predictor = Predictor(model, strategy, target=target, training_set=training_set)
    response = predictor.predict(prompt)
    print(response) 