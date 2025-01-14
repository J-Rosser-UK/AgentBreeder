import openai

from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

from dotenv import load_dotenv
from base import System
import numpy as np

load_dotenv(override=True)


class Descriptor:

    def __init__(self, model="text-embedding-3-small", output_dim=12):
        """
        Initializes the Descriptor class.

        Args:
            model (str): The name of the OpenAI embedding model to use for generating embeddings.
            output_dim (int): The dimensionality of the output embeddings.
        """
        self.client = openai.Client()
        self.model = model
        self.output_dim = output_dim

    def batch_generate(self, systems: list[System]):
        """
        Generates embeddings for a batch of systems using threading.

        Args:
            systems (list[System]): A list of system objects for which embeddings will be generated.

        Returns:
            np.ndarray: A NumPy array containing the embeddings for all systems in the batch.
        """
        with ThreadPoolExecutor(max_workers=16) as executor:
            embeddings = list(
                tqdm(
                    executor.map(self.generate, systems),
                    total=len(systems),
                    desc="Generating embeddings",
                )
            )
        return np.array(embeddings)

    def generate(self, system: System):
        """
        Generates an embedding for a single system.

        Args:
            system (System): The system object for which the embedding will be generated.

        Returns:
            list[float]: The embedding vector for the given system.
        """
        text = system.system_name + ": " + "\n" + system.system_code
        response = self.client.embeddings.create(
            input=text, model=self.model, dimensions=self.output_dim
        )
        return response.data[0].embedding
