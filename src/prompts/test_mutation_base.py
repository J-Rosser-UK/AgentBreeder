import sys

sys.path.append("src")


from base import initialize_session
import argparse
from prompts.meta_agent_base import get_base_prompt_with_archive


if __name__ == "__main__":

    for session in initialize_session():

        parser = argparse.ArgumentParser()

        parser.add_argument(
            "--population_id", type=str, default="547116b7-2b16-42fe-9b28-0c3f24ead54f"
        )

        args = parser.parse_args()

        prompt, response_format = get_base_prompt_with_archive(args, session)

        print(prompt)
