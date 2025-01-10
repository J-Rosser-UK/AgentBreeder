import sys

sys.path.append("src")


from base import initialize_session
import argparse
from prompts.meta_agent_base import get_base_prompt_with_archive


if __name__ == "__main__":

    session, Base = initialize_session()

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--population_id", type=str, default="b8128583-6bfb-46da-b7f8-9bc2b7d24a75"
    )

    args = parser.parse_args()

    prompt, response_format = get_base_prompt_with_archive(args, session)
