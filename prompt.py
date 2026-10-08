import re

def create_prompt(description):
    """
    Converts witness description into a prompt
    for realistic face generation.
    """

    description = description.lower()

    prompt = (
        "A realistic forensic portrait, passport style, "
        "front facing, highly detailed human face, "
        "neutral background, "
        + description
    )

    return prompt