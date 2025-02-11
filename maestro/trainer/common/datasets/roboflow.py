import re
from typing import Optional

ROBOFLOW_PROJECT_TYPE_TO_DATASET_FORMAT = {
    "object-detection": "coco",
    "text-image-pairs": "jsonl",
}


def parse_roboflow_identifier(identifier: str) -> Optional[tuple[str, str, Optional[int]]]:
    """
    Parses a Roboflow identifier and extracts the workspace, project, and optional dataset version.

    Args:
        identifier (str): The Roboflow identifier, which can be a full URL or a partial identifier string.

    Returns:
        Optional[tuple[str, str, Optional[int]]]: A tuple in the form of (workspace_id, project_id, dataset_version)
            if the identifier is valid; otherwise, None.
    """
    identifier_no_protocol = re.sub(r"^https?://", "", identifier.strip())
    domain_pattern = r"^(?:[^/]*roboflow\.com)/?"
    identifier_no_domain = re.sub(domain_pattern, "", identifier_no_protocol)
    tokens = [segment for segment in identifier_no_domain.split("/") if segment]

    if len(tokens) < 2:
        return None

    workspace = tokens[0]
    project = tokens[1]
    version = None

    if len(tokens) > 3:
        return None
    elif len(tokens) == 3:
        try:
            version = int(tokens[2])
        except ValueError:
            return None

    return workspace, project, version
