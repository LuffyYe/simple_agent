"""
Safety Filtering Tool - Detect and block harmful queries
"""

from typing import Dict

class SafetyTool:
    """Safety filtering tool"""

    def __init__(
        self,
        name: str = "safety",
        description: str = "Safety filter to detect harmful, inappropriate, or policy-violating queries"
    ):
        self.name = name
        self.description = description

        # Safety policies with keywords and messages
        self.policies = {
            "violence": {
                "keywords": [
                    "kill", "hurt", "attack", "weapon", "bomb", "suicide", "self-harm",
                ],
                "message": "Content related to violence or harm is not allowed"
            },
            "illegal": {
                "keywords": [
                    "hack", "crack", "piracy", "scam", "drugs", "gambling",
                ],
                "message": "Content related to illegal activities is not allowed"
            },
            "inappropriate": {
                "keywords": [
                    "porn", "adult content", "hate speech", "discrimination", "abuse",
                ],
                "message": "Inappropriate or offensive content is not allowed"
            },
            "privacy": {
                "keywords": [
                    "password", "private key", "id number", "bank card", "phone number",
                ],
                "message": "Sensitive personal information is not allowed"
            },
            "company_confidential": {
                "keywords": [
                    "source code", "database password", "internal system",
                ],
                "message": "Potential company confidential information access is restricted"
            }
        }

    def run(self, params: Dict) -> Dict:
        """
        Check query safety

        Returns:
            {
                "safe": bool,
                "category": str or None,
                "message": str
            }
        """
        text = params.get("text", "")

        if not text:
            return {"safe": True, "category": None, "message": ""}

        text_lower = text.lower()

        for category, policy in self.policies.items():
            for keyword in policy["keywords"]:
                if keyword in text_lower:
                    return {
                        "safe": False,
                        "category": category,
                        "message": f"[BLOCKED] {policy['message']} (detected: {keyword})"
                    }

        return {"safe": True, "category": None, "message": ""}

    def get_policies(self) -> Dict:
        """Get all safety policies"""
        return {k: v["message"] for k, v in self.policies.items()}