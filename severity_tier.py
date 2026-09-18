from enum import Enum

class SeverityTier(Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"

def severity_from_scale(pain_score: int) -> SeverityTier:
    if pain_score <= 4:
        return SeverityTier.LOW
    elif pain_score <= 7:
        return SeverityTier.MODERATE
    else:
        return SeverityTier.HIGH
