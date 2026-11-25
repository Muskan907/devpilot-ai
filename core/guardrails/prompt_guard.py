"""Prompt Injection and Input Validation Guardrails."""

import re
from typing import List, Tuple
from core.schemas.guardrails import SecurityViolation, ViolationLevel


# Known prompt injection and jailbreak heuristic patterns
INJECTION_PATTERNS = [
    (r"(?i)ignore\s+(all\s+)?(previous|prior|above)\s+instructions", "Prompt Injection: Instruction override attempt"),
    (r"(?i)system\s+prompt\s+(leak|reveal|display|output|print)", "Prompt Injection: System prompt extraction attempt"),
    (r"(?i)you\s+are\s+now\s+in\s+DAN\s+mode", "Prompt Injection: Jailbreak pattern (DAN mode)"),
    (r"(?i)disregard\s+(all\s+)?safety\s+guidelines", "Prompt Injection: Safety guideline bypass"),
    (r"(?i)act\s+as\s+an\s+unrestricted\s+ai", "Prompt Injection: Persona hijacking"),
    (r"(?i)developer\s+mode\s+enabled", "Prompt Injection: Fake developer mode injection"),
]

# Toxic / abusive language directed at people. Patterns target abuse aimed at a person
# (insults, threats, harassment) so that legitimate engineering wording such as
# "kill the worker process" or "abort the job" is not blocked.
TOXIC_PATTERNS = [
    (r"(?i)\b(f+u+c+k+|screw)\s+(you|off|u)\b", "Toxic Language: Profane abuse"),
    (r"(?i)\byou(\s+are|'re|\s+r)?\s+(an?\s+)?(stupid|idiot|moron|dumb|useless|pathetic|worthless)\b",
     "Toxic Language: Insult directed at a person"),
    (r"(?i)\b(stupid|idiotic|useless|dumb)\s+(bot|ai|assistant|developer|team)\b", "Toxic Language: Insult"),
    (r"(?i)\b(i\s+will|i'll|gonna|going\s+to)\s+(kill|hurt|attack|destroy)\s+(you|him|her|them)\b",
     "Toxic Language: Threat of violence"),
    (r"(?i)\bkill\s+(yourself|urself)\b", "Toxic Language: Self-harm encouragement"),
    (r"(?i)\b(shut\s+up|go\s+to\s+hell)\b", "Toxic Language: Harassment"),
    (r"(?i)\bi\s+hate\s+(all\s+)?(women|men|muslims|hindus|christians|jews|blacks|whites|gays|immigrants)\b",
     "Toxic Language: Hate speech"),
]


class PromptGuardrail:
    """Validates developer inputs against prompt injections, toxic language and malicious instructions."""

    def __init__(self):
        self.compiled_rules = [(re.compile(p), desc) for p, desc in INJECTION_PATTERNS]
        self.toxic_rules = [(re.compile(p), desc) for p, desc in TOXIC_PATTERNS]

    def validate_prompt(self, raw_prompt: str) -> Tuple[bool, List[SecurityViolation]]:
        """Scans prompt for injections and malformed inputs."""
        violations: List[SecurityViolation] = []

        if not raw_prompt or len(raw_prompt.strip()) < 5:
            violations.append(SecurityViolation(
                rule_name="InputLengthRule",
                severity=ViolationLevel.CRITICAL,
                details="Prompt is empty or too short (minimum 5 characters required).",
            ))
            return False, violations

        if len(raw_prompt) > 10000:
            violations.append(SecurityViolation(
                rule_name="InputLengthRule",
                severity=ViolationLevel.WARNING,
                details="Prompt exceeds maximum recommended length of 10,000 characters.",
            ))

        for pattern, description in self.compiled_rules:
            match = pattern.search(raw_prompt)
            if match:
                violations.append(SecurityViolation(
                    rule_name="PromptInjectionRule",
                    severity=ViolationLevel.CRITICAL,
                    details=f"{description}: matched '{match.group(0)}'",
                    target_snippet=match.group(0),
                ))

        for pattern, description in self.toxic_rules:
            match = pattern.search(raw_prompt)
            if match:
                violations.append(SecurityViolation(
                    rule_name="ToxicLanguageRule",
                    severity=ViolationLevel.CRITICAL,
                    details=f"{description}: matched '{match.group(0)}'",
                    target_snippet=match.group(0),
                ))

        passed = all(v.severity != ViolationLevel.CRITICAL for v in violations)
        return passed, violations


# Global singleton
prompt_guardrail = PromptGuardrail()

